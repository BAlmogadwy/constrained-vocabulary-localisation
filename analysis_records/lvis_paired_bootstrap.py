"""Exact paired image bootstrap for official LVIS semantics, using cached matches."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
OUT = Path(__file__).resolve().parent
from review_paths import output_path
import argparse
import collections
import copy
import hashlib
import json
import time
import zipfile
import numpy as np
import lvis_semantics_sensitivity as ls
ra = ls.ra


class LVISReplicateBootstrap(ra.ReplicateBootstrap):
    def __init__(self, ev):
        self.ids = ev.params.img_ids
        self.recall = ev.params.rec_thrs
        self.T = len(ev.params.iou_thrs)
        self.groups = []
        n, na = len(self.ids), len(ev.params.area_rng)
        for k in range(len(ev.params.cat_ids)):
            scores, tps, fps, slices, gts = [], [], [], [], []
            offset = 0
            for i in range(n):
                e = ev.eval_imgs[k * na * n + i]
                if e is None:
                    slices.append(np.empty(0, dtype=int)); gts.append(0)
                    continue
                # LVISResults already applied global 300/image before matching.
                # Do not impose COCO's additional 100/category truncation.
                sc = np.asarray(e["dt_scores"])
                match = np.asarray(e["dt_matches"], dtype=bool)
                ignored = np.asarray(e["dt_ignore"], dtype=bool)
                scores.append(sc); tps.append(match & ~ignored); fps.append(~match & ~ignored)
                slices.append(np.arange(offset, offset + len(sc)))
                offset += len(sc)
                gts.append(np.count_nonzero(np.asarray(e["gt_ignore"]) == 0))
            if scores:
                self.groups.append((np.concatenate(scores), np.concatenate(tps, axis=1),
                                    np.concatenate(fps, axis=1), slices, np.asarray(gts)))


def official_eval(gt, dets):
    results = ls.LVISResults(gt, copy.deepcopy(dets), max_dets=300)
    evaluator = ls.LVISEval(gt, results, "bbox")
    evaluator.run()
    return evaluator


def materialized_ap(gt, dets, ids, counts):
    dataset = {"images": [], "annotations": [], "categories": copy.deepcopy(gt.dataset["categories"])}
    by_image = collections.defaultdict(list)
    for d in dets:
        by_image[d["image_id"]].append(d)
    outd = []
    aid = 0
    for iid, w in zip(ids, counts):
        for _ in range(int(w)):
            new_id = len(dataset["images"]) + 1
            # Metadata for negative and non-exhaustive categories is copied too.
            im = copy.deepcopy(gt.imgs[iid]); im["id"] = new_id; dataset["images"].append(im)
            for a in gt.img_ann_map[iid]:
                aa = copy.deepcopy(a); aid += 1; aa.update(id=aid, image_id=new_id)
                dataset["annotations"].append(aa)
            for d in by_image[iid]:
                dd = dict(d); dd["image_id"] = new_id; outd.append(dd)
    duplicate_gt = copy.deepcopy(gt)
    duplicate_gt.dataset = dataset
    duplicate_gt._create_index()
    return float(official_eval(duplicate_gt, outd).get_results()["AP"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    started = time.perf_counter()
    annotation = ra.ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"
    c = ra.coco_load(annotation)
    gt = ls.LVIS(str(annotation))
    ids = sorted(c.imgs)
    predictions = {}
    folder = ra.ROOT / "results/raw/traditional_lowthr/lvis_v1_unseen/per_image/yolo_world"
    recs = []
    for path in sorted(folder.glob("*.json")):
        item = ra.read(path); recs.append((path.stem, item.get("result", item).get("detections", [])))
    assert len(recs) == len(ids)
    predictions["yolo_world"] = ra.as_coco(c, recs, family="traditional")
    archive = ra.ROOT / "results/ablation_normalization.zip"; ra.digest(archive)
    with zipfile.ZipFile(archive) as z:
        for model in ["gemini-3.5-flash", "gemini-3.1-pro"]:
            names = sorted(n for n in z.namelist() if f"/ON_fromraw/lvis/per_image/{model}/" in n and n.endswith(".json"))
            assert len(names) == len(ids)
            recs = []
            for name in names:
                item = json.loads(z.read(name)); stem = Path(name).stem; im = c.imgs[int(stem)]
                labels = {ra.CM.label_lookup_key(x) for x in item["labels"]}
                parsed = ra.PARSER.parse_detections_from_text(ra.text_from_raw(item.get("raw_response"), model))
                normalized = ra.PARSER.normalize_detections_to_image(parsed, (im["width"], im["height"]), ra.MODES[model])
                recs.append((stem, [d for d in normalized if ra.CM.label_lookup_key(d["label"]) in labels]))
            predictions[model] = ra.as_coco(c, recs, uniform=True)
    prior = ra.read(OUT / "lvis_semantics_sensitivity.json")
    evaluators = {m: official_eval(gt, d) for m, d in predictions.items()}
    fast = {m: LVISReplicateBootstrap(e) for m, e in evaluators.items()}
    n = len(ids)
    validation = {}
    validation_counts = np.random.default_rng(7421).multinomial(n, np.full(n, 1/n))
    for m, f in fast.items():
        point = f.ap(np.ones(n, dtype=int))
        expected = float(evaluators[m].get_results()["AP"])
        assert abs(point - expected) < 1e-12
        assert abs(expected - prior["models"][m]["official_lvis"]["metrics"]["AP"]) < 1e-12
        materialized = materialized_ap(gt, predictions[m], ids, validation_counts)
        cached = f.ap(validation_counts)
        assert abs(materialized - cached) < 1e-12, (m, materialized, cached)
        validation[m] = {"unit_weights_error": abs(point - expected),
                         "materialized_duplicate_images_error": abs(materialized - cached),
                         "materialized_validation_ap": materialized}
    bootstrap_start = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    samples = {m: [] for m in fast}
    for _ in range(args.bootstrap):
        counts = rng.multinomial(n, np.full(n, 1/n))
        for m, f in fast.items():
            samples[m].append(f.ap(counts))
    summary = {}
    for m in fast:
        s = np.asarray(samples[m])
        point = float(evaluators[m].get_results()["AP"])
        summary[m] = {"ap": point, "percentile_95_ci": np.quantile(s, [.025, .975]).tolist()}
        if m != "yolo_world":
            delta = s - np.asarray(samples["yolo_world"])
            summary[m].update(delta_vs_yolo=point - summary["yolo_world"]["ap"],
                              paired_delta_percentile_95_ci=np.quantile(delta, [.025, .975]).tolist(),
                              positive_delta_fraction=float(np.mean(delta > 0)))
    difference = np.asarray(samples["gemini-3.5-flash"]) - np.asarray(samples["gemini-3.1-pro"])
    output = {"status": "separate new official-LVIS paired bootstrap; frozen COCO replay unchanged",
              "B": args.bootstrap, "seed": args.seed, "n_images": n, "categories": len(c.cats),
              "method": "Paired nonparametric image bootstrap using multinomial image counts; exact full detection-sequence replication, source-image ID order, stable score sorting. Official LVIS 0.5.3 bbox matching, retained federated negative/non-exhaustive metadata, global 300 detections per image, IoU 0.50:0.05:0.95, all areas, 101 recall thresholds. Category macro AP omits categories with no nonignored ground truth in a replicate. Percentile intervals. Confidence intervals are conditional on these images and cached predictions; no model-call or prompt randomness resampled.",
              "notes": ["All accepted Gemini detections have uniform score 1.0; YOLO retains archived native scores at inference floor 0.001.",
                        "The complete per-image detection sequence is repeated for every image copy; TP/FP weighted-count shortcuts are not used, preserving tied-score behavior.",
                        "Materialized validation duplicates annotations including federated image metadata and reruns the official LVIS evaluator; validation seed 7421.",
                        "Positive-delta fraction is descriptive, not an acceptance probability or a hypothesis-test p-value.",
                        "The point estimates and intervals are for this constrained 10-category subset, not full LVIS or established pretraining-unseen generalization."],
              "models": summary, "flash_minus_pro": {"delta": summary["gemini-3.5-flash"]["ap"] - summary["gemini-3.1-pro"]["ap"],
                  "paired_delta_percentile_95_ci": np.quantile(difference, [.025, .975]).tolist()},
              "validation": validation, "bootstrap_seconds": time.perf_counter()-bootstrap_start,
              "total_seconds": time.perf_counter()-started}
    for path in sorted(Path(ls.lvis.__file__).resolve().parent.glob("*.py")):
        ra.digest(path)
    for name in ["replay_audit.py", "lvis_semantics_sensitivity.py"]:
        ra.digest(OUT / name)
    output["input_sha256"] = ra.INPUTS
    output["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output_path("lvis_paired_bootstrap.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({k: output[k] for k in ["B", "models", "flash_minus_pro", "validation", "bootstrap_seconds", "total_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
