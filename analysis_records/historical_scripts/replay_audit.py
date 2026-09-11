"""Offline, read-only source replay for the Neurocomputing revision.

Only this directory receives output. No API clients, credentials, models, or
network calls are used. Run: python replay_audit.py --bootstrap 1000
"""
from __future__ import annotations
import argparse
import collections
import contextlib
import copy
import hashlib
import importlib.util
import importlib.metadata
import io
import json
import platform
import time
import types
import zipfile
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent.parent / "zero-shot-detection-benchmark"
MODES = {
    "gemini-3.5-flash": "normalized_1000_yxyx",
    "gemini-3.1-pro": "normalized_1000_yxyx",
    "qwen3-vl-235b-openrouter": "normalized_1000_xyxy",
    "gemma-3-27b-openrouter": "normalized_1000_xyxy",
    "mistral-large-openrouter": "normalized_1000_xyxy",
}
INPUTS = {}

def digest(path):
    path = Path(path)
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[str(path)] = h
    return h

def read(path):
    digest(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))

def module(name, path):
    digest(path)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

CM = module("replay_compute_metrics", ROOT / "experiments/compute_metrics.py")
PARSER = module("replay_parser", ROOT / "models/vlm/parsing.py")
_parser_source = (ROOT / "models/vlm/parsing.py").read_text(encoding="utf-8")
_gate = "if parsed_score <= 0:\n                return None"
assert _parser_source.count(_gate) == 1, "Historical score gate changed; review audit probe"
PARSER_KEEP_NONPOSITIVE = types.ModuleType("replay_parser_score_gate_probe")
exec(compile(_parser_source.replace(_gate, "if False:  # audit probe only\n                return None"),
             "<in-memory score gate probe>", "exec"), PARSER_KEEP_NONPOSITIVE.__dict__)

@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield

def coco_load(path):
    d = read(path)
    for a in d["annotations"]:
        a.setdefault("iscrowd", 0)
    c = COCO()
    c.dataset = d
    with quiet():
        c.createIndex()
    return c

def mappings(c):
    return {CM.label_lookup_key(x["name"]): x["id"] for x in c.loadCats(c.getCatIds())}

def text_from_raw(raw, model):
    if isinstance(raw, str):
        return raw
    if not isinstance(raw, dict):
        return ""
    if model.startswith("gemini"):
        resp = raw.get("response", raw)
        # Same textual response parts as the existing batch collector. Thought
        # signatures are opaque provider metadata, not output or instructions.
        return "\n".join(
            part["text"]
            for candidate in resp.get("candidates", [])
            for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part, dict) and isinstance(part.get("text"), str)
            and not part.get("thought", False)
        )
    return PARSER.extract_openai_compatible_text(raw)

def evaluate(c, dets, ids, cats=None):
    if not dets:
        return {"ap": 0.0, "ap50": 0.0, "detections": 0}, None
    with quiet():
        dt = c.loadRes(dets)
        ev = COCOeval(c, dt, "bbox")
        ev.params.imgIds = sorted(ids)
        ev.params.catIds = sorted(cats if cats is not None else c.getCatIds())
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {"ap": float(ev.stats[0]), "ap50": float(ev.stats[1]), "detections": len(dets)}, ev

def as_coco(c, records, family="vlm", uniform=False):
    pred = CM.ModelPredictions(family, "lvis-unseen", "audit")
    for stem, dd in records:
        pred.image_stems.add(stem)
        pred.detections.extend({"image_stem": stem, "raw": d} for d in dd)
    dets = CM.prepare_coco_detections(pred, c, mappings(c))
    if uniform:
        for d in dets:
            d["score"] = 1.0
    return dets

class ReplicateBootstrap:
    """Exact image replication of fixed COCO matching, preserving score ties.

    Resampled images are sorted by source ID, with each duplicate keeping the
    full within-image detection sequence. This is more faithful than multiplying
    TP/FP counts at each detection when equal scores occur.
    """
    def __init__(self, ev):
        self.ids = ev.params.imgIds
        self.recall = ev.params.recThrs
        self.T = len(ev.params.iouThrs)
        self.groups = []
        n, na = len(self.ids), len(ev.params.areaRng)
        for k in range(len(ev.params.catIds)):
            scores, tps, fps, slices, gts = [], [], [], [], []
            offset = 0
            for i in range(n):
                e = ev.evalImgs[k * na * n + i]  # area=all
                if e is None:
                    slices.append(np.empty(0, dtype=int)); gts.append(0)
                    continue
                sc = np.asarray(e["dtScores"][:100])
                match = np.asarray(e["dtMatches"][:, :100], dtype=bool)
                ignored = np.asarray(e["dtIgnore"][:, :100], dtype=bool)
                scores.append(sc)
                tps.append(match & ~ignored)
                fps.append(~match & ~ignored)
                slices.append(np.arange(offset, offset + len(sc)))
                offset += len(sc)
                gts.append(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
            if not scores:
                continue
            self.groups.append((np.concatenate(scores), np.concatenate(tps, axis=1),
                                np.concatenate(fps, axis=1), slices, np.asarray(gts)))

    def ap(self, counts):
        category_aps = []
        for scores, tp, fp, slices, gts in self.groups:
            npositive = int(gts @ counts)
            if not npositive:
                continue
            idx = np.concatenate([np.tile(s, int(w)) for s, w in zip(slices, counts) if w])
            if not len(idx):
                category_aps.append(0.0)
                continue
            order = idx[np.argsort(-scores[idx], kind="mergesort")]
            tpc = np.cumsum(tp[:, order], axis=1, dtype=float)
            fpc = np.cumsum(fp[:, order], axis=1, dtype=float)
            rc = tpc / npositive
            pr = tpc / (tpc + fpc + np.spacing(1))
            pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
            q = np.zeros((self.T, len(self.recall)))
            for t in range(self.T):
                indexes = np.searchsorted(rc[t], self.recall, side="left")
                valid = indexes < pr.shape[1]
                q[t, valid] = pr[t, indexes[valid]]
            category_aps.append(float(q.mean()))
        return float(np.mean(category_aps)) if category_aps else 0.0

def materialized_ap(c, dets, ids, counts, cats):
    dataset = {"images": [], "annotations": [], "categories": copy.deepcopy(c.dataset["categories"])}
    det_by = collections.defaultdict(list)
    for d in dets:
        det_by[d["image_id"]].append(d)
    outd = []
    aid = 0
    for iid, w in zip(ids, counts):
        for _ in range(int(w)):
            new_id = len(dataset["images"]) + 1
            im = dict(c.imgs[iid]); im["id"] = new_id; dataset["images"].append(im)
            for a in c.imgToAnns[iid]:
                aa = dict(a); aid += 1; aa.update(id=aid, image_id=new_id)
                dataset["annotations"].append(aa)
            for d in det_by[iid]:
                dd = dict(d); dd["image_id"] = new_id; outd.append(dd)
    cc = COCO(); cc.dataset = dataset
    with quiet():
        cc.createIndex()
    return evaluate(cc, outd, list(cc.imgs), cats)[0]["ap"]

def paired_bootstrap(c, models, ids, cats, B, seed, validate=False):
    fast = {m: ReplicateBootstrap(ev) for m, (dets, ev) in models.items()}
    n = len(ids)
    validation = {}
    for m, f in fast.items():
        point = f.ap(np.ones(n, dtype=int))
        expected = float(models[m][1].stats[0])
        assert abs(point - expected) < 1e-12, (m, point, expected)
        validation[m] = {"all_weights_one_absolute_error": abs(point - expected)}
    if validate:
        counts = np.random.default_rng(7421).multinomial(n, np.full(n, 1 / n))
        for m in list(models)[:2]:
            exact = materialized_ap(c, models[m][0], ids, counts, cats)
            approx = fast[m].ap(counts)
            assert abs(exact - approx) < 1e-12, (m, exact, approx)
            validation[m]["materialized_duplicate_images_absolute_error"] = abs(exact - approx)
    rng = np.random.default_rng(seed)
    samples = {m: [] for m in models}
    start = time.perf_counter()
    for b in range(B):
        counts = np.bincount(rng.integers(0, n, size=n), minlength=n)
        for m, f in fast.items():
            samples[m].append(f.ap(counts))
        if b == 19:
            estimated = (time.perf_counter() - start) * B / 20
            print(f"Bootstrap benchmark: {n} images, {len(models)} models; estimated {estimated:.1f}s for B={B}", flush=True)
        if (b + 1) % 200 == 0:
            print(f"Bootstrap {b+1}/{B}", flush=True)
    reference = next(iter(models))
    summary = {}
    for m, a in samples.items():
        a = np.asarray(a)
        delta = a - np.asarray(samples[reference])
        summary[m] = {"point_ap": float(models[m][1].stats[0]),
                      "ap_ci95": np.percentile(a, [2.5, 97.5]).tolist(),
                      "paired_difference_vs_reference": float(models[m][1].stats[0] - models[reference][1].stats[0]),
                      "paired_difference_ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
                      "superiority_proportion": float(np.mean(delta > 0))}
    return {"B": B, "seed": seed, "reference": reference, "n_images": n,
            "method": "paired image resampling with replacement; exact duplicate detection sequence; canonical source-image ordering; percentile intervals; COCO macro average over categories with nonignored ground truth in each resample; IoU 0.50:0.95, area=all, maxDets=100 per category/image",
            "validation": validation, "seconds": time.perf_counter() - start, "models": summary}

def main():
    args = argparse.ArgumentParser()
    args.add_argument("--bootstrap", type=int, default=1000)
    args = args.parse_args()
    start = time.perf_counter()
    c = coco_load(ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json")
    ids = sorted(c.imgs)
    dyn = ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json"
    digest(dyn)
    legacy = read(ROOT / "results/metrics_summary.json")
    result = {"status": "new offline reevaluation, not reproduction of every legacy aggregate",
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "pycocotools": importlib.metadata.version("pycocotools")},
              "dataset": {"lvis_images": len(ids), "lvis_annotations": len(c.anns), "lvis_categories": len(c.cats),
                          "maximum_image_width": max(i["width"] for i in c.imgs.values()),
                          "maximum_image_height": max(i["height"] for i in c.imgs.values())},
              "vlm": {}, "detectors": {}, "router": {}, "notes": [
                  "No paid or network model calls; source data are never overwritten.",
                  "Raw VLM outputs are reparsed once with the manuscript's specified per-model convention, clipped to actual image dimensions, and restricted to each image's candidate labels.",
                  "Uniform-score AP sets every detection accepted by the fixed historical parser to 1.0; supplied-score AP preserves model score/confidence, with missing values defaulting to 1.0. The historical parser first rejects score/confidence <= 0; a separate in-memory sensitivity probe counts changes if only this gate is removed.",
                  "Scores are not calibrated probabilities. Category coverage is not recognition accuracy.",
                  "All 396 images are scored, including empty outputs and two Gemini string-format pilot responses.",
                  "Three OpenRouter conventions are inherited assumptions from the submitted manuscript, not independently established provider facts.",
                  "The original LVIS/COCO subsets overlap; the revised 868-image COCO router analysis excludes all 33 overlaps.",
                  "COCO overlap exclusion is a post hoc evaluation repair; label/category tasks differ across datasets.",
                  "Router VLM-only primary scores are uniform 1.0, matching run_live_hybrid.vlm_to_simple; supplied-score sensitivity is reported separately.",
                  "Only five of the twelve current VLM roster models have retained full LVIS raw outputs in the located archive. Other legacy headline aggregates are not promoted to reproduced results."]}
    archive = ROOT / "results/ablation_normalization.zip"; digest(archive)
    headline = {}
    with zipfile.ZipFile(archive) as z:
        for model, mode in MODES.items():
            names = sorted(n for n in z.namelist() if f"/ON_fromraw/lvis/per_image/{model}/" in n and n.endswith(".json"))
            assert len(names) == len(ids), (model, len(names))
            data = [json.loads(z.read(n)) for n in names]
            records = {"stored_arrays": [], "raw_correct_decoder": [], "raw_pixel_decoder": []}
            raw_types = collections.Counter()
            score_gate_extra = 0
            for name, d in zip(names, data):
                stem = Path(name).stem; iid = int(stem); image = c.imgs[iid]
                labels = {CM.label_lookup_key(x) for x in d["labels"]}
                raw_types[type(d.get("raw_response")).__name__] += 1
                raw = PARSER.parse_detections_from_text(text_from_raw(d.get("raw_response"), model))
                score_gate_probe = PARSER_KEEP_NONPOSITIVE.parse_detections_from_text(text_from_raw(d.get("raw_response"), model))
                score_gate_extra += len(score_gate_probe) - len(raw)
                for condition, dd in [("stored_arrays", d.get("detections", [])),
                                       ("raw_correct_decoder", PARSER.normalize_detections_to_image(raw, (image["width"], image["height"]), mode)),
                                       ("raw_pixel_decoder", PARSER.normalize_detections_to_image(raw, (image["width"], image["height"]), "pixel"))]:
                    dd = [v for v in dd if CM.label_lookup_key(v["label"]) in labels]
                    records[condition].append((stem, dd))
            r = {"images": len(data), "coordinate_mode": mode, "raw_response_types": dict(raw_types),
                 "nonpositive_score_gate_probe_extra_geometrically_valid_parsed_boxes": score_gate_extra,
                 "source": str(archive), "archive_member_prefix": f"ablation_normalization/ON_fromraw/lvis/per_image/{model}/",
                 "legacy_aggregate": next((x for x in legacy["lvis-unseen"] if x["model"] == model), None), "conditions": {}}
            for condition, recs in records.items():
                r["conditions"][condition] = {}
                for scoring in ["supplied", "uniform"]:
                    dets = as_coco(c, recs, uniform=scoring == "uniform")
                    metric, ev = evaluate(c, dets, ids)
                    r["conditions"][condition][scoring] = metric
                    if condition == "raw_correct_decoder" and scoring == "uniform" and model.startswith("gemini"):
                        headline[model] = (dets, ev)
            result["vlm"][model] = r
            print(model, r["conditions"]["raw_correct_decoder"], flush=True)
    detector_models = {}
    for m in ["yolo_world", "grounding_dino", "owl_vit"]:
        folder = ROOT / "results/raw/traditional_lowthr/lvis_v1_unseen/per_image" / m
        records = []
        for f in sorted(folder.glob("*.json")):
            d = read(f); block = d.get("result", d)
            records.append((f.stem, block.get("detections", [])))
        assert len(records) == len(ids)
        dets = as_coco(c, records, family="traditional")
        metric, ev = evaluate(c, dets, ids)
        result["detectors"][m] = {"score_floor": 0.001, "images": len(records), "source": str(folder), **metric}
        detector_models[m] = (dets, ev)
        print(m, metric, flush=True)
    if args.bootstrap:
        result["paired_lvis_bootstrap_uniform_vlm"] = paired_bootstrap(
            c, {"yolo_world": detector_models["yolo_world"], **headline}, ids, list(c.cats), args.bootstrap, 12345, validate=True)
    coco = coco_load(ROOT / "data/raw/coco2017/annotations/instances_val2017.json")
    dcpath = ROOT / "data/processed/coco2017/val/coco_unseen_dynamic_labels.json"; digest(dcpath)
    scope = CM.load_eval_scope(dcpath, "coco2017-val", coco, mappings(coco))
    router_boot = {}
    for ds, cc, cats in [("lvis", c, set(c.cats)), ("coco", coco, scope.cat_ids)]:
        rd = ROOT / "results/raw/live_hybrid" / ds
        records = [read(f) for f in sorted((rd / "per_image").glob("*.json"))]
        vrecords = [read(f) for f in sorted((rd / "vlm_only").glob("*.json"))]
        for label in ["original"] + (["exclude_lvis_overlap"] if ds == "coco" else []):
            rr = [r for r in records if label == "original" or int(Path(r["image"]).stem) not in c.imgs]
            vr = [r for r in vrecords if label == "original" or int(Path(r["image"]).stem) not in c.imgs]
            sample_ids = sorted(int(Path(r["image"]).stem) for r in rr)
            val = {"images": len(rr), "escalated": sum(r["escalated"] for r in rr),
                   "escalation_rate": sum(r["escalated"] for r in rr) / len(rr),
                   "mean_total_latency_sec": float(np.mean([r["latency_total_sec"] for r in rr]))}
            for m, pairs in [("detector", [(r["image"], r["detector"]["detections"]) for r in rr]),
                             ("fused", [(r["image"], r["fused"]) for r in rr]),
                             ("vlm_only", [(r["image"], r["detections"]) for r in vr])]:
                dets = []
                names = mappings(cc)
                for img, dd in pairs:
                    for d in dd:
                        cid = CM.category_id_for_label(d["label"], names)
                        if cid not in cats:
                            continue
                        x1, y1, x2, y2 = d.get("box_2d") or d.get("box") or d["bbox"]
                        dets.append({"image_id": int(Path(img).stem), "category_id": cid,
                                     "bbox": [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)],
                                     "score": float(d.get("score", d.get("confidence", 1.0)))})
                metric, ev = evaluate(cc, dets, sample_ids, cats)
                if m == "vlm_only":
                    val["vlm_only_supplied_score_sensitivity"] = metric
                    for d in dets:
                        d["score"] = 1.0
                    metric, ev = evaluate(cc, dets, sample_ids, cats)
                val[m] = metric
                if ds == "coco" and label == "exclude_lvis_overlap" and m in {"detector", "fused"}:
                    router_boot[m] = (dets, ev)
            result["router"][f"{ds}_{label}"] = val
            print(ds, label, val, flush=True)
        if ds == "coco":
            overlap = sorted(int(Path(r["image"]).stem) for r in records if int(Path(r["image"]).stem) in c.imgs)
            result["dataset"]["shared_image_ids"] = overlap
            result["dataset"]["coco_original_positive_images"] = len(records)
            result["dataset"]["coco_image_disjoint_images"] = len(records)-len(overlap)
    if args.bootstrap:
        result["paired_coco_disjoint_router_bootstrap"] = paired_bootstrap(
            coco, router_boot, router_boot["detector"][1].params.imgIds, scope.cat_ids, args.bootstrap, 12345)
    result["elapsed_seconds"] = time.perf_counter() - start
    result["input_sha256"] = INPUTS
    result["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = OUT / "replay_results.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {output}; elapsed {result['elapsed_seconds']:.1f}s", flush=True)

if __name__ == "__main__":
    main()
