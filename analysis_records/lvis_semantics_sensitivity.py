"""Separate LVIS-aware sensitivity; leaves replay_results.json unchanged."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
from pathlib import Path
OUT = Path(__file__).resolve().parent
from review_paths import output_path
import collections
import copy
import hashlib
import importlib.metadata
import json
import logging
import time
import zipfile
import numpy as np
# LVIS 0.5.3 uses the removed np.float alias, whose historical value was float.
# Restore only that alias in memory; evaluator source and algorithm are unchanged.
np.float = float
import lvis
from lvis import LVIS, LVISResults, LVISEval
import replay_audit as ra
logging.getLogger("lvis").setLevel(logging.ERROR)


def per_category_cap(dets, cap=100):
    grouped = collections.defaultdict(list)
    for d in dets:
        grouped[d["image_id"], d["category_id"]].append(d)
    # Stable score sorting retains historical order for ties.
    return [d for values in grouped.values()
            for d in sorted(values, key=lambda x: -x["score"])[:cap]]


def lvis_eval(gt, dets, official=True):
    selected = copy.deepcopy(dets if official else per_category_cap(dets))
    results = LVISResults(gt, selected, max_dets=300 if official else -1)
    evaluator = LVISEval(gt, results, "bbox")
    evaluator.run()
    return {"metrics": {k: float(v) for k, v in evaluator.get_results().items()},
            "input_detections": len(dets),
            "detections_after_cap": len(results.dataset["annotations"]),
            "detections_after_federated_category_filter": sum(len(x) for x in evaluator._dts.values())}


def main():
    started = time.perf_counter()
    annotation = ra.ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"
    c = ra.coco_load(annotation)
    gt = LVIS(str(annotation))
    ids = sorted(c.imgs)
    cats = set(c.cats)
    affected = {str(i): sorted(cats.intersection(c.imgs[i]["not_exhaustive_category_ids"]))
                for i in ids if cats.intersection(c.imgs[i]["not_exhaustive_category_ids"])}
    output = {
        "status": "separate new offline sensitivity; frozen COCO-style replay unchanged",
        "runtime": {"python": sys.version, "numpy": np.__version__, "lvis": importlib.metadata.version("lvis")},
        "official_source": "https://github.com/lvis-dataset/lvis-api",
        "annotation_path": str(annotation),
        "dataset": {"images": len(ids), "categories": len(cats), "annotations": len(c.anns),
                    "images_with_selected_not_exhaustive_categories": len(affected),
                    "selected_not_exhaustive_image_categories": affected},
        "settings": {
            "iou_type": "bbox", "iou_thresholds": "0.50:0.05:0.95", "recall_thresholds": "0.00:0.01:1.00",
            "official_lvis": "LVIS 0.5.3 official defaults; max 300 detections globally per image, before federated filtering; preserved negative and not-exhaustive category metadata; macro category AP; frequency groups retained from original category records",
            "compatible_cap_sensitivity": "Identical LVIS federated semantics after stable top-100 detections per category per image; LVISResults global cap disabled (-1). This isolates LVIS annotation semantics at the frozen replay's COCO cap. It is not the official LVIS default cap. The evaluator's AR@300 labels are not meaningful under this custom input cap; use AP metrics.",
            "confidence": "VLM score 1.0 only after fixed historical parser accepts the box; detector native scores at archived inference floor 0.001",
            "compatibility_patch": "np.float = float in memory, restoring alias removed in NumPy; installed evaluator source unmodified",
            "scope": "All 396 images and 10 selected categories only; this is not full LVIS benchmark performance. Existing ground-truth-conditioned candidate labels preserved. No new inference or API calls."
        }, "models": {}
    }
    archive = ra.ROOT / "results/ablation_normalization.zip"
    ra.digest(archive)
    predictions = {}
    with zipfile.ZipFile(archive) as z:
        for model in ["gemini-3.5-flash", "gemini-3.1-pro", "qwen3-vl-235b-openrouter"]:
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
    for model in ["yolo_world", "grounding_dino", "owl_vit"]:
        folder = ra.ROOT / "results/raw/traditional_lowthr/lvis_v1_unseen/per_image" / model
        recs = []
        for path in sorted(folder.glob("*.json")):
            item = ra.read(path); block = item.get("result", item)
            recs.append((path.stem, block.get("detections", [])))
        assert len(recs) == len(ids)
        predictions[model] = ra.as_coco(c, recs, family="traditional")
    frozen = ra.read(OUT / "replay_results.json")
    for model, dets in predictions.items():
        coco_metrics, _ = ra.evaluate(c, copy.deepcopy(dets), ids)
        expected = (frozen["vlm"][model]["conditions"]["raw_correct_decoder"]["uniform"]
                    if model in frozen["vlm"] else frozen["detectors"][model])
        assert abs(coco_metrics["ap"] - expected["ap"]) < 1e-12
        entry = {"reproduced_frozen_coco_style": coco_metrics,
                 "official_lvis": lvis_eval(gt, dets, True),
                 "compatible_cap_lvis": lvis_eval(gt, dets, False)}
        entry["annotation_semantics_ap_delta_at_compatible_cap"] = entry["compatible_cap_lvis"]["metrics"]["AP"] - coco_metrics["ap"]
        output["models"][model] = entry
        print(model, entry["official_lvis"]["metrics"]["AP"], entry["compatible_cap_lvis"]["metrics"]["AP"], flush=True)
    for path in sorted(Path(lvis.__file__).resolve().parent.glob("*.py")):
        ra.digest(path)
    ra.digest(OUT / "replay_audit.py")
    output["input_sha256"] = ra.INPUTS
    output["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output["elapsed_seconds"] = time.perf_counter() - started
    output_path("lvis_semantics_sensitivity.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    lines = ["# LVIS annotation-semantics sensitivity", "",
             "This separate reevaluation preserves the frozen `replay_results.json`. It uses the official LVIS 0.5.3 API on the same 396-image, 10-category subset and unchanged cached predictions. It is not a full LVIS benchmark result.", "",
             "The subset contains " + str(len(affected)) + " images with selected categories marked not exhaustive. LVIS ignores unmatched detections for those image/category pairs and removes detections in categories whose presence or absence is unverified. COCO-style evaluation does not implement these rules.", "",
             "| Model | Frozen COCO AP | LVIS AP, global 300 | LVIS AP50, global 300 | LVIS AP, compatible 100/category |", "|---|---:|---:|---:|---:|"]
    for model, e in output["models"].items():
        lines.append(f'| {model} | {e["reproduced_frozen_coco_style"]["ap"]:.6f} | {e["official_lvis"]["metrics"]["AP"]:.6f} | {e["official_lvis"]["metrics"]["AP50"]:.6f} | {e["compatible_cap_lvis"]["metrics"]["AP"]:.6f} |')
    lines += ["", "Official default: at most 300 detections per image across categories, before federated filtering. The compatible-cap sensitivity first applies a stable top-100 cap per category/image, then disables the LVIS global cap; it isolates annotation semantics relative to frozen COCO-style AP and is not the official LVIS cap. Uniform-score ties retain source order. Detector scores remain native.", "",
              "LVIS 0.5.3 refers to `np.float`; this script restores that removed alias as `float` in memory without changing evaluator source or scoring logic. Source input and evaluator file hashes are recorded in JSON. AP and AP50 are fractions. No bootstrap was added to this bounded sensitivity.", "",
              "From the repository root: `python -m pip install -r requirements-review.txt`, then `python analysis_records/lvis_semantics_sensitivity.py`. See REVIEWER_GUIDE.md for input preparation. Outputs are written separately under analysis_records/recomputed/.", "",
              "Official API: https://github.com/lvis-dataset/lvis-api", "",
              "Do not reuse the frozen COCO bootstrap intervals as LVIS intervals. This sensitivity does not resolve ground-truth-conditioned candidate labels, historical model provenance, or pretraining overlap."]
    output_path("LVIS_SEMANTICS_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved separate LVIS sensitivity in", output["elapsed_seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()
