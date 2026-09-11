#!/usr/bin/env python3
"""Decode cached serverless-VLM boxes to pixel xyxy for a dataset (lvis|coco), write
compute_metrics-ready trees, and rank each VLM by CORRECTED AP on the images it
actually covers (intersected with the eval scope). Also prints a pixel-mode control
(should collapse for the normalized models) to confirm the decode.

Usage: python experiments/_vlm_decode_rank.py {lvis|coco}
"""
from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments.compute_metrics as CM
from models.vlm.parsing import normalize_detections_to_image
from pycocotools.coco import COCO

DS = sys.argv[1] if len(sys.argv) > 1 else "coco"
if DS == "coco":
    ANN = ROOT / "data/raw/coco2017/annotations/instances_val2017.json"
    DYN = ROOT / "data/processed/coco2017/val/coco_unseen_dynamic_labels.json"
    DKEY = "coco2017-val"
    CACHES = {  # cache_dir -> (model_subdir, mode)
        "coco_gemini": ("gemini-2.5-flash", "normalized_1000_yxyx"),
        "coco_qwen":   ("Qwen3-VL-8B-Instruct", "normalized_1000_xyxy"),
        "coco_gemma":  ("gemma-3-12b-it", "normalized_1000_xyxy"),
        "coco_gpt5":   ("gpt-5", "pixel"),
        "coco_llava":  ("llava-v1.6-mistral-7b-hf", "pixel"),
    }
    OUT = ROOT / "results/raw/vlm_decoded_coco"
else:
    ANN = ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"
    DYN = ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json"
    DKEY = "lvis-unseen"
    CACHES = {
        "lvis_gemini": ("gemini-2.5-flash", "normalized_1000_yxyx"),
        "lvis_qwen":   ("Qwen3-VL-8B-Instruct", "normalized_1000_xyxy"),
        "lvis_gemma":  ("gemma-3-12b-it", "normalized_1000_xyxy"),
        "lvis_gpt5":   ("gpt-5", "pixel"),
        "lvis_llama":  ("Llama-3.2-11B-Vision-Instruct", "pixel"),
        "lvis_llava":  ("llava-v1.6-mistral-7b-hf", "pixel"),
    }
    OUT = ROOT / "results/raw/vlm_decoded_lvis2"

coco = COCO(str(ANN))
for a in coco.dataset.get("annotations", []):
    a.setdefault("iscrowd", 0)
coco.createIndex()
cat_name_to_id = {CM.label_lookup_key(c["name"]): c["id"] for c in coco.loadCats(coco.getCatIds())}
scope = CM.load_eval_scope(DYN, DKEY, coco, cat_name_to_id)
CAT_IDS = set(scope.cat_ids)
# Restrict to POSITIVE images (>=1 unseen GT annotation), mirroring LVIS's all-positive
# scope, so scoreless-VLM false positives on negative images don't confound the comparison.
SCOPE_IMGS = {iid for iid in scope.image_ids
              if any(a["category_id"] in CAT_IDS for a in coco.imgToAnns.get(iid, []))}
print(f"[scope] {DS}: {len(SCOPE_IMGS)} positive images (of {len(scope.image_ids)} in dynamic labels)")
size_idx = {}
for iid, im in coco.imgs.items():
    fn = im.get("file_name") or im.get("coco_url", "").split("/")[-1]
    size_idx[Path(fn).stem if fn else str(iid)] = (int(im["width"]), int(im["height"]))

def decode_and_eval(cache_dir, model_sub, mode):
    src = ROOT / "results/raw/vlm" / cache_dir / "per_image" / model_sub
    dst = OUT / f"{DS}_{model_sub}" / "per_image" / model_sub
    dst.mkdir(parents=True, exist_ok=True)
    covered = set()
    dets_on, dets_off = [], []
    for f in sorted(src.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        stem = Path(d.get("image", f.stem)).stem
        iid = CM.stem_to_image_id(stem, DKEY, coco)
        size = size_idx.get(stem, (0, 0))
        raw = d.get("detections") or []
        prepped = [{**de, "label": de.get("class_name") or de.get("label")} for de in raw]
        on = normalize_detections_to_image(prepped, size, coordinate_mode=mode)
        off = normalize_detections_to_image(prepped, size, coordinate_mode="pixel")
        for de in on:
            de["score"] = 1.0
        (dst / f.name).write_text(json.dumps({"model": model_sub, "image": d.get("image", ""),
                                              "detections": on, "latency_sec": d.get("latency_sec")}),
                                  encoding="utf-8")
        if iid is None or iid not in SCOPE_IMGS:
            continue
        covered.add(iid)
        for variant, dd in ((dets_on, on), (dets_off, off)):
            for de in dd:
                if not de.get("label"):
                    continue
                cid = CM.category_id_for_label(de.get("label"), cat_name_to_id)
                if cid is None or int(cid) not in CAT_IDS:
                    continue
                b = de.get("box_2d")
                x1, y1, x2, y2 = b
                variant.append({"image_id": int(iid), "category_id": int(cid),
                                "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": 1.0})
    ap_on = CM.evaluate_model(coco, dets_on, image_ids=covered, cat_ids=CAT_IDS)[0] if covered else 0.0
    ap_off = CM.evaluate_model(coco, dets_off, image_ids=covered, cat_ids=CAT_IDS)[0] if covered else 0.0
    return len(covered), ap_on, ap_off

print(f"{'VLM':30s} {'covered':>8s} {'AP(correct)':>12s} {'AP(pixel)':>10s}")
rows = []
for cache_dir, (model_sub, mode) in CACHES.items():
    n, ap_on, ap_off = decode_and_eval(cache_dir, model_sub, mode)
    rows.append((model_sub, n, ap_on, ap_off))
    print(f"{model_sub:30s} {n:8d} {ap_on:12.4f} {ap_off:10.4f}")
best = max(rows, key=lambda r: r[2])
print(f"\nBEST {DS} serverless VLM: {best[0]}  AP={best[2]:.4f} on {best[1]} covered images")
print("decoded trees ->", OUT)
