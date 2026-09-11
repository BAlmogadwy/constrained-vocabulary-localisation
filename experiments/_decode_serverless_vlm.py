#!/usr/bin/env python3
"""Decode cached serverless-VLM boxes to pixel xyxy and write compute_metrics-ready
per-image trees, so we can measure each serverless VLM's CORRECTED LVIS-unseen AP
and pick the best one to fuse into the hybrid.

Per-model coordinate_mode follows the paper's convention (normalization_ablation.py):
Gemini -> normalized_1000_yxyx; Qwen3-VL / Gemma -> normalized_1000_xyxy; else pixel.
We also emit a 'pixel' (legacy) variant as a decode-correctness control: for the
normalized models the pixel variant should collapse to ~0 AP.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.vlm.parsing import normalize_detections_to_image  # noqa
from pycocotools.coco import COCO  # noqa

LVIS_ANN = ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"
VLM_ROOT = ROOT / "results/raw/vlm"
OUT_ON = ROOT / "results/raw/vlm_decoded"          # correct per-model mode
OUT_OFF = ROOT / "results/raw/vlm_pixel"           # legacy pixel (control)

# cache dir -> (model_subdir, coordinate_mode)
MODELS = {
    "lvis_gemini": ("gemini-2.5-flash", "normalized_1000_yxyx"),
    "lvis_qwen":   ("Qwen3-VL-8B-Instruct", "normalized_1000_xyxy"),
    "lvis_gemma":  ("gemma-3-12b-it", "normalized_1000_xyxy"),
    "lvis_gpt5":   ("gpt-5", "pixel"),
    "lvis_llama":  ("Llama-3.2-11B-Vision-Instruct", "pixel"),
    "lvis_llava":  ("llava-v1.6-mistral-7b-hf", "pixel"),
}

coco = COCO(str(LVIS_ANN))
size_idx = {}
for img_id, img in coco.imgs.items():
    fn = img.get("file_name") or img.get("coco_url", "").split("/")[-1]
    stem = Path(fn).stem if fn else str(img_id)
    size_idx[stem] = (int(img["width"]), int(img["height"]))

def decode_tree(cache_dir, model_sub, mode, out_base):
    src = VLM_ROOT / cache_dir / "per_image" / model_sub
    dst = out_base / f"lvis_{model_sub}" / "per_image" / model_sub
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        stem = Path(d.get("image", f.stem)).stem
        size = size_idx.get(stem, (0, 0))
        raw = d.get("detections") or []
        # attach a label key from class_name so compute_metrics can read it
        prepped = [{**det, "label": det.get("class_name") or det.get("label")} for det in raw]
        dec = normalize_detections_to_image(prepped, size, coordinate_mode=mode)
        for det in dec:
            det["score"] = 1.0
        payload = {"model": model_sub, "image": d.get("image", ""),
                   "detections": dec, "latency_sec": d.get("latency_sec")}
        (dst / f.name).write_text(json.dumps(payload), encoding="utf-8")
        n += 1
    return n

for cache_dir, (model_sub, mode) in MODELS.items():
    n1 = decode_tree(cache_dir, model_sub, mode, OUT_ON)
    n2 = decode_tree(cache_dir, model_sub, "pixel", OUT_OFF)
    print(f"{model_sub:32s} mode={mode:22s} files={n1}")
print("wrote:", OUT_ON, "and", OUT_OFF)
