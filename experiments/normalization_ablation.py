#!/usr/bin/env python3
"""
Normalization ablation: quantify how much of each VLM's LVIS-unseen AP depends on
the coordinate-normalization step (the fix introduced in the 2026 refresh) versus
the naive 'treat every box as pixel xyxy' interpretation used by earlier pipelines.

ON  = each model's correct coordinate_mode (the paper's reported numbers).
OFF = every model forced to coordinate_mode='pixel' (the legacy bug).

Only 5 models differ between ON and OFF (the two Gemini models, normalized_1000_yxyx;
and qwen3-vl-235b / gemma-3-27b / mistral-large, normalized_1000_xyxy). All other
models are already pixel-mode, so OFF == ON for them and we reuse their cached files.

Writes two parallel per-image trees and leaves AP computation to compute_metrics.py.
"""
from __future__ import annotations
import json, os, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.vlm.parsing import parse_detections_from_text, normalize_detections_to_image  # noqa
from experiments.vlm_batch import (  # noqa
    extract_gemini_response_text,
    extract_anthropic_message_text,
    extract_openai_response_text,
)
from models.vlm.parsing import extract_openai_compatible_text  # noqa
from pycocotools.coco import COCO  # noqa

SRC = ROOT / "results/metrics_inputs/vlm_completed_refresh_2026/lvis_completed_refresh_2026/per_image"
OUT = ROOT / "results/ablation_normalization"
LVIS_ANN = ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"

# affected model -> (real coordinate_mode, raw-text extractor)
def gemini_text(rr):
    resp = rr.get("response", rr) if isinstance(rr, dict) else {}
    return extract_gemini_response_text(resp)

def openrouter_text(rr):
    return extract_openai_compatible_text(rr) if isinstance(rr, dict) else ""

AFFECTED = {
    "gemini-3.5-flash":          ("normalized_1000_yxyx", gemini_text),
    "gemini-3.1-pro":            ("normalized_1000_yxyx", gemini_text),
    "qwen3-vl-235b-openrouter":  ("normalized_1000_xyxy", openrouter_text),
    "gemma-3-27b-openrouter":    ("normalized_1000_xyxy", openrouter_text),
    "mistral-large-openrouter":  ("normalized_1000_xyxy", openrouter_text),
}


def build_size_index():
    coco = COCO(str(LVIS_ANN))
    idx = {}
    for img_id, img in coco.imgs.items():
        fn = img.get("file_name") or img.get("coco_url", "").split("/")[-1]
        stem = Path(fn).stem if fn else str(img_id)
        idx[stem] = (int(img["width"]), int(img["height"]))
        idx[str(img_id)] = (int(img["width"]), int(img["height"]))
    return idx


def rederive(model, mode_on, extractor, size_idx, out_on, out_off):
    """Return (n_files, n_match) where match = ON-from-raw reproduces stored ON detection count loosely."""
    src_dir = SRC / model
    (out_on / model).mkdir(parents=True, exist_ok=True)
    (out_off / model).mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        stem = f.stem
        size = size_idx.get(stem)
        if size is None:
            img = Path(d.get("image", "")).stem
            size = size_idx.get(img, (0, 0))
        rr = d.get("raw_response") or {}
        text = extractor(rr)
        raw_dets = parse_detections_from_text(text, image_size=None)
        on_dets = normalize_detections_to_image(raw_dets, size, coordinate_mode=mode_on)
        off_dets = normalize_detections_to_image(raw_dets, size, coordinate_mode="pixel")
        for tree, dets in ((out_on, on_dets), (out_off, off_dets)):
            payload = dict(d)
            payload["detections"] = dets
            (tree / model / f.name).write_text(json.dumps(payload), encoding="utf-8")
        n += 1
    return n


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    out_on = OUT / "ON_fromraw/lvis/per_image"      # validation: should match stored ON
    out_off = OUT / "OFF_pixel/lvis/per_image"      # the bug condition
    out_on.mkdir(parents=True); out_off.mkdir(parents=True)

    size_idx = build_size_index()

    # 1) affected models: rederive from raw
    for model, (mode_on, extractor) in AFFECTED.items():
        n = rederive(model, mode_on, extractor, size_idx, out_on, out_off)
        print(f"  rederived {model}: {n} files")

    # 2) unaffected models: OFF == ON == cached. Symlink their dirs into OFF tree
    for model_dir in sorted(SRC.iterdir()):
        if not model_dir.is_dir() or model_dir.name in AFFECTED:
            continue
        (out_off / model_dir.name).symlink_to(model_dir, target_is_directory=True)

    print("OFF tree:", out_off)
    print("ON-from-raw (validation) tree:", out_on)


if __name__ == "__main__":
    main()
