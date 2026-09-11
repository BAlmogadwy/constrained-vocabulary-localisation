"""Prepare ablation evaluation inputs:
  1. Write a reduced dynamic-labels file containing only the reduced-subset
     images, so compute_metrics evaluates AP and label accuracy over those
     images (not all 396).
  2. Normalise the synchronous Gemini detections to pixel xyxy, matching the
     batch collector (Gemini emits box_2d as [y1, x1, y2, x2] on a 0-1000 scale).
     Idempotent: marks each per-image file once converted.
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANN = PROJECT_ROOT / "data/processed/lvis_v1/val/lvis_v1_val_unseen.json"
DYN = PROJECT_ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json"
DYN_RED = PROJECT_ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels_reduced40.json"
SUBSET = PROJECT_ROOT / "data/subsets/lvis_v1_val_unseen_reduced/subset_manifest.json"
ABL = PROJECT_ROOT / "results/raw/vlm/ablation_lvis_reduced_2026"
GEMINI_MODELS = ["gemini-3.1-pro", "gemini-3.5-flash"]


def build_reduced_labels():
    files = set(json.loads(SUBSET.read_text())["files"])
    dyn = json.loads(DYN.read_text())
    lpi = dyn["labels_per_image"]
    red = {k: v for k, v in lpi.items() if k in files}
    out = {"source": dyn.get("source"), "num_images": len(red), "labels_per_image": red}
    DYN_RED.write_text(json.dumps(out, indent=2))
    print(f"[reduced-labels] {len(red)} images -> {DYN_RED.name}")


def image_sizes():
    ann = json.loads(ANN.read_text())
    return {f"{im['id']:012d}.jpg": (im["width"], im["height"]) for im in ann["images"]}


def normalise_gemini():
    sizes = image_sizes()
    for prompt in ("promptA", "promptB"):
        for model in GEMINI_MODELS:
            pdir = ABL / prompt / "per_image" / model
            if not pdir.exists():
                continue
            n_files = n_boxes = 0
            for jf in pdir.glob("*.json"):
                rec = json.loads(jf.read_text())
                if rec.get("_gem_normalized"):
                    continue
                fn = Path(rec.get("image", "")).name
                wh = sizes.get(fn)
                dets = rec.get("detections") or []
                if wh and dets:
                    W, H = wh
                    for d in dets:
                        b = d.get("box_2d")
                        if not (isinstance(b, list) and len(b) == 4):
                            continue
                        y1, x1, y2, x2 = b  # Gemini order on 0-1000 scale
                        d["box_2d"] = [
                            round(x1 / 1000.0 * W, 2), round(y1 / 1000.0 * H, 2),
                            round(x2 / 1000.0 * W, 2), round(y2 / 1000.0 * H, 2),
                        ]
                        n_boxes += 1
                rec["_gem_normalized"] = True
                jf.write_text(json.dumps(rec, indent=2))
                n_files += 1
            print(f"[gemini-norm] {prompt}/{model}: {n_files} files, {n_boxes} boxes converted")


if __name__ == "__main__":
    build_reduced_labels()
    normalise_gemini()
