#!/usr/bin/env python3
"""Reconstruct coco_unseen_dynamic_labels.json from the cached COCO-unseen run.

Same approach as the LVIS reconstruction: per-image `labels` = the exact candidate
labels used at inference (cache `labels_used`); `positives` = the unseen categories
actually annotated in that image (from instances_val2017.json, restricted to the 12
COCO unseen categories). Guarantees the eval scope matches the original cached run.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data" / "raw" / "coco2017" / "annotations" / "instances_val2017.json"
CACHE = ROOT / "results" / "raw" / "traditional" / "coco2017_unseen" / "grounding_dino_coco_unseen.json"
OUT = ROOT / "data" / "processed" / "coco2017" / "val" / "coco_unseen_dynamic_labels.json"

UNSEEN = {"microwave", "oven", "toaster", "sink", "refrigerator", "book",
          "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"}

gt = json.loads(GT.read_text(encoding="utf-8"))
catid_to_name = {c["id"]: c["name"] for c in gt["categories"]}
pos_by_img: dict[int, set[str]] = defaultdict(set)
for ann in gt["annotations"]:
    name = catid_to_name.get(ann["category_id"])
    if name in UNSEEN:
        pos_by_img[ann["image_id"]].add(name)

cache = json.loads(CACHE.read_text(encoding="utf-8"))
labels_per_image: dict[str, dict] = {}
for rec in cache["results"]:
    image_name = rec["image"]
    image_id = int(Path(image_name).stem)
    labels = list(rec.get("labels_used") or [])
    positives = sorted(pos_by_img.get(image_id, set()))
    labels_per_image[image_name] = {"labels": labels, "positives": positives}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"labels_per_image": labels_per_image}, indent=2), encoding="utf-8")

n_img = len(labels_per_image)
n_pos = sum(1 for v in labels_per_image.values() if v["positives"])
union = sorted({x for v in labels_per_image.values() for x in v["labels"]})
print(f"images: {n_img}")
print(f"images with >=1 unseen positive: {n_pos}")
print(f"candidate label union ({len(union)}): {union}")
print(f"wrote: {OUT.relative_to(ROOT)}")
