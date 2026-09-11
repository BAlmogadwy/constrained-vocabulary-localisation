#!/usr/bin/env python3
"""Reconstruct lvis_unseen_dynamic_labels.json from the cached detector run.

- `labels` per image = the exact candidate labels used at inference (from the
  cached aggregate JSON `labels_used`), guaranteeing the eval scope matches the
  original run.
- `positives` per image = the ground-truth-present unseen categories, derived
  from the rebuilt lvis_v1_val_unseen.json.

This sidesteps the seeded-random generator entirely, so the eval scope is exact.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "data" / "processed" / "lvis_v1" / "val" / "lvis_v1_val_unseen.json"
CACHE = ROOT / "results" / "raw" / "traditional" / "lvis_v1_unseen" / "grounding_dino_lvis_unseen.json"
OUT = ROOT / "data" / "processed" / "lvis_v1" / "val" / "lvis_unseen_dynamic_labels.json"

gt = json.loads(GT.read_text(encoding="utf-8"))
catid_to_name = {c["id"]: c["name"] for c in gt["categories"]}
# image_id -> set of annotated (positive) unseen category names
pos_by_img: dict[int, set[str]] = defaultdict(set)
for ann in gt["annotations"]:
    pos_by_img[ann["image_id"]].add(catid_to_name[ann["category_id"]])
# image_id -> file stem (12-digit)
id_to_stem = {img["id"]: Path(img.get("file_name") or img.get("coco_url", "")).stem for img in gt["images"]}

cache = json.loads(CACHE.read_text(encoding="utf-8"))
labels_per_image: dict[str, dict] = {}
for rec in cache["results"]:
    image_name = rec["image"]          # e.g. 000000000294.jpg
    stem = Path(image_name).stem
    image_id = int(stem)
    labels = list(rec.get("labels_used") or [])
    positives = sorted(pos_by_img.get(image_id, set()))
    labels_per_image[image_name] = {"labels": labels, "positives": positives}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({"labels_per_image": labels_per_image}, indent=2), encoding="utf-8")

# quick stats
n_img = len(labels_per_image)
all_labels = set()
for v in labels_per_image.values():
    all_labels.update(x.lower() for x in v["labels"])
n_pos = sum(1 for v in labels_per_image.values() if v["positives"])
print(f"images: {n_img}")
print(f"images with >=1 positive: {n_pos}")
print(f"union of candidate labels ({len(all_labels)}): {sorted(all_labels)}")
print(f"GT categories (10): {sorted(c['name'] for c in gt['categories'])}")
print(f"wrote: {OUT.relative_to(ROOT)}")
