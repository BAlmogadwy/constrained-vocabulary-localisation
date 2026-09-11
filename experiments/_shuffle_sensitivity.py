#!/usr/bin/env python3
"""Order-shuffle sensitivity for uniform-score VLM detections under COCOeval.

VLM detections carry uniform score 1.0, so COCOeval's stable sort preserves
response order; greedy matching could in principle depend on that order. This
test shuffles the within-image detection order (5 seeds) for the replayable
decoded VLM (Qwen3-VL-8B, LVIS-unseen) and recomputes AP.
"""
from __future__ import annotations
import json, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments.compute_metrics as cm
from pycocotools.coco import COCO

ANN = cm.DATASET_CONFIGS["lvis-unseen"]["annotation"]
DYN = ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json"
VLM = ROOT / "results/raw/vlm_decoded/lvis_Qwen3-VL-8B-Instruct/per_image/Qwen3-VL-8B-Instruct"

coco = COCO(str(ANN))
for a in coco.dataset.get("annotations", []):
    a.setdefault("iscrowd", 0)
coco.createIndex()
c2i = {cm.label_lookup_key(c["name"]): c["id"] for c in coco.loadCats(coco.getCatIds())}
scope = cm.load_eval_scope(DYN, "lvis-unseen", coco, c2i)

# load per-image detections (pixel xyxy, uniform score 1.0)
perimg = {}
for f in sorted(VLM.glob("*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    iid = cm.stem_to_image_id(Path(d.get("image", f.stem)).stem, "lvis-unseen", coco)
    if iid is None or iid not in scope.image_ids:
        continue
    dets = []
    for de in d.get("detections") or []:
        lbl = de.get("label") or de.get("class_name")
        box = de.get("box_2d")
        if not lbl or not box or len(box) != 4:
            continue
        cid = cm.category_id_for_label(lbl, c2i)
        if cid is None or int(cid) not in scope.cat_ids:
            continue
        x1, y1, x2, y2 = box
        dets.append({"image_id": int(iid), "category_id": int(cid),
                     "bbox": [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], "score": 1.0})
    perimg[iid] = dets

def ap_with_order(order_seed):
    all_dets = []
    for iid in sorted(perimg):
        dets = list(perimg[iid])
        if order_seed is not None:
            random.Random(order_seed * 100003 + iid).shuffle(dets)
        all_dets.extend(dets)
    ap, ap50 = cm.evaluate_model(coco, all_dets, image_ids=scope.image_ids, cat_ids=scope.cat_ids)
    return ap, ap50

base = ap_with_order(None)
print(f"original order : AP={base[0]:.4f} AP50={base[1]:.4f}")
aps = []
for s in range(1, 6):
    ap, ap50 = ap_with_order(s)
    aps.append(ap)
    print(f"shuffle seed {s} : AP={ap:.4f} AP50={ap50:.4f}")
print(f"max |delta AP| vs original: {max(abs(a - base[0]) for a in aps):.5f}")
