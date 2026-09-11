#!/usr/bin/env python3
"""Image-level bootstrap CIs for the SCORE-FLOOR-CORRECTED detectors on LVIS-unseen.

Reuses the paper's weighted-accumulate bootstrap (experiments/bootstrap_ci.py) but
points at results/raw/traditional_lowthr (threshold-free outputs). Reports, for each
corrected detector: point AP (validation vs known 0.347/0.268/0.161), the B=1000
percentile CI, and the fraction of resamples exceeding the published Gemini
lower-CI bounds — enabling an interval-separation statement against the corrected
best detector without the (unavailable) frontier per-image outputs.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pycocotools.coco import COCO
import experiments.compute_metrics as cm
from experiments.bootstrap_ci import weighted_ap, build_evaluator

LAB = ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json"
ANN = cm.DATASET_CONFIGS["lvis-unseen"]["annotation"]
TRAD_LOW = ROOT / "results/raw/traditional_lowthr"
B, SEED = 1000, 12345
GEMINI_LOWER = {"gemini-3.5-flash": 0.526, "gemini-3.1-pro": 0.504}  # published CI lower bounds

coco = COCO(str(ANN))
upd = False
for ann in coco.dataset.get("annotations", []):
    if "iscrowd" not in ann:
        ann["iscrowd"] = 0; upd = True
if upd:
    coco.createIndex()
cat_name_to_id = {cm.label_lookup_key(c["name"]): c["id"] for c in coco.loadCats(coco.getCatIds())}
scope = cm.load_eval_scope(LAB, "lvis-unseen", coco, cat_name_to_id)

trad = cm.collect_predictions("traditional", TRAD_LOW)["lvis-unseen"]
img_ids = sorted(scope.image_ids)
n = len(img_ids)
img_arr = np.array(img_ids)

out = {}
rng = np.random.default_rng(SEED)
# pre-generate shared resamples so models are evaluated on identical resamples
samples = [rng.integers(0, n, size=n) for _ in range(B)]

for name, preds in sorted(trad.items()):
    ev, _ = build_evaluator(preds, coco, cat_name_to_id, scope)
    w1 = {i: 1 for i in img_ids}
    point = weighted_ap(ev, w1)
    boots = np.zeros(B)
    for b, sample in enumerate(samples):
        counts = np.bincount(sample, minlength=n)
        weight = {int(img_arr[i]): int(counts[i]) for i in range(n) if counts[i] > 0}
        boots[b] = weighted_ap(ev, weight)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rec = {"point_ap": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
           "frac_above_gemini_flash_lower": float((boots > GEMINI_LOWER["gemini-3.5-flash"]).mean()),
           "frac_above_gemini_pro_lower": float((boots > GEMINI_LOWER["gemini-3.1-pro"]).mean())}
    out[name] = rec
    print(f"{name:16s} AP={point:.4f}  CI95=[{lo:.4f},{hi:.4f}]  "
          f"P(boot>0.526)={rec['frac_above_gemini_flash_lower']:.4f}  "
          f"P(boot>0.504)={rec['frac_above_gemini_pro_lower']:.4f}", flush=True)

(ROOT / "results/bootstrap_corrected_detectors.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print("wrote results/bootstrap_corrected_detectors.json")
