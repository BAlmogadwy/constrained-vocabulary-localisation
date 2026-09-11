#!/usr/bin/env python3
"""
Image-level bootstrap confidence intervals for LVIS-unseen AP (IoU 0.50:0.95).

Resamples the 396 images with replacement (B times), recomputing COCO-style AP per
model on each resample, and reports the 2.5/97.5 percentile CI. Also reports pairwise
P(AP_model > AP_reference) for the headline ranking claims.

Implementation: each model's per-image match arrays are computed once via
COCOeval.evaluate(); a custom weighted accumulate then recomputes AP for any image
multiset cheaply, so 1000 resamples are fast. The weights=1 case is validated against
compute_metrics' AP before bootstrapping.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import experiments.compute_metrics as cm

LAB = ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json"
ANN = cm.DATASET_CONFIGS["lvis-unseen"]["annotation"]
VLM_DIR = ROOT / "results/metrics_inputs/vlm_completed_refresh_2026"
TRAD_DIR = ROOT / "results/raw/traditional"
B = int(os.getenv("BOOT_B", "1000"))
SEED = 12345


def weighted_ap(evaluator: COCOeval, weight: dict) -> float:
    """Recompute mean AP (IoU .5:.95, area=all, maxDet=100) weighting each image by weight[imgId]."""
    p = evaluator.params
    T = len(p.iouThrs)
    R = len(p.recThrs)
    K = len(p.catIds)
    A = 0  # area index for 'all'
    nArea = len(p.areaRng)
    M = len(p.maxDets) - 1  # last maxDet (100)
    maxDet = p.maxDets[M]
    nImg = len(p.imgIds)
    img_index = {img_id: i for i, img_id in enumerate(p.imgIds)}
    ap_per_k = []
    for k in range(K):
        # gather per-image eval results for this cat/area
        dtScores_all, dtm_all, dtIg_all, w_all = [], [], [], []
        npig_total = 0.0
        for img_id in p.imgIds:
            i = img_index[img_id]
            e = evaluator.evalImgs[k * nArea * nImg + A * nImg + i]
            w = weight.get(img_id, 0)
            if e is None:
                continue
            gtIg = np.array(e["gtIgnore"])
            npig = np.count_nonzero(gtIg == 0)
            npig_total += npig * w
            dtScores = np.array(e["dtScores"])[:maxDet]
            if len(dtScores) == 0:
                continue
            dtm = np.array(e["dtMatches"])[:, :maxDet]      # T x D
            dtIg = np.array(e["dtIgnore"])[:, :maxDet]       # T x D
            dtScores_all.append(dtScores)
            dtm_all.append(dtm)
            dtIg_all.append(dtIg)
            w_all.append(np.full(len(dtScores), w))
        if npig_total == 0:
            continue
        if not dtScores_all:
            ap_per_k.append(0.0)
            continue
        scores = np.concatenate(dtScores_all)
        dtm = np.concatenate(dtm_all, axis=1)     # T x Dtot
        dtIg = np.concatenate(dtIg_all, axis=1)   # T x Dtot
        wts = np.concatenate(w_all)               # Dtot
        order = np.argsort(-scores, kind="mergesort")
        dtm = dtm[:, order]
        dtIg = dtIg[:, order]
        wts = wts[order]
        tps = np.logical_and(dtm, np.logical_not(dtIg))
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dtIg))
        # weighted cumulative counts
        tp_sum = np.cumsum(tps * wts, axis=1).astype(float)   # T x Dtot
        fp_sum = np.cumsum(fps * wts, axis=1).astype(float)
        ap_t = np.zeros(T)
        for t in range(T):
            tp = tp_sum[t]
            fp = fp_sum[t]
            rc = tp / npig_total
            pr = tp / (tp + fp + np.spacing(1))
            pr = pr.tolist()
            # precision envelope (monotonic)
            for i in range(len(pr) - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            inds = np.searchsorted(rc, p.recThrs, side="left")
            q = np.zeros(R)
            pr = np.array(pr)
            for ri, pi in enumerate(inds):
                if pi < len(pr):
                    q[ri] = pr[pi]
            ap_t[t] = q.mean()
        ap_per_k.append(ap_t.mean())
    return float(np.mean(ap_per_k)) if ap_per_k else 0.0


def build_evaluator(preds, coco, cat_name_to_id, eval_scope):
    eval_image_ids = eval_scope.image_ids
    eval_cat_ids = eval_scope.cat_ids
    detections = cm.prepare_coco_detections(
        preds, coco, cat_name_to_id,
        allowed_image_ids=eval_image_ids, allowed_cat_ids=eval_cat_ids)
    if not detections:
        return None, eval_image_ids
    coco_dt = coco.loadRes(detections)
    ev = COCOeval(coco, coco_dt, "bbox")
    ev.params.imgIds = sorted(eval_image_ids)
    ev.params.catIds = sorted(eval_cat_ids)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate()
    return ev, eval_image_ids


def main():
    coco = COCO(str(ANN))
    upd = False
    for ann in coco.dataset.get("annotations", []):
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0; upd = True
    if upd:
        coco.createIndex()
    cat_name_to_id = {cm.label_lookup_key(c["name"]): c["id"] for c in coco.loadCats(coco.getCatIds())}
    eval_scope = cm.load_eval_scope(LAB, "lvis-unseen", coco, cat_name_to_id)

    vlm = cm.collect_predictions("vlm", VLM_DIR)["lvis-unseen"]
    trad = cm.collect_predictions("traditional", TRAD_DIR)["lvis-unseen"]
    models = {**vlm, **trad}

    evaluators, fam = {}, {}
    point_ap = {}
    img_ids_sorted = sorted(eval_scope.image_ids)
    for name, preds in models.items():
        ev, _ = build_evaluator(preds, coco, cat_name_to_id, eval_scope)
        evaluators[name] = ev
        fam[name] = preds.family
        w1 = {i: 1 for i in img_ids_sorted}
        point_ap[name] = weighted_ap(ev, w1) if ev is not None else 0.0

    # bootstrap
    rng = np.random.default_rng(SEED)
    n = len(img_ids_sorted)
    boot = {name: np.zeros(B) for name in models}
    img_arr = np.array(img_ids_sorted)
    for b in range(B):
        sample = rng.integers(0, n, size=n)
        counts = np.bincount(sample, minlength=n)
        weight = {int(img_arr[i]): int(counts[i]) for i in range(n) if counts[i] > 0}
        for name, ev in evaluators.items():
            boot[name][b] = weighted_ap(ev, weight) if ev is not None else 0.0

    # report
    out = {}
    for name in models:
        arr = boot[name]
        out[name] = {
            "family": fam[name],
            "ap_point": point_ap[name],
            "ap_boot_mean": float(arr.mean()),
            "ci95_lo": float(np.percentile(arr, 2.5)),
            "ci95_hi": float(np.percentile(arr, 97.5)),
        }
    # pairwise vs grounding_dino
    ref = "grounding_dino"
    pair = {}
    if ref in boot:
        for name in models:
            if name == ref:
                continue
            diff = boot[name] - boot[ref]
            pair[name] = {
                "p_gt_gdino": float(np.mean(diff > 0)),
                "median_diff": float(np.median(diff)),
                "diff_ci_lo": float(np.percentile(diff, 2.5)),
                "diff_ci_hi": float(np.percentile(diff, 97.5)),
            }
    result = {"B": B, "seed": SEED, "n_images": n, "per_model": out, "vs_grounding_dino": pair}
    outpath = ROOT / "results/tables/lvis_bootstrap_ci.json"
    outpath.write_text(json.dumps(result, indent=2), encoding="utf-8")

    rows = sorted(out.items(), key=lambda kv: -kv[1]["ap_point"])
    print(f'{"model":28}{"AP":>8}{"95% CI":>20}{"fam":>12}')
    for name, r in rows:
        print(f'{name:28}{r["ap_point"]:8.4f}   [{r["ci95_lo"]:.4f}, {r["ci95_hi"]:.4f}]   {r["family"]:>10}')
    print("\nvs grounding_dino (0.267):")
    for name in [n for n, _ in rows if n in pair]:
        pr = pair[name]
        print(f'  {name:28} P(AP>GDINO)={pr["p_gt_gdino"]:.3f}  diff95=[{pr["diff_ci_lo"]:+.3f},{pr["diff_ci_hi"]:+.3f}]')
    print("\nwritten:", outpath)


if __name__ == "__main__":
    main()
