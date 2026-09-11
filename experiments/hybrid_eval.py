#!/usr/bin/env python3
"""Offline late-fusion hybrid (detector + VLM) from cached boxes. Parameterized over
dataset (lvis|coco). Evaluated on POSITIVE images (>=1 unseen GT) that the VLM covers,
so detector-only / VLM-only / fused are all measured on the identical image set.

  python experiments/hybrid_eval.py --dataset lvis            # full run + Pareto + oracle
  python experiments/hybrid_eval.py --dataset coco
  python experiments/hybrid_eval.py --dataset lvis --sensitivity   # fusion hyperparam grid
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments.compute_metrics as CM
from pycocotools.coco import COCO

CFG = {
    "lvis": dict(
        ann="data/processed/lvis_v1/val/lvis_v1_val_unseen.json",
        dyn="data/processed/lvis_v1/val/lvis_unseen_dynamic_labels.json",
        dkey="lvis-unseen",
        det="results/raw/traditional_lowthr/lvis_v1_unseen/per_image/yolo_world",
        vlm="results/raw/vlm_decoded/lvis_Qwen3-VL-8B-Instruct/per_image/Qwen3-VL-8B-Instruct",
    ),
    "coco": dict(
        ann="data/raw/coco2017/annotations/instances_val2017.json",
        dyn="data/processed/coco2017/val/coco_unseen_dynamic_labels.json",
        dkey="coco2017-val",
        det="results/raw/traditional_lowthr/coco2017_unseen/per_image/yolo_world",
        vlm="results/raw/vlm_decoded_coco/coco_Qwen3-VL-8B-Instruct/per_image/Qwen3-VL-8B-Instruct",
    ),
}

def norm(s): return "".join(c for c in str(s).lower() if c.isalnum())

def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2]-a[0])*max(0.0, a[3]-a[1]) + max(0.0, b[2]-b[0])*max(0.0, b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def load_perimage(d: Path):
    boxes, lat = {}, {}
    for f in sorted(d.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        res = data.get("result") if isinstance(data.get("result"), dict) else None
        dets = (res.get("detections") if res else data.get("detections")) or []
        name = Path(data.get("image", f.stem)).name
        lst = []
        for de in dets:
            box = de.get("box_2d") or de.get("box") or de.get("bbox")
            if not box or len(box) != 4:
                continue
            lbl = de.get("label") or de.get("class_name") or de.get("class")
            if not lbl:
                continue
            sc = de.get("score")
            lst.append({"label": lbl, "box": [float(x) for x in box], "score": 1.0 if sc is None else float(sc)})
        boxes[name] = lst
        lt = (res.get("latency_sec") if res else data.get("latency_sec"))
        if lt is not None:
            lat[name] = float(lt)
    return boxes, lat

def make_fuse(p_v, boost, iou_thr=0.5):
    def fuse(D, V):
        out, used = [], set()
        for d in D:
            bj, bi = -1, iou_thr
            for j, v in enumerate(V):
                if j in used or norm(v["label"]) != norm(d["label"]):
                    continue
                i = iou(d["box"], v["box"])
                if i >= bi:
                    bi, bj = i, j
            if bj >= 0:
                used.add(bj)
                out.append({"label": d["label"], "box": d["box"], "score": min(1.0, d["score"] + boost)})
            else:
                out.append(dict(d))
        for j, v in enumerate(V):
            if j not in used:
                out.append({"label": v["label"], "box": v["box"], "score": p_v})
        return out
    return fuse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(CFG), default="lvis")
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()
    c = CFG[args.dataset]

    coco = COCO(str(ROOT / c["ann"]))
    for a in coco.dataset.get("annotations", []):
        a.setdefault("iscrowd", 0)
    coco.createIndex()
    cat_name_to_id = {CM.label_lookup_key(cc["name"]): cc["id"] for cc in coco.loadCats(coco.getCatIds())}
    scope = CM.load_eval_scope(ROOT / c["dyn"], c["dkey"], coco, cat_name_to_id)
    CAT_IDS = set(scope.cat_ids)
    POS = {iid for iid in scope.image_ids
           if any(a["category_id"] in CAT_IDS for a in coco.imgToAnns.get(iid, []))}

    D, dlat = load_perimage(ROOT / c["det"])
    V, vlat = load_perimage(ROOT / c["vlm"])
    # eval set = positive images the VLM covers (so escalation is possible everywhere)
    names = [n for n in (set(D) & set(V)) if CM.stem_to_image_id(Path(n).stem, c["dkey"], coco) in POS]
    names.sort()
    IMG_IDS = {CM.stem_to_image_id(Path(n).stem, c["dkey"], coco) for n in names}
    det_lat = sum(dlat.get(n, 0) for n in names) / len(names)
    vlm_lat = sum(vlat.get(n, 0) for n in names) / max(1, len([n for n in names if n in vlat]))
    det_conf = {n: max([d["score"] for d in D.get(n, [])], default=0.0) for n in names}

    def ap_of(perimg):
        dets = []
        for name, lst in perimg.items():
            iid = CM.stem_to_image_id(Path(name).stem, c["dkey"], coco)
            if iid not in IMG_IDS:
                continue
            for d in lst:
                cid = CM.category_id_for_label(d["label"], cat_name_to_id)
                if cid is None or int(cid) not in CAT_IDS:
                    continue
                x1, y1, x2, y2 = d["box"]
                dets.append({"image_id": int(iid), "category_id": int(cid),
                             "bbox": [x1, y1, max(0.0, x2-x1), max(0.0, y2-y1)], "score": float(d["score"])})
        return CM.evaluate_model(coco, dets, image_ids=IMG_IDS, cat_ids=CAT_IDS)

    print(f"[{args.dataset}] eval images={len(names)} (positive & VLM-covered)  "
          f"det_lat={det_lat:.3f}s  vlm_lat={vlm_lat:.2f}s")

    if args.sensitivity:
        print("\nSensitivity — fused-all AP / router-peak AP over (VLM prior p_v, agreement boost):")
        print(f"{'p_v|boost':>10s}" + "".join(f"{b:>14.2f}" for b in (0.0, 0.15, 0.30, 0.45)))
        for p_v in (0.20, 0.30, 0.40, 0.50, 0.60):
            cells = []
            for boost in (0.0, 0.15, 0.30, 0.45):
                fuse = make_fuse(p_v, boost)
                fused = {n: fuse(D.get(n, []), V.get(n, [])) for n in names}
                ap_fus = ap_of(fused)[0]
                peak = max(ap_of({n: (fused[n] if det_conf[n] < t/20 else D.get(n, [])) for n in names})[0]
                           for t in range(0, 21))
                cells.append(f"{ap_fus:.3f}/{peak:.3f}")
            print(f"{p_v:>10.2f}" + "".join(f"{x:>14s}" for x in cells))
        return

    fuse = make_fuse(0.40, 0.30)
    det_only = {n: D.get(n, []) for n in names}
    vlm_only = {n: V.get(n, []) for n in names}
    fused_all = {n: fuse(D.get(n, []), V.get(n, [])) for n in names}
    ap_det, ap_vlm, ap_fus = ap_of(det_only), ap_of(vlm_only), ap_of(fused_all)
    print(f"detector-only AP={ap_det[0]:.4f}   VLM-only AP={ap_vlm[0]:.4f}   fused-all AP={ap_fus[0]:.4f}")

    print("\ntau    AP      AP50    escal%   latency(s)")
    sweep = []
    for k in range(0, 21):
        tau = k / 20.0
        esc = set(n for n in names if det_conf[n] < tau)
        perimg = {n: (fused_all[n] if n in esc else D.get(n, [])) for n in names}
        a, a5 = ap_of(perimg); ef = len(esc) / len(names)
        sweep.append((tau, a, a5, ef, det_lat + ef * vlm_lat))
        print(f"{tau:.2f}  {a:.4f}  {a5:.4f}  {ef*100:5.1f}   {det_lat+ef*vlm_lat:8.2f}")

    state = {n: D.get(n, []) for n in names}; cur = ap_of(state)[0]; esc_or = set()
    for n in sorted(names, key=lambda n: det_conf[n]):
        trial = dict(state); trial[n] = fused_all[n]; a = ap_of(trial)[0]
        if a > cur + 1e-9:
            state, cur = trial, a; esc_or.add(n)
    ef_or = len(esc_or) / len(names)
    print(f"\n[oracle ceiling] AP={cur:.4f}  escal%={ef_or*100:.1f}  latency={det_lat+ef_or*vlm_lat:.2f}s")

    peak = max(sweep, key=lambda r: r[1])
    out = {"dataset": args.dataset, "n_images": len(names),
           "detector_only": ap_det[0], "vlm_only": ap_vlm[0], "fused_all": ap_fus[0],
           "router_peak": {"tau": peak[0], "ap": peak[1], "escal_frac": peak[3]},
           "oracle_ceiling": {"ap": cur, "escal_frac": ef_or},
           "sweep": [{"tau": t, "ap": a, "ap50": a5, "escal_frac": e, "latency_s": l} for (t, a, a5, e, l) in sweep]}
    (ROOT / f"results/hybrid_{args.dataset}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote results/hybrid_{args.dataset}.json")

if __name__ == "__main__":
    main()
