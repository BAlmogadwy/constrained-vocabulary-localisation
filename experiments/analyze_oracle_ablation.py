"""Build tables + figures for the 2026 Oracle-Localisation and prompt-ablation
re-runs. Oracle metrics are computed directly from the cached per-crop records;
ablation AP/AP50/label-accuracy are computed by invoking compute_metrics.py on
the promptA/promptB output dirs (reduced subset), and latency/parse-rate from the
per-image records.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORACLE_RUN = PROJECT_ROOT / "results/raw/oracle/oracle_lvis_2026"
ABL_ROOT = PROJECT_ROOT / "results/raw/vlm/ablation_lvis_reduced_2026"
DYN = PROJECT_ROOT / "data/processed/lvis_v1/val/lvis_unseen_dynamic_labels_reduced40.json"
OUT_TABLES = PROJECT_ROOT / "results/tables"
OUT_FIGS = PROJECT_ROOT / "experiments/results/figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

DISPLAY = {
    "gpt-5.5": "GPT-5.5", "gpt-5.4-mini": "GPT-5.4 Mini",
    "claude-opus-4.8": "Claude Opus 4.8", "claude-haiku-4.5": "Claude Haiku 4.5",
    "gemini-3.1-pro": "Gemini 3.1 Pro", "gemini-3.5-flash": "Gemini 3.5 Flash",
    "qwen-vl-max": "Qwen-VL-Max", "qwen3-vl-235b-openrouter": "Qwen3-VL-235B",
    "qwen2.5-vl-72b-openrouter": "Qwen2.5-VL-72B", "mistral-large-openrouter": "Mistral Large",
    "llama-4-maverick-openrouter": "Llama 4 Maverick", "gemma-3-27b-openrouter": "Gemma 3 27B",
}


# ----------------------------- ORACLE -----------------------------
def analyze_oracle():
    summ = json.loads((ORACLE_RUN / "oracle_summary.json").read_text())
    rows = []
    per_cat = {}
    for model in summ["models"]:
        agg = json.loads((ORACLE_RUN / f"{model.replace('/', '_')}_oracle.json").read_text())
        recs = agg["records"]
        by_cat = defaultdict(lambda: [0, 0])
        for r in recs:
            by_cat[r["true_label"]][1] += 1
            if r["correct"]:
                by_cat[r["true_label"]][0] += 1
        per_cat[model] = {c: (n_ok / n) for c, (n_ok, n) in by_cat.items() if n}
        rows.append({
            "model": model, "display": DISPLAY.get(model, model),
            "recognition_accuracy": agg["recognition_accuracy"],
            "format_failure_rate": agg["format_failure_rate"],
            "n_crops": agg["n_crops"], "n_correct": agg["n_correct"], "n_error": agg["n_error"],
        })
    rows.sort(key=lambda r: r["recognition_accuracy"], reverse=True)

    # Markdown + CSV
    md = ["| Model | Recognition acc. | Format-failure rate | n crops |",
          "|---|---:|---:|---:|"]
    csv = ["model,display_model,recognition_accuracy,format_failure_rate,n_crops,n_correct,n_error"]
    for r in rows:
        md.append(f"| {r['display']} | {r['recognition_accuracy']:.3f} | {r['format_failure_rate']:.3f} | {r['n_crops']} |")
        csv.append(f"{r['model']},{r['display']},{r['recognition_accuracy']:.4f},{r['format_failure_rate']:.4f},{r['n_crops']},{r['n_correct']},{r['n_error']}")
    (OUT_TABLES / "oracle_lvis_recognition_2026.md").write_text("\n".join(md) + "\n")
    (OUT_TABLES / "oracle_lvis_recognition_2026.csv").write_text("\n".join(csv) + "\n")

    # Figure: recognition accuracy by model (sorted)
    labels = [r["display"] for r in rows][::-1]
    vals = [r["recognition_accuracy"] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(labels, vals, color="#4C9F70")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.05); ax.set_xlabel("Oracle recognition accuracy (GT crops)")
    ax.set_title("LVIS-unseen Oracle-Localisation: recognition upper bound (2026)")
    ax.grid(axis="x", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "oracle_lvis_recognition_2026.pdf")
    fig.savefig(OUT_FIGS / "oracle_lvis_recognition_2026.png", dpi=150)
    print("[oracle] wrote table + figure; top:",
          ", ".join(f"{r['display']}={r['recognition_accuracy']:.2f}" for r in rows[:4]))
    return rows


# ----------------------------- ABLATION -----------------------------
def run_compute_metrics(vlm_dir: Path):
    # compute_metrics expects <vlm-dir>/<run>/per_image/<model>; wrap this single
    # run dir under a temp parent so promptA and promptB are scored in isolation.
    wrap = vlm_dir.parent / f"_eval_{vlm_dir.name}"
    wrap.mkdir(parents=True, exist_ok=True)
    link = wrap / vlm_dir.name
    if not link.exists():
        link.symlink_to(vlm_dir.resolve())
    empty_trad = PROJECT_ROOT / "results/raw/_empty_traditional"
    empty_trad.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PROJECT_ROOT / "experiments/compute_metrics.py"),
           "--dataset", "lvis-unseen", "--vlm-dir", str(wrap),
           "--traditional-dir", str(empty_trad),
           "--eval-labels-file", str(DYN)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    # pycocotools prints progress chatter to stdout before the JSON; the JSON is
    # the first object, and the chatter lines contain no braces.
    stdout = out.stdout
    brace = stdout.find("{")
    if brace < 0:
        print("compute_metrics produced no JSON:\nSTDOUT:\n", stdout[-2000:], "\nSTDERR:\n", out.stderr[-2000:])
        raise RuntimeError("compute_metrics produced no JSON")
    return json.loads(stdout[brace:])


def _index_metrics(data):
    """Return {model_name: {ap, ap50, label_accuracy, latency_mean}} from the
    compute_metrics JSON, which is {dataset: [ {model, ap, ap50, ...}, ... ]}."""
    idx = {}
    datasets = list(data.values()) if isinstance(data, dict) else [data]
    for entries in datasets:
        if not isinstance(entries, list):
            continue
        for m in entries:
            name = m.get("model")
            if not name:
                continue
            idx[name] = {
                "ap": m.get("ap"),
                "ap50": m.get("ap50"),
                "label_accuracy": m.get("label_accuracy"),
                "latency_mean": m.get("latency_mean"),
            }
    return idx


def parse_rate(vlm_dir: Path, model: str):
    """Fraction of images with a non-error, usable response."""
    pdir = vlm_dir / "per_image" / model
    if not pdir.exists():
        return None, None
    n = ok = 0
    lat = []
    for jf in pdir.glob("*.json"):
        rec = json.loads(jf.read_text())
        n += 1
        raw = rec.get("raw_response")
        err = isinstance(raw, dict) and bool(raw.get("error"))
        if not err:
            ok += 1
        if rec.get("latency_sec") is not None:
            lat.append(rec["latency_sec"])
    return (ok / n if n else None), (sum(lat) / len(lat) if lat else None)


def analyze_ablation(models):
    a = _index_metrics(run_compute_metrics(ABL_ROOT / "promptA"))
    b = _index_metrics(run_compute_metrics(ABL_ROOT / "promptB"))
    md = ["| Model | Strategy | AP | AP@0.5 | Label acc. | Parsed-OK % | Lat. mean (s) |",
          "|---|---|---:|---:|---:|---:|---:|"]
    csv = ["model,strategy,ap,ap50,label_accuracy,parsed_ok_pct,latency_mean_s"]
    def fmt(x, p=3):
        return f"{x:.{p}f}" if isinstance(x, (int, float)) else "--"
    for model in models:
        for tag, idx, vdir in (("A (single)", a, ABL_ROOT / "promptA"), ("B (iterative)", b, ABL_ROOT / "promptB")):
            m = idx.get(model, {})
            pr, lat = parse_rate(vdir, model)
            latv = m.get("latency_mean") if m.get("latency_mean") is not None else lat
            disp = DISPLAY.get(model, model)
            md.append(f"| {disp} | {tag} | {fmt(m.get('ap'))} | {fmt(m.get('ap50'))} | {fmt(m.get('label_accuracy'))} | {fmt((pr or 0)*100,1)} | {fmt(latv,2)} |")
            csv.append(f"{model},{tag},{fmt(m.get('ap'),4)},{fmt(m.get('ap50'),4)},{fmt(m.get('label_accuracy'),4)},{fmt((pr or 0)*100,1)},{fmt(latv,3)}")
    (OUT_TABLES / "ablation_lvis_reduced_2026.md").write_text("\n".join(md) + "\n")
    (OUT_TABLES / "ablation_lvis_reduced_2026.csv").write_text("\n".join(csv) + "\n")

    # Figure: Prompt A vs Prompt B AP, grouped bars
    import numpy as np
    disp = [DISPLAY.get(m, m) for m in models]
    ap_a = [a.get(m, {}).get("ap") or 0.0 for m in models]
    ap_b = [b.get(m, {}).get("ap") or 0.0 for m in models]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, ap_a, w, label="Prompt A (single-query)", color="#4C9F70")
    ax.bar(x + w / 2, ap_b, w, label="Prompt B (iterative)", color="#E07B53")
    ax.set_xticks(x); ax.set_xticklabels(disp, rotation=25, ha="right")
    ax.set_ylabel("AP (IoU 0.50:0.95)")
    ax.set_title("LVIS-unseen prompt ablation (reduced 40-image subset, 2026)")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUT_FIGS / "ablation_lvis_ap_2026.pdf")
    fig.savefig(OUT_FIGS / "ablation_lvis_ap_2026.png", dpi=150)
    print("[ablation] wrote table + figure; models:", models)
    return a, b


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["oracle", "ablation", "all"], default="all")
    ap.add_argument("--ablation-models", default="gpt-5.5,gpt-5.4-mini,claude-opus-4.8,claude-haiku-4.5,gemini-3.1-pro,gemini-3.5-flash")
    args = ap.parse_args()
    if args.what in ("oracle", "all"):
        analyze_oracle()
    if args.what in ("ablation", "all"):
        analyze_ablation([m.strip() for m in args.ablation_models.split(",") if m.strip()])
