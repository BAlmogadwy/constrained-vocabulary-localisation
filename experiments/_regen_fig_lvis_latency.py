#!/usr/bin/env python3
"""Regenerate fig_lvis_latency.pdf with batch-API models removed (reviewer Option A).

The six batch-API models' tabulated LVIS latencies were single-probe measurements
(mean==median exactly in metrics_summary.json), inconsistent with the paper's
latency-comparability policy. This figure now shows only the locally run detectors
and the synchronous OpenRouter/DashScope VLMs, matching Section 3's policy and the
corrected Table 1. Style matches the original (horizontal grouped bars, Blues).
Original preserved as fig_lvis_latency_with_batch.pdf.bak.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(r"C:\Users\user\zeroshot\paper_v2.4\paper\latex\figures\fig_lvis_latency.pdf")
REVBLUE = "#005AC8"

# (display name, mean, median, p95) — from results/metrics_summary.json (synchronous roster only)
DATA = [
    ("Grounding DINO",   0.293, 0.287, 0.319),
    ("OWL-ViT",          0.081, 0.080, 0.086),
    ("YOLO-World",       0.047, 0.035, 0.039),
    ("Gemma 3 27B",      7.376, 5.945, 18.432),
    ("Llama 4 Maverick", 2.279, 1.954, 5.155),
    ("Mistral Large 3",  3.263, 2.973, 5.218),
    ("Qwen-VL-Max",      3.763, 3.713, 4.838),
    ("Qwen2.5-VL-72B",   5.375, 2.597, 12.773),
    ("Qwen3-VL-235B",    3.955, 3.205, 8.809),
]
COLORS = {"mean": "#c6dbef", "median": "#6baed6", "p95": "#2171b5"}

fig, ax = plt.subplots(figsize=(7.6, 4.0))
names = [d[0] for d in DATA]
n = len(names)
h = 0.26
ys = np.arange(n)

for i, (key, off) in enumerate([("mean", -h), ("median", 0.0), ("p95", h)]):
    vals = [d[1 + i] for d in DATA]
    ax.barh(ys + off, vals, height=h * 0.92, color=COLORS[key], label=key, zorder=3)

ax.invert_yaxis()
ax.set_yticks(ys)
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel("Latency (s)", fontsize=9)
ax.set_ylabel("Model", fontsize=9)
ax.set_title("LVIS-unseen: per-image latency", fontsize=9.5)
ax.grid(axis="x", color="0.9", zorder=0)
ax.legend(title="Statistic", fontsize=8, title_fontsize=8.5, loc="lower right")
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
# (batch-API exclusion is stated in the caption; no in-figure note in the final version)

fig.tight_layout()
fig.savefig(OUT, format="pdf")
print("wrote", OUT)
