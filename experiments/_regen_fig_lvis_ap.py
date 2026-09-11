#!/usr/bin/env python3
"""Regenerate fig_lvis_ap.pdf: LVIS-unseen AP by model, near-zero-floor operating point.

Design: horizontal bars sorted by AP, coloured by family (VLM blue, detector
orange); value labels at bar ends; a dashed vertical line at the best detector
(YOLO-World); star markers on the models that are significantly above the best
detector under the image-level bootstrap of Section 4.1; three-part legend.
All values are the audited AP of Table 2. No new experiments: significance is
the established bootstrap result (Gemini 3.5 Flash and Gemini 3.1 Pro only).
Previous figure preserved as fig_lvis_ap_prev.pdf.bak on first run.
"""
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

OUT = Path(r"C:\Users\user\zeroshot\paper_v2.4\paper\latex\figures\fig_lvis_ap.pdf")
BAK = OUT.with_name("fig_lvis_ap_prev.pdf.bak")
if OUT.exists() and not BAK.exists():
    shutil.copy(OUT, BAK)

# (display name, AP, family, significant-above-best-detector) — audited AP, Table 2
DATA = [
    ("Gemini 3.5 Flash", 0.607, "vlm", True),
    ("Gemini 3.1 Pro",   0.565, "vlm", True),
    ("Qwen3-VL-235B",    0.369, "vlm", False),
    ("YOLO-World",       0.347, "det", False),
    ("Qwen2.5-VL-72B",   0.312, "vlm", False),
    ("Grounding DINO",   0.268, "det", False),
    ("Qwen-VL-Max",      0.262, "vlm", False),
    ("GPT-5.5",          0.249, "vlm", False),
    ("OWL-ViT",          0.161, "det", False),
    ("Claude Opus 4.8",  0.126, "vlm", False),
    ("GPT-5.4-mini",     0.110, "vlm", False),
    ("Llama 4 Maverick", 0.035, "vlm", False),
    ("Claude Haiku 4.5", 0.002, "vlm", False),
    ("Gemma 3 27B",      0.000, "vlm", False),
    ("Mistral Large 3",  0.000, "vlm", False),
]

BEST_DET = ("YOLO-World", 0.347)
VLM_BLUE = "#4C72B0"
DET_ORANGE = "#DD8452"
FAMILY = {"vlm": VLM_BLUE, "det": DET_ORANGE}

plt.rcParams.update({"font.size": 9, "axes.edgecolor": "0.4"})
fig, ax = plt.subplots(figsize=(7.6, 4.4))

names = [d[0] for d in DATA]
aps = [d[1] for d in DATA]
colors = [FAMILY[d[2]] for d in DATA]
ys = range(len(DATA))

ax.barh(list(ys), aps, color=colors, height=0.72, zorder=3)
ax.invert_yaxis()
ax.set_yticks(list(ys))
ax.set_yticklabels(names, fontsize=8.5)
ax.set_xlabel("AP (IoU 0.50:0.95) at the near-zero floor, LVIS-unseen (396 images)", fontsize=8.5)
ax.set_title("Zero-shot detection AP at a fair, near-zero-floor operating point", fontsize=9.5)
ax.set_xlim(0, 0.80)
ax.grid(axis="x", color="0.88", zorder=0)
ax.grid(axis="y", visible=False)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)

# best-detector reference line
ax.axvline(BEST_DET[1], color=DET_ORANGE, linestyle="--", linewidth=1.3, zorder=2)
ax.text(BEST_DET[1] + 0.010, 10.6, f"best detector\n({BEST_DET[0]}, {BEST_DET[1]:.3f})",
        color=DET_ORANGE, fontsize=7.5, va="center", ha="left")

# value labels and significance stars
for i, (_, ap, _, sig) in enumerate(DATA):
    label = f"{ap:.3f}" + ("  ★" if sig else "")
    ax.text(ap + 0.008, i, label, va="center", fontsize=8,
            color="0.15", fontweight="bold" if sig else "normal")

legend_handles = [
    Patch(facecolor=VLM_BLUE, label="Vision-language model"),
    Patch(facecolor=DET_ORANGE, label="Open-vocabulary detector"),
    Line2D([0], [0], marker="*", color="none", markerfacecolor="0.15",
           markeredgecolor="0.15", markersize=9,
           label="significantly $>$ Grounding DINO (paired bootstrap)"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=7.8, frameon=True,
          borderpad=0.6, handletextpad=0.5)

fig.tight_layout()
fig.savefig(OUT, format="pdf")
print("wrote", OUT)
