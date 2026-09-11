#!/usr/bin/env python3
"""Regenerate fig_coco_cost.pdf (COCO cost vs AP scatter).

Data recovered exactly from the original figure's vector content (dot centres
mapped through the axis ticks; all AP values verified against Table 3) plus the
cost anchors stated in the text (Qwen3 $0.26, Gemini 3.1 Pro $1.80, GPT-5.5 $5.68).
Layout fix: only the Pareto-relevant models are labelled directly; the crowded
low-cost cluster moves to a marker legend. Original saved as *_alllabels.pdf.bak.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"C:\Users\user\zeroshot\paper_v2.4\paper\latex\figures\fig_coco_cost.pdf")

GREEN = "#16A34A"   # proprietary API (as original)
ORANGE = "#F97316"  # open-weight via OpenRouter (as original)

# (name, cost USD, AP, colour, direct-label?)
DATA = [
    ("Gemini 3.1 Pro",   1.80, 0.428, GREEN,  True),
    ("Gemini 3.5 Flash", 2.87, 0.413, GREEN,  True),
    ("Qwen3-VL-235B",    0.26, 0.321, ORANGE, True),
    ("GPT-5.5",          5.68, 0.278, GREEN,  True),
    ("Qwen2.5-VL-72B",   0.45, 0.198, ORANGE, False),
    ("Claude Opus 4.8",  2.10, 0.145, GREEN,  False),
    ("GPT-5.4-mini",     0.25, 0.091, GREEN,  False),
    ("Llama 4 Maverick", 0.21, 0.047, ORANGE, False),
    ("Claude Haiku 4.5", 0.43, 0.006, GREEN,  False),
    ("Gemma 3 27B",      0.07, 0.001, ORANGE, False),
    ("Mistral Large 3",  0.38, 0.000, ORANGE, False),
]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]

fig, ax = plt.subplots(figsize=(7.4, 4.9))
mi = 0
legend_handles = []
for name, cost, ap, col, direct in DATA:
    if direct:
        ax.scatter([cost], [ap], s=55, color=col, zorder=3)
        dx, ha = (0.08, "left") if name != "GPT-5.5" else (-0.08, "right")
        ax.annotate(name, (cost, ap), xytext=(cost + dx, ap + 0.006),
                    fontsize=9, ha=ha, va="bottom")
    else:
        h = ax.scatter([cost], [ap], s=55, color=col, marker=MARKERS[mi],
                       zorder=3, label=name)
        legend_handles.append(h)
        mi += 1

ax.set_xlabel("Estimated cost for 901 images (USD)", fontsize=10)
ax.set_ylabel("AP", fontsize=10)
ax.set_title("COCO-unseen: cost vs AP", fontsize=10.5)
ax.set_xlim(-0.15, 6.1)
ax.set_ylim(-0.015, 0.45)
ax.grid(color="0.92", zorder=0)
leg = ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5,
                frameon=True, title="Low-cost cluster", title_fontsize=9)
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)

fig.tight_layout()
fig.savefig(OUT, format="pdf")
print("wrote", OUT)
