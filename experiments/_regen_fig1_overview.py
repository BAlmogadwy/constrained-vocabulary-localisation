#!/usr/bin/env python3
"""Regenerate figure1_v1.png (framework overview schematic).

Faithful replica of the original layout/style. Boxes are sized so labels fit
at readable sizes; after layout every label's rendered width is measured and
auto-shrunk if needed, then sizes are harmonised within each box, so no text
can cross a shape boundary and lines in one box share one size.
Original preserved as figure1_v1_original.png.bak.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from collections import defaultdict

OUT = Path(r"C:\Users\user\zeroshot\paper_v2.4\paper\latex\figures\figure1_v1.png")

BLUE = "#2E5FA3"       # box borders / arrows (as original)
ORANGE = "#E8701A"     # reliability layer accent (as original)
ORANGE_FILL = "#FDEEDC"
TEXT = "#1a1a1a"

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

fig, ax = plt.subplots(figsize=(20.48, 7.68), dpi=100)
ax.set_xlim(0, 2048); ax.set_ylim(0, 768)
ax.invert_yaxis(); ax.axis("off")
fig.patch.set_facecolor("white")

def box(x, y, w, h, ec=BLUE, fc="white", lw=2.6, rad=18):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rad}",
                       edgecolor=ec, facecolor=fc, linewidth=lw, zorder=2)
    ax.add_patch(p)
    return p

FITS = []  # (text artist, max width in data units, group key)

def label(x, y, s, size=19, color=TEXT, weight="normal", ha="center",
          fit_w=None, group=None):
    t = ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
                ha=ha, va="center", zorder=3, linespacing=1.35)
    if fit_w:
        FITS.append((t, fit_w, group))
    return t

def arrow(x1, y1, x2, y2, color=BLUE, lw=3.2):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>,head_width=5,head_length=9",
                        color=color, linewidth=lw, zorder=1, shrinkA=0, shrinkB=0)
    ax.add_patch(a)

MID = 308
PAD = 12  # min gap between text and box border, in data units (px)

# 1. Input image
box(30, 218, 220, 182)
label(140, MID, "Input image", size=20, fit_w=220, group="input")
arrow(255, MID, 289, MID)

# 2. Per-image candidate class list
box(294, 218, 300, 182)
label(444, MID - 18, "Per-image", size=18, fit_w=300, group="perimg")
label(444, MID + 20, "candidate class list", size=18, fit_w=300, group="perimg")
arrow(599, MID, 629, MID)

# 3. Perception module
box(634, 145, 392, 305)
label(830, 192, "Perception module", size=21, weight="bold", fit_w=392, group="pm-title")
box(660, 238, 340, 78, lw=2.2, rad=12)
label(830, 277, "Open-vocabulary detector", size=16, fit_w=340, group="pm-a")
box(660, 338, 340, 78, lw=2.2, rad=12)
label(830, 377, "Vision-language model", size=16, fit_w=340, group="pm-b")
arrow(1031, MID, 1083, MID)

# Diagnostics box + arrow up
box(565, 518, 530, 118)
label(830, 560, "Diagnostics: oracle-localisation,", size=19, fit_w=530, group="diag")
label(830, 598, "prompt-strategy ablation", size=19, fit_w=530, group="diag")
arrow(830, 512, 830, 458)

# 4. Reliability layer
box(1088, 143, 340, 320, ec=ORANGE, fc=ORANGE_FILL, lw=3.0)
label(1258, 205, "Output reliability\nlayer", size=21, color=ORANGE, weight="bold",
      fit_w=340, group="rel-title")
label(1258, 310, "VLM coordinate\nnormalisation", size=19, color=ORANGE,
      fit_w=340, group="rel-body")
label(1258, 398, "Detector score-floor\naudit", size=19, color=ORANGE,
      fit_w=340, group="rel-body")
arrow(1428, MID, 1464, MID, color=BLUE)

# 5. Evaluation
box(1470, 218, 270, 182)
label(1605, 270, "Evaluation:", size=18, fit_w=270, group="eval-h")
label(1605, 308, "AP · cand.-label cov.", size=16, fit_w=270, group="eval")
label(1605, 346, "latency · cost", size=16, fit_w=270, group="eval")
arrow(1745, MID, 1781, MID)

# 6. Decision policy
box(1786, 218, 240, 182)
label(1906, 289, "Expert-system", size=18, fit_w=240, group="dec")
label(1906, 327, "decision policy", size=18, fit_w=240, group="dec")

# --- auto-fit: shrink any label whose rendered width crosses its box ---
fig.canvas.draw()
r = fig.canvas.get_renderer()
scale = ax.get_window_extent(r).width / 2048.0  # px per data unit
for t, w, _ in FITS:
    allowed = (w - 2 * PAD) * scale
    while t.get_window_extent(renderer=r).width > allowed and t.get_fontsize() > 10:
        t.set_fontsize(t.get_fontsize() - 0.5)

# --- harmonise: lines sharing a box (group) get that group's smallest size ---
groups = defaultdict(list)
for t, _, g in FITS:
    if g:
        groups[g].append(t)
for g, ts in groups.items():
    m = min(t.get_fontsize() for t in ts)
    for t in ts:
        t.set_fontsize(m)

fig.savefig(OUT, dpi=100, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
for t, w, g in FITS:
    print(f"  [{g or '-':9s}] {t.get_text().splitlines()[0][:34]!r:38s} -> {t.get_fontsize():.1f}pt")
