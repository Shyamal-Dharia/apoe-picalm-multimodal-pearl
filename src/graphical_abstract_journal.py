"""Graphical abstract in the author's PowerPoint design, at Elsevier proportions.

Elsevier asks for at least 1328 x 531 px (w x h, 2.5:1), readable at 13 x 5 cm. This draws the
author's layout (participants -> modalities -> 16 tests) at 13.28 x 5.31 in, 300 dpi
(3984 x 1593 px), with large type. Run from the repository root after src/graphical_abstract.py
(which writes the four transparent icons it reuses):
    python src/graphical_abstract.py && python src/graphical_abstract_journal.py [TABLE_CSV]
Writes results/figures/graphical_abstract_journal.png, .pdf, .svg and .tiff.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

TABLE_CSV = sys.argv[1] if len(sys.argv) > 1 else "results/permutation/publication_f1_permutation_table_raw.csv"
FIGDIR = Path("results/figures")
OUT = FIGDIR / "graphical_abstract_journal"
MINUS, ARROW = "−", "→"
# colours sampled from the author's PowerPoint version
BOX, PIPE, EDGE, INK, MUTED = "#F2F2F2", "#D9D9D9", "#1F2A30", "#111111", "#555555"
PEOPLE = {"N": "#ABDB9D", f"A+P{MINUS}": "#FF8F8F", "A+P+": "#E59EDD"}
FILL = {"none": "#EBEBEB", "uncorrected": "#F2C14E", "corrected": "#B5651D"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

fig = plt.figure(figsize=(13.28, 5.31))
fig.patch.set_facecolor("white")
T = fig.transFigure


def box(x0, x1, y0=0.04, y1=0.96, face=BOX, lw=1.6):
    fig.patches.append(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, transform=T, facecolor=face, edgecolor=EDGE,
                                      lw=lw, boxstyle="round,pad=0,rounding_size=0.025", zorder=-2))


def arrow(x0, x1, y=0.50):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=T, arrowstyle="-|>", mutation_scale=34,
                                       lw=4, color="black", zorder=5))


def image(path, rect):
    ax = fig.add_axes(rect)
    ax.imshow(np.asarray(Image.open(path).convert("RGBA")))
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------- participants
box(0.006, 0.200)
fig.text(0.103, 0.915, "Participants", ha="center", va="top", fontsize=24, fontweight="bold")
fig.text(0.103, 0.775, "PEARL-Neuro\n64 Healthy\nMiddle-Aged Adults", ha="center", va="top", fontsize=15,
         fontweight="bold", linespacing=1.2)
ax = fig.add_axes([0.014, 0.08, 0.178, 0.40])
ax.set_xlim(0, 3)
ax.set_ylim(-0.45, 1.45)
ax.set_aspect("equal")
ax.axis("off")
for k, (g, col) in enumerate(PEOPLE.items()):
    cx = 0.5 + k
    ax.add_patch(Circle((cx, 1.18), 0.19, facecolor=col, edgecolor=EDGE, lw=1.2))
    ax.add_patch(FancyBboxPatch((cx - 0.28, 0.18), 0.56, 0.72, boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=col, edgecolor=EDGE, lw=1.2))
    ax.text(cx, 0.06, g, ha="center", va="top", fontsize=16, fontweight="bold")
arrow(0.203, 0.233)

# ---------------------------------------------------------------- modalities
box(0.236, 0.656)
fig.text(0.446, 0.915, "Modalities", ha="center", va="top", fontsize=24, fontweight="bold")
icons = [("EEG", "eeg"), ("fMRI", "fmri"), ("sMRI", "structural_mri"), ("Non-Imaging", "non_imaging")]
w, gap = 0.098, 0.004
for k, (label, name) in enumerate(icons):
    cx = 0.248 + w / 2 + k * (w + gap)
    fig.text(cx, 0.775, label, ha="center", va="top", fontsize=15.5, fontweight="bold")
    image(FIGDIR / f"ga_icon_{name}.png", [cx - w / 2 + 0.004, 0.335, w - 0.008, 0.38])
fig.text(0.446, 0.19, f"Leave-one-subject-out classification\n{ARROW} 1,000 permutation tests\n"
         f"{ARROW} one correction across 16 tests", ha="center", va="center", fontsize=14.5, fontweight="bold",
         linespacing=1.35, bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.4", facecolor=PIPE, edgecolor=EDGE,
                                     lw=1.2))
arrow(0.659, 0.689)

# ---------------------------------------------------------------- 16 tests
box(0.692, 0.994, face="white", lw=1.2)
fig.text(0.843, 0.915, "Results", ha="center", va="top", fontsize=24, fontweight="bold")
table = pd.read_csv(TABLE_CSV)
table["Modality"] = table["Modality"].replace({"Psychometric": "Clinical", "Non-imaging": "Clinical"})
p = table.set_index(["Modality", "Group Comparison"])["p_value_raw"]
assert len(p) == 16, "expected 16 tests"
rows = {"EEG": "EEG", "fMRI": "fMRI", "sMRI": "sMRI", "Clinical": "Non-\nimaging"}
cols = {"N vs A+P-": f"N vs\nA+P{MINUS}", "N vs A+P+": "N vs\nA+P+", "A+P- vs A+P+": f"A+P{MINUS}\nvs A+P+",
        "N vs (A+P- and A+P+)": "N vs all\ncarriers"}
ax = fig.add_axes([0.697, 0.06, 0.292, 0.74])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
X0, CW, RH, G, Y0 = 0.175, 0.205, 0.135, 0.012, 0.80
for i, mod in enumerate(rows):
    for j, comp in enumerate(cols):
        pv = p[(mod, comp)]
        level = "corrected" if pv < 0.05 / 16 else "uncorrected" if pv < 0.05 else "none"
        x, y = X0 + j * CW, Y0 - i * RH
        ax.add_patch(Rectangle((x, y), CW - G, -(RH - G), facecolor=FILL[level], edgecolor="none"))
        ax.text(x + (CW - G) / 2, y - (RH - G) / 2, f"{pv:.3f}", ha="center", va="center", fontsize=14,
                fontweight="bold" if level != "none" else "normal")
for i, label in enumerate(rows.values()):
    ax.text(X0 - 0.02, Y0 - i * RH - (RH - G) / 2, label, ha="right", va="center", fontsize=13,
            linespacing=1.0)
for j, label in enumerate(cols.values()):
    ax.text(X0 + j * CW + (CW - G) / 2, Y0 + 0.01, label, ha="center", va="bottom", fontsize=11.5,
            linespacing=1.05)
legend = [("uncorrected", "p < 0.05, uncorrected"), ("corrected", "p < 0.0031, corrected (none)")]
for k, (level, text) in enumerate(legend):
    ly = Y0 - 4 * RH - 0.045 - 0.07 * k
    ax.add_patch(Rectangle((X0, ly), 0.045, -0.045, facecolor=FILL[level], edgecolor="none"))
    ax.text(X0 + 0.065, ly - 0.0225, text, va="center", fontsize=12, color=MUTED)
ax.text(0.50, 0.03, "0 of 16 tests survive correction", ha="center", va="center", fontsize=14,
        fontweight="bold")

for ext, kw in ((".png", {"dpi": 300}), (".tiff", {"dpi": 300}), (".pdf", {}), (".svg", {})):
    fig.savefig(OUT.with_suffix(ext), facecolor="white", **kw)
print(f"Wrote {OUT}.png/.tiff/.pdf/.svg")
