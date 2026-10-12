"""Result panel for the graphical abstract: the 16 permutation tests of Table 2 as a 4 x 4 grid.

Run from the repository root after the permutation table is final:
    python src/ga_result_grid.py [TABLE_CSV]
Rows are the modalities and columns the comparisons. Each cell shows the permutation p of that
test and is coloured by whether it is below 0.05 (uncorrected) or below the Bonferroni threshold
0.05/16 (none is). Writes results/figures/ga_result_grid.png (600 dpi) and .svg (editable),
plus transparent copies (_transparent.png/.svg).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

TABLE_CSV = sys.argv[1] if len(sys.argv) > 1 else "results/permutation/publication_f1_permutation_table_raw.csv"
OUT = Path("results/figures/ga_result_grid")
THRESHOLD = 0.05 / 16

ROWS = {"EEG": "EEG", "fMRI": "fMRI", "sMRI": "Structural MRI", "Clinical": "Non-imaging"}
COLS = {"N vs A+P-": "N vs\nA+P−", "N vs A+P+": "N vs\nA+P+", "A+P- vs A+P+": "A+P− vs\nA+P+",
        "N vs (A+P- and A+P+)": "N vs all\ncarriers"}
INK, MUTED = "#262626", "#6b6b6b"
FILL = {"none": "#EBEBEB", "uncorrected": "#F2C14E", "corrected": "#B5651D"}  # grey, amber, dark amber

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK})
table = pd.read_csv(TABLE_CSV)
table["Modality"] = table["Modality"].replace({"Psychometric": "Clinical", "Non-imaging": "Clinical"})
p = table.set_index(["Modality", "Group Comparison"])["p_value_raw"]
assert len(p) == 16, "expected 16 tests"

CW, RH, GAP = 1.6, 0.8, 0.06  # cell width and height (data units), white gap between cells
fig, ax = plt.subplots(figsize=(4.4, 3.4))
for i, mod in enumerate(ROWS):
    for j, comp in enumerate(COLS):
        pv = p[(mod, comp)]
        level = "corrected" if pv < THRESHOLD else "uncorrected" if pv < 0.05 else "none"
        x, y = j * CW, -i * RH
        ax.add_patch(Rectangle((x, y), CW - GAP, -(RH - GAP), facecolor=FILL[level], edgecolor="none"))
        ax.text(x + (CW - GAP) / 2, y - (RH - GAP) / 2, f"{pv:.3f}", ha="center", va="center",
                fontsize=9, color=INK, fontweight="bold" if level != "none" else "normal")
for i, label in enumerate(ROWS.values()):
    ax.text(-0.12, -i * RH - (RH - GAP) / 2, label, ha="right", va="center", fontsize=9)
for j, label in enumerate(COLS.values()):
    ax.text(j * CW + (CW - GAP) / 2, 0.1, label, ha="center", va="bottom", fontsize=8.5, linespacing=1.15)

# legend and the take-home line below the grid
legend = [("none", "p \u2265 0.05"), ("uncorrected", "p < 0.05, uncorrected"),
          ("corrected", "p < 0.0031, corrected (none)")]
top = -4 * RH - 0.35
for k, (level, text) in enumerate(legend):
    y = top - 0.36 * k
    ax.add_patch(Rectangle((0, y), 0.32, -0.24, facecolor=FILL[level], edgecolor="none"))
    ax.text(0.45, y - 0.12, text, va="center", fontsize=8, color=MUTED)
ax.text(2 * CW - GAP / 2, top - 0.36 * 3 - 0.2, "0 of 16 tests survive correction", ha="center",
        fontsize=9.5, fontweight="bold", va="center")

ax.set_xlim(-2.3, 4 * CW + 0.05)
ax.set_ylim(top - 0.36 * 3 - 0.45, 0.85)
ax.axis("off")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
fig.savefig(OUT.parent / f"{OUT.name}_transparent.png", dpi=600, bbox_inches="tight", transparent=True)
fig.savefig(OUT.parent / f"{OUT.name}_transparent.svg", bbox_inches="tight", transparent=True)
print(f"Wrote {OUT}.png and .svg")
