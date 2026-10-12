"""Graphical abstract: participants -> four modalities tested the same way -> the 16 tests.

Run from the repository root (needs MNE for the EEG sensor positions, e.g. ~/envs/mmeeg):
    python src/graphical_abstract.py [TABLE_CSV]
Inputs: the permutation table (Table 2), data/EEG_data/channel_names.txt, and two brain renderings
in figures/ (nilearn_surface_schaefer400_lateral_colors.png for fMRI parcels,
nilearn_surface_docs_gray_grid.png for structure). Writes results/figures/graphical_abstract
.png (300 dpi, 3600 x 1440 px, above Elsevier's 1328 x 531 minimum), .pdf and .svg, a
transparent copy, and the four modality icons alone (ga_icon_*.png/.svg, transparent).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from PIL import Image

TABLE_CSV = sys.argv[1] if len(sys.argv) > 1 else "results/permutation/publication_f1_permutation_table_raw.csv"
OUT = Path("results/figures/graphical_abstract")
INK, MUTED, LINE = "#262626", "#6b6b6b", "#BDBDBD"
MINUS, ARROW, BULLET = "−", "→", "•"
GROUPS = {"N": ("#1B9E77", 25), f"A+P{MINUS}": ("#D95F02", 23), "A+P+": ("#7570B3", 16)}  # paper colours
FILL = {"none": "#EBEBEB", "uncorrected": "#F2C14E", "corrected": "#B5651D"}
REGION_COLORS = {"frontal": "#9ACBFF", "temporal": "#FFB6D9", "central": "#98EBA8",
                 "parietal": "#E8DCA6", "occipital": "#D8BFD8"}  # as in Supplementary Figure S2
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def crop(path, box):
    """Crop a region (fractions x0, y0, x1, y1) of an image to its non-white content and make the
    white background transparent (soft edge: alpha rises as the darkest channel drops below 250)."""
    im = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = im.shape
    im = im[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
    ys, xs = np.where((im < 245).any(axis=2))
    im = im[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    alpha = np.clip((250.0 - im.min(axis=2)) / 30.0, 0, 1)
    return np.dstack([im / 255.0, alpha])


def eeg_region(name):
    """Region of a 10-05 channel name, as in Supplementary Figure S2."""
    n = name.upper()
    if n.startswith("O"):
        return "occipital"
    if n.startswith("T"):
        return "temporal"
    if n.startswith("C"):
        return "central"
    if n.startswith("P"):
        return "parietal"
    return "frontal"  # Fp, AF, F, FC, FT, FFC, FCC, FFT, FTT


def eeg_positions():
    names = [c.strip() for c in open("data/EEG_data/channel_names.txt") if c.strip()]
    pos = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    lower = {k.lower(): v for k, v in pos.items()}
    xy, keep = [], []
    for nm in names:
        p = lower.get(nm.lower())
        if p is None:
            continue
        x, y, z = p / np.linalg.norm(p)
        theta, phi = np.arccos(np.clip(z, -1, 1)), np.arctan2(y, x)  # azimuthal equidistant projection
        xy.append((theta * np.cos(phi), theta * np.sin(phi)))
        keep.append(nm)
    return np.array(xy), keep


def arrow(fig, x0, x1, y):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=22, lw=2.2, color=MUTED))


ICON_NAMES = ["eeg", "fmri", "structural_mri", "non_imaging"]
STATE_COLORS = ["#B3CDE3", "#FBB4AE", "#CCEBC5"]  # pastel blue, pink, green (fMRI states)


def draw_icon(icon, k):
    """Draw modality icon k (0 EEG, 1 fMRI, 2 structural MRI, 3 non-imaging) into an axes."""
    icon.axis("off")
    if k == 0:  # EEG sensors coloured by region
        xy, names = eeg_positions()
        icon.add_patch(Circle((0, 0), np.pi / 2, fill=False, lw=1.2, color=MUTED))
        icon.plot([-0.18, 0, 0.18], [np.pi / 2 - 0.02, np.pi / 2 + 0.25, np.pi / 2 - 0.02], color=MUTED, lw=1.2)
        icon.scatter(xy[:, 0], xy[:, 1], s=15, c=[REGION_COLORS[eeg_region(n)] for n in names],
                     edgecolors="#555555", linewidths=0.3, zorder=3)
        icon.set_xlim(-2.35, 2.35)
        icon.set_ylim(-2.0, 2.0)
        icon.set_aspect("equal")
    elif k == 1:  # fMRI parcels and a sequence of states (dwell time)
        icon.imshow(crop("figures/nilearn_surface_schaefer400_lateral_colors.png", (0, 0, 0.5, 1)),
                    extent=(0.05, 0.95, 0.32, 1.0))
        states = [0, 0, 1, 1, 1, 2, 0, 0, 2, 1]
        for j, st in enumerate(states):
            icon.add_patch(Rectangle((0.03 + j * 0.094, 0.05), 0.088, 0.16, color=STATE_COLORS[st]))
        icon.set_xlim(0, 1)
        icon.set_ylim(0, 1)
    elif k == 2:  # structure, plain surface
        icon.imshow(crop("figures/nilearn_surface_docs_gray_grid.png", (0.38, 0.05, 0.70, 0.45)),
                    extent=(0.05, 0.95, 0.12, 0.92))
        icon.set_xlim(0, 1)
        icon.set_ylim(0, 1)
    else:  # questionnaire and blood drop
        icon.add_patch(FancyBboxPatch((0.10, 0.12), 0.42, 0.72, boxstyle="round,pad=0,rounding_size=0.04",
                                      facecolor="white", edgecolor=MUTED, lw=1.4))
        icon.add_patch(Rectangle((0.22, 0.80), 0.18, 0.08, color=MUTED))
        for yy in (0.66, 0.52, 0.38, 0.24):
            icon.add_patch(Rectangle((0.17, yy), 0.06, 0.06, facecolor="white", edgecolor=MUTED, lw=1))
            icon.plot([0.27, 0.46], [yy + 0.03, yy + 0.03], color=MUTED, lw=1.4)
        drop = MPath([(0.76, 0.80), (0.70, 0.62), (0.60, 0.48), (0.60, 0.36), (0.60, 0.22), (0.68, 0.14),
                      (0.76, 0.14), (0.84, 0.14), (0.92, 0.22), (0.92, 0.36), (0.92, 0.48), (0.82, 0.62),
                      (0.76, 0.80)], [MPath.MOVETO] + [MPath.CURVE4] * 12)
        icon.add_patch(PathPatch(drop, facecolor="#C44E52", edgecolor="none"))
        icon.set_xlim(0, 1)
        icon.set_ylim(0, 1)


fig = plt.figure(figsize=(12, 4.8))
fig.patch.set_facecolor("white")

# ---------------------------------------------------------------- column 1, participants
fig.text(0.09, 0.94, "Participants", ha="center", va="top", fontsize=15, fontweight="bold")
fig.text(0.09, 0.80, "PEARL-Neuro\n64 healthy\nmiddle-aged adults", ha="center", va="top", fontsize=11.5,
         linespacing=1.25)
ax = fig.add_axes([0.005, 0.15, 0.17, 0.40])
ax.set_xlim(0, 3)
ax.set_ylim(-0.75, 1.25)
ax.set_aspect("equal")
ax.axis("off")
for k, (g, (col, n)) in enumerate(GROUPS.items()):
    cx = 0.5 + k
    ax.add_patch(Circle((cx, 0.98), 0.17, color=col))
    ax.add_patch(FancyBboxPatch((cx - 0.24, 0.18), 0.48, 0.56, boxstyle="round,pad=0,rounding_size=0.16",
                                color=col))
    ax.text(cx, 0.02, g, ha="center", va="top", fontsize=11.5, fontweight="bold")
    ax.text(cx, -0.38, f"n = {n}", ha="center", va="top", fontsize=10.5, color=MUTED)
arrow(fig, 0.172, 0.197, 0.60)

# ---------------------------------------------------------------- column 2, four modalities
x0, w, gap = 0.205, 0.1, 0.006
mid2 = x0 + 2 * (w + gap) - gap / 2
fig.text(mid2, 0.94, "Four modalities, same participants", ha="center", va="top", fontsize=15,
         fontweight="bold")
labels = [("EEG", "within-state HFD\nvariability"), ("fMRI", "dwell time in\nconnectivity states"),
          ("Structural MRI", "1,610 regional\nmeasures"), ("Non-imaging", "75 psychometric,\nblood and health")]
for k, (title, sub) in enumerate(labels):
    left = x0 + k * (w + gap)
    fig.patches.append(FancyBboxPatch((left, 0.33), w, 0.53, transform=fig.transFigure, facecolor="white",
                                      edgecolor=LINE, lw=1, boxstyle="round,pad=0,rounding_size=0.01",
                                      zorder=-1))  # behind the icon axes
    fig.text(left + w / 2, 0.84, title, ha="center", va="top", fontsize=11, fontweight="bold")
    fig.text(left + w / 2, 0.415, sub, ha="center", va="top", fontsize=9, color=MUTED, linespacing=1.15)
    icon = fig.add_axes([left + 0.008, 0.44, w - 0.016, 0.33])
    draw_icon(icon, k)
fig.text(mid2, 0.225, f"leave-one-subject-out classification  {ARROW}  1,000 permutations\n"
         f"{ARROW}  one correction across all 16 tests", ha="center", va="center", fontsize=10.5,
         linespacing=1.35, bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=LINE, lw=1))
arrow(fig, 0.618, 0.643, 0.60)

# ---------------------------------------------------------------- column 3, results
table = pd.read_csv(TABLE_CSV)
table["Modality"] = table["Modality"].replace({"Psychometric": "Clinical", "Non-imaging": "Clinical"})
p = table.set_index(["Modality", "Group Comparison"])["p_value_raw"]
assert len(p) == 16, "expected 16 tests"
rows = {"EEG": "EEG", "fMRI": "fMRI", "sMRI": "Structural MRI", "Clinical": "Non-imaging"}
cols3 = {"N vs A+P-": f"N vs\nA+P{MINUS}", "N vs A+P+": "N vs\nA+P+", "A+P- vs A+P+": f"A+P{MINUS} vs\nA+P+",
         "N vs (A+P- and A+P+)": "N vs all\ncarriers"}
threshold = 0.05 / 16
fig.text(0.815, 0.94, "16 tests (permutation p)", ha="center", va="top", fontsize=15, fontweight="bold")
ax = fig.add_axes([0.645, 0.25, 0.34, 0.58])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
CW, RH, G, X0 = 0.155, 0.19, 0.012, 0.30
for i, mod in enumerate(rows):
    for j, comp in enumerate(cols3):
        pv = p[(mod, comp)]
        level = "corrected" if pv < threshold else "uncorrected" if pv < 0.05 else "none"
        x, y = X0 + j * CW, 0.78 - i * RH
        ax.add_patch(Rectangle((x, y), CW - G, -(RH - G), facecolor=FILL[level], edgecolor="none"))
        ax.text(x + (CW - G) / 2, y - (RH - G) / 2, f"{pv:.3f}", ha="center", va="center", fontsize=10.5,
                fontweight="bold" if level != "none" else "normal")
for i, label in enumerate(rows.values()):
    ax.text(X0 - 0.02, 0.78 - i * RH - (RH - G) / 2, label, ha="right", va="center", fontsize=11)
for j, label in enumerate(cols3.values()):
    ax.text(X0 + j * CW + (CW - G) / 2, 0.80, label, ha="center", va="bottom", fontsize=9.5, linespacing=1.1)
legend = [("uncorrected", "p < 0.05, uncorrected"), ("corrected", "p < 0.0031, corrected (none)")]
for k, (level, text) in enumerate(legend):
    ly = -0.06 - 0.10 * k
    ax.add_patch(Rectangle((X0, ly), 0.035, -0.06, facecolor=FILL[level], edgecolor="none", clip_on=False))
    ax.text(X0 + 0.05, ly - 0.03, text, va="center", fontsize=9.5, color=MUTED)

# ---------------------------------------------------------------- take-home line
fig.text(0.5, 0.05, f"0 of 16 tests survive correction  {BULLET}  3 leads for replication  {BULLET}  "
         "no statistical indirect associations", ha="center", va="center", fontsize=12, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF6DD", edgecolor="#F2C14E", lw=1.2))

OUT.parent.mkdir(parents=True, exist_ok=True)
for ext, kw in ((".png", {"dpi": 300}), (".pdf", {}), (".svg", {})):
    fig.savefig(OUT.with_suffix(ext), facecolor="white", **kw)
fig.savefig(OUT.parent / "graphical_abstract_transparent.png", dpi=300, transparent=True)
for k, name in enumerate(ICON_NAMES):  # each modality image on its own
    f1, a1 = plt.subplots(figsize=(2.4, 2.4))
    draw_icon(a1, k)
    for ext in (".png", ".svg"):
        f1.savefig(OUT.parent / f"ga_icon_{name}{ext}", dpi=600, transparent=True, bbox_inches="tight")
    plt.close(f1)
print(f"Wrote {OUT}.png/.pdf/.svg and ga_icon_*.png/.svg")
