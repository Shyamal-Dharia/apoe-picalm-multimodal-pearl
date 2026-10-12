"""Figures for the paper (n = 64) and the group statistics drawn on them.

Run from the repository root after the EEG, fMRI and table scripts:
    python src/paper_figures.py
Writes PNGs to results/figures/ and the statistics to results/figures/figure_stats.csv.

  fig_eeg_hfd_variability_{N,AposPminus,AposPpos}.png
                               per-subject mean within-state HFD variability (SD of the distance
                               to the state centroid), averaged over the states selected in all
                               five seeds, one file per direction (the group with the highest
                               mean in the state); directions without states are left out
  fig_fmri_dwell_time_{N,carrier}.png
                               per-subject dwell time, averaged over the significant states of
                               all five seeds, N-dominant and carrier-dominant states
  (one file per panel; the LaTeX places them as subfigures)
  fig_smri_regions.png         the sMRI regions of Table S4 (Figure 4)
  fig_nonimaging.png           the non-imaging measures of Table 3 (Figure 5)

Brackets show JASP-style ANOVA post hoc contrasts (Bonferroni over the three contrasts,
src/group_stats.py) with p_bonf < .10. sMRI brackets are the contrasts listed in Table S4.
"""
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from group_stats import GROUPS, kruskal_p, posthoc, split_by_group

OUT_DIR = Path("results/figures")
SAMPLE_CSV = "data/participants_n64.csv"
EEG_FILES = {  # panel name -> (input, output suffix)
    "N-highest states": ("results/eeg/eeg_subject_mean_variability_N_higher_multiseed.csv", "N"),
    "A+P$-$-highest states": ("results/eeg/eeg_subject_mean_variability_A_plus_P_minus_higher_multiseed.csv",
                              "AposPminus"),
    "A+P+-highest states": ("results/eeg/eeg_subject_mean_variability_A_plus_P_plus_higher_multiseed.csv",
                            "AposPpos"),
}
PANEL_SIZE = (3.3, 2.9)  # inches, one panel per file
FMRI_SEEDS = [9, 42, 123, 2021, 2022]
FMRI_SEED_CSV = ("results/fmri/fMRI_significant_clusters_60s_75overlap_multiseed/seed_{s}/"
                 "fMRI_all_significant_features_seed{s}_60s_75overlap.csv")
TABLE5_CSV = "results/tables/table5_smri_posthoc.csv"
TABLE6_CSV = "results/tables/table6_clinical_posthoc.csv"
SMRI_UNITS = {"Sulcal Depth": " (mm)", "Cortical Thickness": " (mm)", "Gray Matter Volume": " (ml)"}
SMRI_FILES = {
    "Sulcal Depth": "data/MRI_data/ROI_aparc_HCP_MMP1_depth.csv_with_groups.csv",
    "Cortical Thickness": "data/MRI_data/ROI_aparc_HCP_MMP1_thickness.csv_with_groups.csv",
    "Gyrification": "data/MRI_data/ROI_aparc_HCP_MMP1_gyrification.csv_with_groups.csv",
    "Gray Matter Volume": "data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv",
    "Fractal Dimension": "data/MRI_data/ROI_aparc_HCP_MMP1_fractaldimension.csv_with_groups.csv",
}

# Group colours (Dark2; validated for colour-vision deficiency) and ink
COLORS = {"N": "#1B9E77", "A+P-": "#D95F02", "A+P+": "#7570B3"}
INK, MUTED = "#262626", "#6b6b6b"
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 8,
    "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})
RNG = np.random.default_rng(0)


def reset_jitter():
    """Each figure gets its own jitter, so that adding or removing a figure changes no other."""
    global RNG
    RNG = np.random.default_rng(0)
stats_rows = []


def p_marker(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "†" if p < 0.10 else ""


def group_panel(ax, by_group, contrasts, ylabel=None, title=None):
    """Points + box per group; brackets for the given contrasts [(g1, g2, p), ...]."""
    for i, g in enumerate(GROUPS):
        v = by_group[g]
        ax.boxplot(v, positions=[i], widths=0.42, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=COLORS[g] + "2e", edgecolor=COLORS[g], linewidth=1),
                   whiskerprops=dict(color=COLORS[g], linewidth=1), capprops=dict(color=COLORS[g], linewidth=1),
                   medianprops=dict(color=INK, linewidth=1.4))
        ax.scatter(i + RNG.uniform(-0.15, 0.15, len(v)), v, s=11, color=COLORS[g],
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xticks(range(len(GROUPS)), [g.replace("A+P-", "A+P\u2212") for g in GROUPS])  # true minus sign
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    lo = min(v.min() for v in by_group.values())
    hi = max(v.max() for v in by_group.values())
    span = hi - lo if hi > lo else 1.0
    level = 0
    for g1, g2, p in sorted(contrasts, key=lambda c: abs(GROUPS.index(c[1]) - GROUPS.index(c[0]))):
        marker = p_marker(p)
        if not marker:
            continue
        y = hi + span * (0.08 + 0.13 * level)
        x1, x2 = GROUPS.index(g1), GROUPS.index(g2)
        ax.plot([x1, x1, x2, x2], [y - span * 0.02, y, y, y - span * 0.02], color=INK, linewidth=0.8)
        ax.text((x1 + x2) / 2, y, marker, ha="center", va="bottom", color=INK, fontsize=8)
        level += 1
    ax.set_ylim(lo - span * 0.06, hi + span * (0.12 + 0.13 * max(level, 1)))
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left", color=INK)


def test_and_record(figure, panel, by_group):
    rows = posthoc(by_group)
    p_kw = kruskal_p(by_group)
    for r in rows:
        stats_rows.append({"figure": figure, "panel": panel, "p_kruskal": p_kw,
                           **{f"n_{g}": len(by_group[g]) for g in GROUPS}, **r})
    return [(r["group_1"], r["group_2"], r["p_bonf"]) for r in rows]


OUT_DIR.mkdir(parents=True, exist_ok=True)
sample = pd.read_csv(SAMPLE_CSV)[["participant_id", "Risk_Group"]]

# EEG: subject mean within-state HFD variability by state direction
reset_jitter()
eeg_panels = {}
for panel, (path, suffix) in EEG_FILES.items():
    d = pd.read_csv(path)
    if d["mean_variability"].notna().any():
        eeg_panels[panel] = (sample.merge(d[["participant_id", "mean_variability"]], on="participant_id"), suffix)
for panel, (eeg, suffix) in eeg_panels.items():
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    by_group = split_by_group(eeg["mean_variability"], eeg["Risk_Group"])
    contrasts = test_and_record("eeg_hfd_variability", panel, by_group)
    group_panel(ax, by_group, contrasts, ylabel="Within-state HFD variability")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_eeg_hfd_variability_{suffix}.png")
    plt.close(fig)

# fMRI: subject mean dwell time over the significant states of all seeds
reset_jitter()
state_col = re.compile(r"^fMRI_K\d+_state\d+_(N|AposPminus|AposPpos)$")
dwell = {"N": [], "carrier": []}
for s in FMRI_SEEDS:
    seed_df = sample[["participant_id"]].merge(pd.read_csv(FMRI_SEED_CSV.format(s=s)), on="participant_id")
    assert len(seed_df) == len(sample)
    for col in seed_df.columns:
        m = state_col.match(col)
        if m:
            dwell["N" if m.group(1) == "N" else "carrier"].append(seed_df[col].to_numpy(float))
fmri = sample.copy()
fmri["dwell_N_states"] = np.mean(dwell["N"], axis=0)
fmri["dwell_carrier_states"] = np.mean(dwell["carrier"], axis=0)
fmri.to_csv(OUT_DIR / "fmri_subject_mean_dwell_time.csv", index=False)

for key, label in [("N", "N-dominant states"), ("carrier", "Carrier-dominant states")]:
    fig, ax = plt.subplots(figsize=PANEL_SIZE)
    by_group = split_by_group(fmri[f"dwell_{key}_states"], fmri["Risk_Group"])
    contrasts = test_and_record("fmri_dwell_time", label, by_group)
    group_panel(ax, by_group, contrasts, ylabel="Dwell time (s)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_fmri_dwell_time_{key}.png")
    plt.close(fig)

# sMRI: Table S4 regions
reset_jitter()
table5 = pd.read_csv(TABLE5_CSV)
regions = list(dict.fromkeys(zip(table5["measure"], table5["region"])))
measures = list(dict.fromkeys(table5["measure"]))
ncol = max(sum(m == measure for m, _ in regions) for measure in measures)
fig, axes = plt.subplots(len(measures), ncol, figsize=(7.0, 1.75 * len(measures)), squeeze=False)
for r, measure in enumerate(measures):
    roi = pd.read_csv(SMRI_FILES[measure])
    roi["participant_id"] = roi["names"].astype(str).str.replace("_T1w", "", regex=False)
    data = sample.merge(roi, on="participant_id")
    measure_regions = [reg for m, reg in regions if m == measure]
    for c in range(ncol):
        ax = axes[r, c]
        if c >= len(measure_regions):
            ax.axis("off")
            continue
        region = measure_regions[c]
        rows = table5[(table5["measure"] == measure) & (table5["region"] == region)]
        contrasts = list(zip(rows["group_1"], rows["group_2"], rows["p_bonf"]))
        by_group = split_by_group(data[region], data["Risk_Group"])
        group_panel(ax, by_group, contrasts, ylabel=measure + SMRI_UNITS.get(measure, "") if c == 0 else None, title=region)
        ax.tick_params(axis="x", labelsize=6.5)
fig.tight_layout(h_pad=1.2)
fig.savefig(OUT_DIR / "fig_smri_regions.png")
plt.close(fig)

# Non-imaging: Table 3 variables
reset_jitter()
table6 = pd.read_csv(TABLE6_CSV)
clinical = pd.read_csv(SAMPLE_CSV)
variables = list(dict.fromkeys(table6["variable"]))
fig, axes = plt.subplots(1, len(variables), figsize=(7.0, 2.3))
for ax, var in zip(axes, variables):
    rows = table6[table6["variable"] == var]
    contrasts = list(zip(rows["group_1"], rows["group_2"], rows["p_bonf"]))
    by_group = split_by_group(clinical[var], clinical["Risk_Group"])
    group_panel(ax, by_group, contrasts, title=var.replace("_", " ").replace("eosinophils", "Eosinophils").replace("MINI-COPE", "Mini-COPE"))
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_nonimaging.png")
plt.close(fig)

pd.DataFrame(stats_rows).to_csv(OUT_DIR / "figure_stats.csv", index=False)
print(f"Wrote figures and figure_stats.csv to {OUT_DIR}/")
