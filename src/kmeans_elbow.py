"""Elbow curves for the k-means states of the dynamic EEG and fMRI features (supplementary figure).

Reproduces src/k-means_plot.ipynb: k-means (seed 1, n_init="auto", max_iter=10000) for
k = 2-14 on all windows, inertia scaled to 0-1, and the knee found with kneed (convex,
decreasing). The analysis uses k = 5-10 for EEG and k = 4-9 for fMRI (K_STATES_LIST_EEG and
K_STATES_LIST_FMRI in src/f1_permutation_pipeline.py), set from the knees of the original
curves (EEG 5, fMRI 4) and the next five values. The shaded band shows the k actually used.
  EEG   HFD window vectors (127 channels x 5 bands) of every participant with EEG features
        and a genotype group in the sMRI table (data/EEG_data/HFD_PSD_4sec_75overlap_6min).
  fMRI  sliding-window connectivity vectors (4950 values), both runs of every participant in
        data/fMRI_data/subject_windows_confounds_old_60sec.

Run from the repository root:
    python src/kmeans_elbow.py
Writes results/figures/fig_kmeans_elbow_eeg.png, results/figures/fig_kmeans_elbow_fmri.png and
results/figures/kmeans_elbow_inertia.csv.
"""
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans

EEG_DIR = "data/EEG_data/HFD_PSD_4sec_75overlap_6min"
SMRI_GROUPS = "data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv"
FMRI_WINDOWS = "data/fMRI_data/subject_windows_confounds_old_60sec/*_windows.npz"
K_RANGE = list(range(2, 15))
K_USED = {"EEG": (5, 10), "fMRI": (4, 9)}  # as in src/f1_permutation_pipeline.py
OUT_DIR = Path("results/figures")


def subject_id(name):
    return "sub-%02d" % int(re.search(r"(\d+)", os.path.basename(str(name))).group(1))


def scaled_inertia(X):
    inertia = np.array([KMeans(n_clusters=k, random_state=1, n_init="auto", max_iter=10000).fit(X).inertia_
                        for k in K_RANGE])
    scaled = (inertia - inertia.min()) / (inertia.max() - inertia.min())
    return scaled, KneeLocator(K_RANGE, scaled, curve="convex", direction="decreasing").knee


grouped = {subject_id(n) for n in pd.read_csv(SMRI_GROUPS)["names"]}
eeg_files = [f for f in sorted(glob.glob(f"{EEG_DIR}/*.npz")) if subject_id(f) in grouped]
X_eeg = np.concatenate([np.load(f, allow_pickle=True)["HFD_features"].reshape(357, -1) for f in eeg_files])
X_fmri = np.concatenate([np.concatenate([z["AP"], z["PA"]]) for z in map(np.load, sorted(glob.glob(FMRI_WINDOWS)))])

curves = {"EEG": scaled_inertia(X_eeg), "fMRI": scaled_inertia(X_fmri)}
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"k": K_RANGE, **{f"{m}_inertia_scaled": c[0] for m, c in curves.items()}}).to_csv(
    OUT_DIR / "kmeans_elbow_inertia.csv", index=False)

# One single-panel figure per modality; the panel labels (a, b) come from LaTeX subfigures.
for modality, (scaled, knee) in curves.items():
    lo, hi = K_USED[modality]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.axvspan(lo - 0.5, hi + 0.5, color="#d9d9d9", alpha=0.6, lw=0, zorder=0, label=f"k used ({lo}-{hi})")
    ax.plot(K_RANGE, scaled, "o-", color="#404040", lw=1.5, ms=4, zorder=2, label="Scaled inertia")
    ax.plot([knee], [scaled[K_RANGE.index(knee)]], "x", color="#b2182b", ms=9, mew=2, zorder=3, label=f"Knee (k = {knee})")
    ax.set_xlabel("Number of states (k)")
    ax.set_ylabel("Scaled inertia")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(K_RANGE[::2])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"fig_kmeans_elbow_{modality.lower()}.png", dpi=300)
    plt.close(fig)
print(f"EEG: {len(eeg_files)} participants, {X_eeg.shape[0]} windows, knee k = {curves['EEG'][1]}")
print(f"fMRI: {X_fmri.shape[0]} windows, knee k = {curves['fMRI'][1]}")
