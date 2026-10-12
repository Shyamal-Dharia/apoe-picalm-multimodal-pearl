"""Head-motion quality control for the resting-state fMRI of the analysis sample (n = 64).

Reads fMRIPrep's framewise displacement (FD, Power et al. 2012, as computed by fMRIPrep) from
$FMRIPREP_DIR/sub-XX/func/sub-XX_task-rest_dir-{AP,PA}_desc-confounds_timeseries.tsv and
summarizes it per participant over both runs. The first volume has no FD and is skipped. Also
counts the initial volumes fMRIPrep flagged as non-steady state (non_steady_state_outlier_XX
columns) per run. No volume or participant is removed.

Run from the repository root:
    FMRIPREP_DIR=/path/to/derivatives python src/fmri_motion_qc.py
Writes results/tables/fmri_motion_qc.csv (one row per participant) and
results/tables/fmri_motion_qc_summary.csv (mean (SD) by group and Kruskal-Wallis p).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from group_stats import GROUPS, kruskal_p, split_by_group  # noqa: E402

FMRIPREP_DIR = os.environ.get("FMRIPREP_DIR")
if not FMRIPREP_DIR:
    raise SystemExit("Set FMRIPREP_DIR to the fMRIPrep derivatives folder")
SAMPLE_CSV = "data/participants_n64.csv"
OUT_DIR = "results/tables"
FD_THRESHOLD = 0.5  # mm

rows = []
for pid, group in pd.read_csv(SAMPLE_CSV)[["participant_id", "Risk_Group"]].itertuples(index=False):
    fd, nonsteady = [], []
    for run in ("AP", "PA"):
        path = f"{FMRIPREP_DIR}/{pid}/func/{pid}_task-rest_dir-{run}_desc-confounds_timeseries.tsv"
        confounds = pd.read_csv(path, sep="\t")
        fd.append(confounds["framewise_displacement"].to_numpy()[1:])
        nonsteady.append(sum(c.startswith("non_steady_state_outlier") for c in confounds.columns))
    fd = np.concatenate(fd)
    rows.append({"participant_id": pid, "Risk_Group": group, "mean_fd_mm": fd.mean(), "max_fd_mm": fd.max(),
                 "pct_volumes_fd_above_0p5": 100 * np.mean(fd > FD_THRESHOLD), "n_volumes": len(fd),
                 "nonsteady_AP": nonsteady[0], "nonsteady_PA": nonsteady[1]})
qc = pd.DataFrame(rows)
os.makedirs(OUT_DIR, exist_ok=True)
qc.to_csv(f"{OUT_DIR}/fmri_motion_qc.csv", index=False)

summary = []
for measure in ("mean_fd_mm", "max_fd_mm", "pct_volumes_fd_above_0p5"):
    by_group = split_by_group(qc[measure], qc["Risk_Group"])
    row = {"measure": measure, "p_kruskal": kruskal_p(by_group)}
    for g in GROUPS:
        row[g] = f"{qc.loc[qc['Risk_Group'] == g, measure].mean():.3f} ({qc.loc[qc['Risk_Group'] == g, measure].std():.3f})"
    row["Overall"] = f"{qc[measure].mean():.3f} ({qc[measure].std():.3f})"
    summary.append(row)
heavy = qc["pct_volumes_fd_above_0p5"] > 20
summary.append({"measure": "n_participants_pct_fd_above_0p5_over_20", "p_kruskal": np.nan,
                **{g: str(int(heavy[qc["Risk_Group"] == g].sum())) for g in GROUPS}, "Overall": str(int(heavy.sum()))})
summary.append({"measure": "max_pct_volumes_fd_above_0p5", "p_kruskal": np.nan,
                **{g: f"{qc.loc[qc['Risk_Group'] == g, 'pct_volumes_fd_above_0p5'].max():.1f}" for g in GROUPS},
                "Overall": f"{qc['pct_volumes_fd_above_0p5'].max():.1f}"})
ns = qc[["nonsteady_AP", "nonsteady_PA"]].to_numpy()
summary.append({"measure": "nonsteady_volumes_per_run_range", "p_kruskal": np.nan,
                **{g: "" for g in GROUPS}, "Overall": f"{ns.min()}-{ns.max()}"})
pd.DataFrame(summary).to_csv(f"{OUT_DIR}/fmri_motion_qc_summary.csv", index=False)
print(pd.DataFrame(summary).to_string(index=False))
