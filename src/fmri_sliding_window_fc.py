"""Sliding-window functional connectivity (60 s windows) from fMRIPrep resting-state BOLD.

Rebuilds data/fMRI_data/subject_windows_confounds_old_60sec/sub-XX_windows.npz, the fMRI
input of the paper. The original code was lost. This script reproduces the 68 published
files to float32 precision (largest absolute difference below 1e-5).

Pipeline, for each subject and each run (dir-AP and dir-PA, 560 volumes, TR = 0.8 s):

1. Input: fMRIPrep 25.0.0 outputs in $FMRIPREP_DIR/sub-XX/func/
     sub-XX_task-rest_dir-{AP,PA}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
     sub-XX_task-rest_dir-{AP,PA}_desc-confounds_timeseries.tsv
2. Parcellation: Schaefer 2018, 100 parcels, 7 networks, in the TemplateFlow version
   registered to MNI152NLin2009cAsym at 2 mm
   (tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-100Parcels7Networks_dseg.nii.gz).
   Each parcel signal is the mean over its voxels (nilearn NiftiLabelsMasker). No smoothing.
3. Denoising, done by nilearn.signal.clean inside the masker, in this order:
     a. linear detrending of the parcel signals and of the confounds;
     b. Butterworth band-pass filter, 0.01-0.1 Hz, applied to the signals and to the confounds;
     c. z-scoring of the confounds, then regression of 7 confounds out of the signals:
        trans_x, trans_y, trans_z, rot_x, rot_y, rot_z (fMRIPrep head motion) and
        a_comp_cor_00 (the first aCompCor component of fMRIPrep's combined WM+CSF mask);
     d. z-scoring of each parcel time series over the whole run.
4. Sliding windows: 75 volumes (60 s), step 19 volumes (15.2 s, 74.7% overlap), so
   1 + (560 - 75) // 19 = 26 windows per run.
5. Connectivity in each window: nilearn ConnectivityMeasure(kind="correlation"). Its default
   covariance estimator is Ledoit-Wolf shrinkage, so the values are shrinkage-regularised
   correlations, slightly smaller in magnitude than plain Pearson r. No Fisher z-transform.
6. Each 100 x 100 matrix is stored as its strict lower triangle, row by row (nilearn
   sym_matrix_to_vec with discard_diagonal=True): 4950 float32 values per window.

Group label (int): 0 = N, 1 = A+P-, 2 = A+P+, from the PEARL-Neuro participants table
(data/participants.csv, tab-separated):
    N     no APOE-e4, PICALM rs3851179 not G/G
    A+P-  APOE-e4 carrier, PICALM rs3851179 not G/G
    A+P+  APOE-e4 carrier, PICALM rs3851179 G/G
Any APOE-e2 carrier is excluded, including e2/e4. Of the 69 subjects with fMRIPrep
outputs, this removes only sub-46 (e2/e4), which leaves the 68 published subjects.

Output: one sub-XX_windows.npz per subject (np.savez_compressed) with the keys
    label  int64 scalar
    PA     (26, 4950) float32
    AP     (26, 4950) float32

Environment variables:
    FMRIPREP_DIR      fMRIPrep derivatives root, the folder that holds sub-XX/ (required)
    FMRI_WINDOWS_OUT  output folder (default: data/fMRI_data/subject_windows_confounds_old_60sec)

Run from anywhere. Repository paths are resolved from this file's location:
    FMRIPREP_DIR=/path/to/derivatives python src/fmri_sliding_window_fc.py

Checked with nilearn 0.11.1, scikit-learn 1.6.0, numpy 2.0.2, scipy 1.13.1, pandas 2.2.2.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiLabelsMasker

# ─── Constants ────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
PARTICIPANTS_TSV = REPO_ROOT / "data" / "participants.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "fMRI_data" / "subject_windows_confounds_old_60sec"

RUNS = ["PA", "AP"]
BOLD_PATTERN = "{sid}_task-rest_dir-{run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
CONFOUNDS_PATTERN = "{sid}_task-rest_dir-{run}_desc-confounds_timeseries.tsv"

TR = 0.8               # seconds per volume
HIGH_PASS = 0.01       # Hz
LOW_PASS = 0.1         # Hz
WIN_LEN = 75           # volumes (60 s)
STEP = 19              # volumes (15.2 s), gives 26 windows from 560 volumes
CONFOUND_COLUMNS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z", "a_comp_cor_00"]

GROUP_CODE = {"N": 0, "A+P-": 1, "A+P+": 2}
N_JOBS = 8             # subjects processed in parallel


def schaefer_atlas_path():
    """Schaefer-100 7-network atlas in MNI152NLin2009cAsym, 2 mm, from TemplateFlow."""
    from templateflow import api as tflow

    return str(tflow.get(
        "MNI152NLin2009cAsym", resolution=2, atlas="Schaefer2018",
        desc="100Parcels7Networks", suffix="dseg", extension=".nii.gz",
    ))


def group_labels():
    """Map participant_id -> group code (0, 1, 2). Excluded participants are left out."""
    df = pd.read_csv(PARTICIPANTS_TSV, sep="\t", dtype=str)
    apoe = df["APOE_haplotype"].str.strip()
    picalm = df["PICALM_rs3851179"].str.strip()

    e4 = apoe.str.contains("e4", na=False)
    e2 = apoe.str.contains("e2", na=False)
    picalm_gg = picalm == "G/G"

    group = pd.Series("Excluded", index=df.index)
    group[~e4 & ~picalm_gg] = "N"
    group[e4 & ~picalm_gg] = "A+P-"
    group[e4 & picalm_gg] = "A+P+"
    group[e2] = "Excluded"

    keep = group != "Excluded"
    return dict(zip(df.loc[keep, "participant_id"], group[keep].map(GROUP_CODE)))


def sliding_window_fc(ts):
    """(n_volumes, n_parcels) time series -> (n_windows, n_edges) connectivity vectors."""
    conn = ConnectivityMeasure(kind="correlation", vectorize=True, discard_diagonal=True)
    n_windows = 1 + (ts.shape[0] - WIN_LEN) // STEP
    windows = [ts[w * STEP: w * STEP + WIN_LEN] for w in range(n_windows)]
    return np.vstack([conn.fit_transform([win])[0] for win in windows])


def process_subject(fmriprep_dir, sid, label, atlas, out_dir):
    func_dir = fmriprep_dir / sid / "func"
    masker = NiftiLabelsMasker(
        labels_img=atlas,
        standardize=True,
        detrend=True,
        high_pass=HIGH_PASS,
        low_pass=LOW_PASS,
        t_r=TR,
    )

    result = {"label": label}
    for run in RUNS:
        bold = func_dir / BOLD_PATTERN.format(sid=sid, run=run)
        confounds = pd.read_csv(func_dir / CONFOUNDS_PATTERN.format(sid=sid, run=run), sep="\t")
        ts = masker.fit_transform(str(bold), confounds=confounds[CONFOUND_COLUMNS].values)
        result[run] = sliding_window_fc(ts)

    np.savez_compressed(out_dir / f"{sid}_windows.npz", **result)
    return sid, label, result["PA"].shape, result["AP"].shape


def main():
    if not os.environ.get("FMRIPREP_DIR"):
        sys.exit("FMRIPREP_DIR is not set. Point it to the fMRIPrep derivatives folder "
                 "that contains sub-XX/func/, e.g.\n"
                 "  FMRIPREP_DIR=/path/to/derivatives python src/fmri_sliding_window_fc.py")
    fmriprep_dir = Path(os.environ["FMRIPREP_DIR"])
    if not fmriprep_dir.is_dir():
        sys.exit(f"FMRIPREP_DIR does not exist: {fmriprep_dir}")

    out_dir = Path(os.environ.get("FMRI_WINDOWS_OUT", DEFAULT_OUT_DIR))
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = group_labels()
    subjects = []
    for sub_dir in sorted(fmriprep_dir.glob("sub-*")):
        sid = sub_dir.name
        has_runs = all((sub_dir / "func" / BOLD_PATTERN.format(sid=sid, run=run)).exists()
                       for run in RUNS)
        if not sub_dir.is_dir() or not has_runs:
            continue
        if sid not in labels:
            print(f"Skipping {sid}: not in a genotype group (or APOE-e2 carrier)")
            continue
        subjects.append(sid)

    print(f"{len(subjects)} subjects -> {out_dir}")
    atlas = schaefer_atlas_path()
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_subject)(fmriprep_dir, sid, labels[sid], atlas, out_dir)
        for sid in subjects
    )
    for sid, label, pa_shape, ap_shape in results:
        print(f"{sid}  label={label}  PA={pa_shape}  AP={ap_shape}")


if __name__ == "__main__":
    main()
