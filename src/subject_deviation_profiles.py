"""Subject-level deviation from the non-carrier (N) profile across EEG, fMRI, sMRI and clinical features.

Exploratory and descriptive. The features were chosen because the groups differ on them,
using all 64 subjects, so carriers deviate from N partly by construction. The output shows
which carriers carry the reported group differences and whether the same carriers deviate in
several modalities; it is not evidence for a group difference.

Run from the repository root after the EEG, fMRI and table scripts:
    python src/subject_deviation_profiles.py
Writes CSVs to results/subject_profiles/.

Features (the paper's reported findings):
  EEG       mean HFD variability in N-higher clusters and in A+P+-higher clusters (Fig 6)
  fMRI      dwell time in the N-dominant state, seed 9, K = 7, state 0 (Fig 7)
  sMRI      Table 5 regions with an N vs <group> contrast (so they differ by group)
  Clinical  Table 6 variables

Scoring, separately for each carrier group's profile (A+P- or A+P+):
  1. Each feature value is placed among the 25 N subjects as a mid-rank percentile, oriented
     so that 1 means further in the direction the carrier group's mean differs from N's.
     Percentiles rather than z-scores because fMRI dwell time is zero-inflated.
  2. A modality score is the mean oriented percentile over its features, placed again among
     the N subjects' modality scores.
  3. A modality counts as deviant when that score is >= DEVIANT_PCT (more extreme than 80%
     of N subjects).
  N subjects are scored the same way against the other 24 N subjects (leave-one-out), giving
  the base rate. A within-group permutation of each modality's flags tests whether deviations
  co-occur in the same carriers more than if they were independent across modalities.
"""
import os

import numpy as np
import pandas as pd

SAMPLE_CSV = "data/participants_n64.csv"
EEG_FILES = {
    "EEG HFD var. (N-higher clusters)": "results/eeg/eeg_subject_mean_variability_N_higher_multiseed.csv",
    "EEG HFD var. (A+P+-higher clusters)": "results/eeg/eeg_subject_mean_variability_A_plus_P_plus_higher_multiseed.csv",
}
FMRI_CSV = ("results/fmri/fMRI_significant_clusters_60s_75overlap_multiseed/seed_9/"
            "fMRI_all_significant_features_seed9_60s_75overlap.csv")
FMRI_COL = "fMRI_K7_state0_N"
SMRI_FILES = {
    "Sulcal Depth": "data/MRI_data/ROI_aparc_HCP_MMP1_depth.csv_with_groups.csv",
    "Cortical Thickness": "data/MRI_data/ROI_aparc_HCP_MMP1_thickness.csv_with_groups.csv",
    "Gyrification": "data/MRI_data/ROI_aparc_HCP_MMP1_gyrification.csv_with_groups.csv",
    "Gray Matter Volume": "data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv",
    "Fractal Dimension": "data/MRI_data/ROI_aparc_HCP_MMP1_fractaldimension.csv_with_groups.csv",
}
TABLE5_CSV = "results/tables/table5_smri_posthoc.csv"
TABLE6_CSV = "results/tables/table6_clinical_posthoc.csv"
OUT_DIR = "results/subject_profiles"

CARRIER_GROUPS = ["A+P-", "A+P+"]
MODALITIES = ["EEG", "fMRI", "sMRI", "Clinical"]
DEVIANT_PCT = 0.80
MIN_DEVIANT_MODALITIES = 3
N_PERMUTATIONS = 10000
SEED = 9


def midrank_pct(x, ref):
    ref = ref[~np.isnan(ref)]
    if np.isnan(x) or len(ref) == 0:
        return np.nan
    return ((ref < x).sum() + 0.5 * (ref == x).sum()) / len(ref)


def pct_against_n(values, is_n):
    """Percentile of each subject among N subjects; N subjects against the other N (leave-one-out)."""
    values = np.asarray(values, float)
    n_idx = np.flatnonzero(is_n)
    out = np.empty(len(values))
    for i, x in enumerate(values):
        ref_idx = n_idx[n_idx != i]
        out[i] = midrank_pct(x, values[ref_idx])
    return out


def hedges_g(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))


# Load features: one row per subject, columns as (modality, feature)
sample = pd.read_csv(SAMPLE_CSV)
data = sample[["participant_id", "Risk_Group"]].copy()
feature_modality = {}

for name, path in EEG_FILES.items():
    eeg = pd.read_csv(path)[["participant_id", "mean_variability"]].rename(columns={"mean_variability": name})
    data = data.merge(eeg, on="participant_id", how="left")
    feature_modality[name] = "EEG"

fmri = pd.read_csv(FMRI_CSV)[["participant_id", FMRI_COL]].rename(columns={FMRI_COL: "fMRI dwell time (N-dominant state)"})
data = data.merge(fmri, on="participant_id", how="left")
feature_modality["fMRI dwell time (N-dominant state)"] = "fMRI"

table5 = pd.read_csv(TABLE5_CSV)
smri_by_group = {}
for g in CARRIER_GROUPS:
    rows = table5[(table5["group_1"] == "N") & (table5["group_2"] == g)]
    smri_by_group[g] = [f"{m}: {r}" for m, r in zip(rows["measure"], rows["region"])]
for measure, path in SMRI_FILES.items():
    regions = {f.split(": ")[1] for g in CARRIER_GROUPS for f in smri_by_group[g] if f.startswith(measure + ":")}
    if not regions:
        continue
    roi = pd.read_csv(path)
    roi["participant_id"] = roi["names"].astype(str).str.replace("_T1w", "", regex=False)
    roi = roi[["participant_id", *sorted(regions)]].rename(columns={r: f"{measure}: {r}" for r in regions})
    data = data.merge(roi, on="participant_id", how="left")
    feature_modality.update({f"{measure}: {r}": "sMRI" for r in regions})

clinical_vars = list(dict.fromkeys(pd.read_csv(TABLE6_CSV)["variable"]))
for var in clinical_vars:
    data[var] = pd.to_numeric(sample[var], errors="coerce")
    feature_modality[var] = "Clinical"

is_n = (data["Risk_Group"] == "N").to_numpy()
os.makedirs(OUT_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
feature_rows, subject_rows, summary_rows = [], [], []

for g in CARRIER_GROUPS:
    features = [f for f, m in feature_modality.items() if m != "sMRI" or f in smri_by_group[g]]
    in_scope = is_n | (data["Risk_Group"] == g).to_numpy()
    sub = data[in_scope].reset_index(drop=True)
    sub_is_n = (sub["Risk_Group"] == "N").to_numpy()

    oriented = {}
    for f in features:
        x = sub[f].to_numpy(float)
        a, b = x[~sub_is_n], x[sub_is_n]
        higher = np.nanmean(a) > np.nanmean(b)
        p = pct_against_n(x, sub_is_n)
        oriented[f] = p if higher else 1 - p
        feature_rows.append({"profile": g, "modality": feature_modality[f], "feature": f,
                             "carrier_direction": "higher" if higher else "lower",
                             "hedges_g_vs_N": hedges_g(a, b)})

    flags = pd.DataFrame(index=sub.index)
    for m in MODALITIES:
        mf = [f for f in features if feature_modality[f] == m]
        composite = np.nanmean(np.column_stack([oriented[f] for f in mf]), axis=1)
        flags[m] = pct_against_n(composite, sub_is_n)
        top = np.column_stack([oriented[f] for f in mf])
        top_idx = np.nanargmax(np.nan_to_num(top, nan=-1), axis=1)
        flags[f"{m}_top_feature"] = [mf[j] for j in top_idx]
        flags[f"{m}_top_feature_pct"] = top[np.arange(len(sub)), top_idx]
    deviant = (flags[MODALITIES] >= DEVIANT_PCT)
    flags["n_deviant_modalities"] = deviant.sum(axis=1)
    flags["deviant_modalities"] = deviant.apply(lambda r: ", ".join(m for m in MODALITIES if r[m]), axis=1)
    out = pd.concat([sub[["participant_id", "Risk_Group"]], flags], axis=1)
    out.insert(0, "profile", g)
    subject_rows.append(out)

    # Do deviations co-occur in the same carriers? Shuffle each modality's flags independently.
    carrier_dev = deviant[~sub_is_n].to_numpy()
    observed = int((carrier_dev.sum(axis=1) >= MIN_DEVIANT_MODALITIES).sum())
    null = np.empty(N_PERMUTATIONS)
    for k in range(N_PERMUTATIONS):
        shuffled = np.column_stack([rng.permutation(col) for col in carrier_dev.T])
        null[k] = (shuffled.sum(axis=1) >= MIN_DEVIANT_MODALITIES).sum()
    n_dev = deviant[sub_is_n].sum(axis=1)
    summary_rows.append({
        "profile": g, "n_carriers": int((~sub_is_n).sum()),
        **{f"carriers_deviant_{m}": int(deviant[~sub_is_n][m].sum()) for m in MODALITIES},
        **{f"N_deviant_{m}": int(deviant[sub_is_n][m].sum()) for m in MODALITIES},
        f"carriers_with_>={MIN_DEVIANT_MODALITIES}": observed,
        f"N_with_>={MIN_DEVIANT_MODALITIES}": int((n_dev >= MIN_DEVIANT_MODALITIES).sum()),
        f"expected_carriers_with_>={MIN_DEVIANT_MODALITIES}_if_independent": null.mean(),
        "p_cooccurrence": (np.sum(null >= observed) + 1) / (N_PERMUTATIONS + 1),
    })

pd.DataFrame(feature_rows).round(3).to_csv(f"{OUT_DIR}/features_used.csv", index=False)
pd.concat(subject_rows).round(3).to_csv(f"{OUT_DIR}/subject_deviation_profiles.csv", index=False)
pd.DataFrame(summary_rows).round(4).to_csv(f"{OUT_DIR}/deviation_summary.csv", index=False)
print(f"Wrote {OUT_DIR}/: features_used.csv, subject_deviation_profiles.csv, deviation_summary.csv")
