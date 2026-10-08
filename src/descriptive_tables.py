"""Group-comparison tables: demographics (Table 1), sMRI post hoc (Table 5), clinical post hoc (Table 6).

Run from the repository root after src/select_participants.py:
    python src/descriptive_tables.py
Writes CSVs to results/tables/.

Post hoc comparisons follow JASP's one-way ANOVA post hoc output: mean difference
(group 1 - group 2), standard error from the pooled ANOVA error term, Bonferroni-adjusted
p-values over the three pairwise contrasts, and Tukey-adjusted 95% confidence intervals.

Feature selection (no correction across features):
  sMRI      the 5 features per measure with the lowest Kruskal-Wallis p; contrasts with
            Bonferroni p < .07 are reported.
  Clinical  variables with Kruskal-Wallis p < .05 (absolute blood cell counts; the
            percentage variants are not tested); all three contrasts are reported.
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.libqsturng import qsturng

SAMPLE_CSV = "data/participants_n64.csv"
OUT_DIR = "results/tables"
GROUPS = ["N", "A+P-", "A+P+"]

SMRI_FILES = {
    "Sulcal Depth": "data/MRI_data/ROI_aparc_HCP_MMP1_depth.csv_with_groups.csv",
    "Cortical Thickness": "data/MRI_data/ROI_aparc_HCP_MMP1_thickness.csv_with_groups.csv",
    "Gyrification": "data/MRI_data/ROI_aparc_HCP_MMP1_gyrification.csv_with_groups.csv",
    "Gray Matter Volume": "data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv",
    "Fractal Dimension": "data/MRI_data/ROI_aparc_HCP_MMP1_fractaldimension.csv_with_groups.csv",
}
SMRI_TOP_K = 5
SMRI_REPORT_P = 0.07
CLINICAL_SELECT_P = 0.05

CLINICAL_VARS = (
    ["age", "sex", "education", "learning_deficits", "BMI", "allergies", "drugs",
     "ibuprofen_intake", "thyroid_diseases", "hypertension", "diabetes", "other_diseases",
     "smoking_status", "coffee_status", "dementia_history_parents", "HSV_r",
     "BDI", "SES", "RPM", "EHI", "NEO_NEU", "NEO_EXT", "NEO_OPE", "NEO_AGR", "NEO_CON", "AUDIT"]
    + [f"MINI-COPE_{i}" for i in range(1, 15)]
    + [f"CVLT_{i}" for i in range(1, 14)]
    + ["leukocytes", "erythrocytes", "hemoglobin", "hematocrit", "MCV", "MCH", "MCHC", "RDW-CV",
       "platelets", "PDW", "MPV", "P-LCR", "neutrophils", "lymphocytes", "monocytes",
       "eosinophils", "basophils", "total_cholesterol", "cholesterol_HDL",
       "non-HDL_cholesterol", "LDL_cholesterol", "triglycerides"]
)


def split_by_group(values, groups):
    values = pd.to_numeric(values, errors="coerce")
    return {g: values[groups == g].dropna().to_numpy(float) for g in GROUPS}


def kruskal_p(by_group):
    samples = list(by_group.values())
    if min(len(s) for s in samples) < 2 or np.var(np.concatenate(samples)) == 0:
        return np.nan
    return stats.kruskal(*samples).pvalue


def posthoc(by_group):
    """JASP-style ANOVA post hoc: Bonferroni p, Tukey 95% CI, pooled SE."""
    df_within = sum(len(v) for v in by_group.values()) - len(GROUPS)
    mse = sum(((v - v.mean()) ** 2).sum() for v in by_group.values()) / df_within
    t_tukey = qsturng(0.95, len(GROUPS), df_within) / np.sqrt(2)
    n_contrasts = len(GROUPS) * (len(GROUPS) - 1) // 2
    rows = []
    for g1, g2 in itertools.combinations(GROUPS, 2):
        a, b = by_group[g1], by_group[g2]
        diff = a.mean() - b.mean()
        se = np.sqrt(mse * (1 / len(a) + 1 / len(b)))
        p_bonf = min(1.0, n_contrasts * 2 * stats.t.sf(abs(diff / se), df_within))
        rows.append({"group_1": g1, "group_2": g2, "diff": diff,
                     "ci_lower": diff - t_tukey * se, "ci_upper": diff + t_tukey * se,
                     "se": se, "p_bonf": p_bonf})
    return rows


os.makedirs(OUT_DIR, exist_ok=True)
sample = pd.read_csv(SAMPLE_CSV)
groups = sample["Risk_Group"]

# Table 1: demographics (sex coded 0 = male, 1 = female)
table1 = []
for name, part in [*((g, sample[groups == g]) for g in GROUPS), ("Overall", sample)]:
    table1.append({"group": name, "n": len(part), "male": int((part["sex"] == 0).sum()),
                   "female": int((part["sex"] == 1).sum()),
                   "age_mean": part["age"].mean(), "age_sd": part["age"].std()})
pd.DataFrame(table1).round(2).to_csv(f"{OUT_DIR}/table1_demographics.csv", index=False)

# Table 5: sMRI post hoc
table5 = []
for measure, path in SMRI_FILES.items():
    roi = pd.read_csv(path).drop(columns=["l???", "r???"], errors="ignore")
    roi["participant_id"] = roi["names"].astype(str).str.replace("_T1w", "", regex=False)
    data = sample[["participant_id", "Risk_Group"]].merge(roi, on="participant_id")
    features = [c for c in roi.columns if c not in ("names", "GROUP", "participant_id")]
    ranked = sorted(((kruskal_p(split_by_group(data[f], data["Risk_Group"])), f) for f in features
                     if not np.isnan(kruskal_p(split_by_group(data[f], data["Risk_Group"])))))
    for p_kw, feature in ranked[:SMRI_TOP_K]:
        for row in posthoc(split_by_group(data[feature], data["Risk_Group"])):
            if row["p_bonf"] < SMRI_REPORT_P:
                table5.append({"measure": measure, "region": feature, "p_kruskal": p_kw, **row})
pd.DataFrame(table5).round(4).to_csv(f"{OUT_DIR}/table5_smri_posthoc.csv", index=False)

# Table 6: clinical post hoc
table6 = []
for var in CLINICAL_VARS:
    by_group = split_by_group(sample[var], groups)
    p_kw = kruskal_p(by_group)
    if p_kw < CLINICAL_SELECT_P:
        table6 += [{"variable": var, "p_kruskal": p_kw, **row} for row in posthoc(by_group)]
pd.DataFrame(table6).round(4).to_csv(f"{OUT_DIR}/table6_clinical_posthoc.csv", index=False)

print(f"Wrote {OUT_DIR}/: table1 ({len(table1)} rows), table5 ({len(table5)} rows), table6 ({len(table6)} rows)")
