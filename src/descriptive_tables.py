"""Group-comparison tables: baseline (Table 1), non-imaging post hoc (Table 3), sMRI post hoc (Table S4).

Run from the repository root after src/select_participants.py:
    python src/descriptive_tables.py
Writes CSVs to results/tables/ at full precision (round when reporting). The file names
table5_smri_posthoc.csv and table6_clinical_posthoc.csv keep the table numbers of the first
submission; they hold Supplementary Table S4 and Table 3 of the paper.

Post hoc comparisons follow JASP's one-way ANOVA post hoc output (src/group_stats.py):
mean difference (group 1 - group 2), standard error from the pooled ANOVA error term,
Bonferroni-adjusted p-values over the three pairwise contrasts, Tukey-adjusted 95%
confidence intervals, and Cohen's d (mean difference / pooled ANOVA SD).

Feature selection (no correction across features):
  sMRI      the 5 features per measure with the lowest Kruskal-Wallis p; contrasts with
            Bonferroni p < .10 are reported (the dagger threshold used in all tables).
  Clinical  variables with Kruskal-Wallis p < .05 (absolute blood cell counts; the
            percentage variants are not tested); all three contrasts are reported.
q_fdr_kw is the Benjamini-Hochberg q of the Kruskal-Wallis p across all features of the
same family (all regions of one sMRI measure; all clinical variables). It documents the
selection, which itself is uncorrected.

table1_baseline.csv adds the baseline characteristics and genotype counts per group.
smri_kruskal_all.csv and nonimaging_kruskal_all.csv list the Kruskal-Wallis p and q of all sMRI regions
and all non-imaging measures.
"""
import os

import numpy as np
import pandas as pd

from group_stats import GROUPS, fdr_q, kruskal_p, posthoc, split_by_group

SAMPLE_CSV = "data/participants_n64.csv"
OUT_DIR = "results/tables"

SMRI_FILES = {
    "Sulcal Depth": "data/MRI_data/ROI_aparc_HCP_MMP1_depth.csv_with_groups.csv",
    "Cortical Thickness": "data/MRI_data/ROI_aparc_HCP_MMP1_thickness.csv_with_groups.csv",
    "Gyrification": "data/MRI_data/ROI_aparc_HCP_MMP1_gyrification.csv_with_groups.csv",
    "Gray Matter Volume": "data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv",
    "Fractal Dimension": "data/MRI_data/ROI_aparc_HCP_MMP1_fractaldimension.csv_with_groups.csv",
}
SMRI_TOP_K = 5
SMRI_REPORT_P = 0.10  # same threshold as the dagger marker in the other tables and figures
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

# Table 1 (extended): baseline characteristics and genotypes, counts or mean (SD)
CATEGORICAL_BASELINE = {
    "APOE_haplotype": "APOE haplotype",
    "PICALM_rs3851179": "PICALM rs3851179",
    "education": "Education (0 primary, 1 secondary, 2 partial higher, 3 higher)",
    "hypertension": "Hypertension (1 = yes)",
    "diabetes": "Diabetes (1 = yes)",
    "dementia_history_parents": "Parents with dementia (0, 1, 2)",
    "smoking_status": "Smoking (0 no, 1 yes, 2 past)",
}
baseline = []
parts = [*((g, sample[groups == g]) for g in GROUPS), ("Overall", sample)]
for var in ["age", "BMI"]:
    baseline.append({"variable": var, "level": "mean (SD)",
                     **{name: f"{part[var].mean():.2f} ({part[var].std():.2f})" for name, part in parts}})
for var, label in CATEGORICAL_BASELINE.items():
    values = sample[var].map(lambda v: "missing" if pd.isna(v) else str(v).strip())
    values = values.str.replace(r"\.0$", "", regex=True)
    for level in sorted(values.unique()):
        baseline.append({"variable": label, "level": level,
                         **{name: int((values[part.index] == level).sum()) for name, part in parts}})
pd.DataFrame(baseline).to_csv(f"{OUT_DIR}/table1_baseline.csv", index=False)

# Table S4: sMRI post hoc
table5 = []
smri_all = []  # Kruskal-Wallis p and q of every sMRI region
for measure, path in SMRI_FILES.items():
    roi = pd.read_csv(path).drop(columns=["l???", "r???"], errors="ignore")
    roi["participant_id"] = roi["names"].astype(str).str.replace("_T1w", "", regex=False)
    data = sample[["participant_id", "Risk_Group"]].merge(roi, on="participant_id")
    features = [c for c in roi.columns if c not in ("names", "GROUP", "participant_id")]
    ranked = sorted(((kruskal_p(split_by_group(data[f], data["Risk_Group"])), f) for f in features
                     if not np.isnan(kruskal_p(split_by_group(data[f], data["Risk_Group"])))))
    q_by_feature = fdr_q({f: p for p, f in ranked})
    smri_all += [{"measure": measure, "region": f, "p_kruskal": p, "q_fdr_kw": q_by_feature[f]} for p, f in ranked]
    for p_kw, feature in ranked[:SMRI_TOP_K]:
        for row in posthoc(split_by_group(data[feature], data["Risk_Group"])):
            if row["p_bonf"] < SMRI_REPORT_P:
                table5.append({"measure": measure, "region": feature, "p_kruskal": p_kw, **row,
                               "q_fdr_kw": q_by_feature[feature], "n_features_in_family": len(ranked)})
pd.DataFrame(table5).to_csv(f"{OUT_DIR}/table5_smri_posthoc.csv", index=False)
pd.DataFrame(smri_all).to_csv(f"{OUT_DIR}/smri_kruskal_all.csv", index=False)

# Table 3: non-imaging post hoc
table6 = []
p_by_var = {var: kruskal_p(split_by_group(sample[var], groups)) for var in CLINICAL_VARS}
q_by_var = fdr_q(p_by_var)
for var in CLINICAL_VARS:
    by_group = split_by_group(sample[var], groups)
    p_kw = p_by_var[var]
    if p_kw < CLINICAL_SELECT_P:
        table6 += [{"variable": var, "p_kruskal": p_kw, **row, "q_fdr_kw": q_by_var[var],
                    "n_features_in_family": len(q_by_var)} for row in posthoc(by_group)]
pd.DataFrame(table6).to_csv(f"{OUT_DIR}/table6_clinical_posthoc.csv", index=False)
# Kruskal-Wallis p and q of every non-imaging measure (diabetes, constant, has no test)
pd.DataFrame({"variable": CLINICAL_VARS, "p_kruskal": [p_by_var[v] for v in CLINICAL_VARS],
              "q_fdr_kw": [q_by_var.get(v, np.nan) for v in CLINICAL_VARS]}).to_csv(
    f"{OUT_DIR}/nonimaging_kruskal_all.csv", index=False)

print(f"Wrote {OUT_DIR}/: table1 ({len(table1)} rows), table1_baseline ({len(baseline)} rows), "
      f"table5 ({len(table5)} rows), table6 ({len(table6)} rows)")
