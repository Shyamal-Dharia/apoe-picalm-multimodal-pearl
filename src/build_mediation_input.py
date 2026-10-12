"""Build the mediation input table: sMRI mediators, covariates and clinical outcomes.

Run from the repository root after src/select_participants.py, the EEG and fMRI state scripts
and src/paper_figures.py (which writes the fMRI time-in-state averages):
    python src/build_mediation_input.py
Writes data/mediation_input_n64.csv with
    participant_id, gene (0 = N, 1 = A+P-, 2 = A+P+),
    med_<measure>_<ROI>  sMRI mediators (CAT12 ROI exports),
    med_EEG_*, med_fMRI_*  four dynamic mediators: the participant-level EEG and fMRI features
                         of the paper (within-state HFD variability in N- and A+P+-highest EEG
                         states, dwell time in N-dominant and carrier-dominant fMRI states; with
                         the two-window rule no EEG state had its highest mean in A+P-). Their states were selected for group differences in these
                         participants, so the genotype -> mediator paths are inflated by that
                         selection. src/mediation_grid.R runs them as a separate set
                         (MEDIATION_SET=dynamic).
    covariates (no prefix),
    y_<outcome>          clinical outcomes.
"""
import pandas as pd

SAMPLE_CSV = "data/participants_n64.csv"
PARTICIPANTS_TSV = "data/participants.csv"  # OpenNeuro ds004796 participants.tsv (tab-separated)
OUT_CSV = "data/mediation_input_n64.csv"
DYNAMIC_MEDIATORS = {
    "med_EEG_var_N_states": ("results/eeg/eeg_subject_mean_variability_N_higher_multiseed.csv", "mean_variability"),
    "med_EEG_var_AposPpos_states": ("results/eeg/eeg_subject_mean_variability_A_plus_P_plus_higher_multiseed.csv", "mean_variability"),
    "med_fMRI_time_N_states": ("results/figures/fmri_subject_mean_dwell_time.csv", "dwell_N_states"),
    "med_fMRI_time_carrier_states": ("results/figures/fmri_subject_mean_dwell_time.csv", "dwell_carrier_states"),
}

# Mediator prefix -> CAT12 ROI export (order fixes the column order, and so the per-model seeds)
SMRI_FILES = {
    "fd": "data/MRI_data/ROI_aparc_HCP_MMP1_fractaldimension.csv_with_groups.csv",
    "depth": "data/MRI_data/ROI_aparc_HCP_MMP1_depth.csv_with_groups.csv",
    "thickness": "data/MRI_data/ROI_aparc_HCP_MMP1_thickness.csv_with_groups.csv",
    "grey": "data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv",
    "gyrification": "data/MRI_data/ROI_aparc_HCP_MMP1_gyrification.csv_with_groups.csv",
}

COVARIATES = [
    "age", "sex", "education", "learning_deficits", "BMI", "allergies", "drugs",
    "ibuprofen_intake", "thyroid_diseases", "hypertension", "diabetes", "other_diseases",
    "smoking_status", "coffee_status", "dementia_history_parents", "HSV_r",
]
OUTCOMES = (
    ["BDI", "SES", "RPM", "EHI", "NEO_NEU", "NEO_EXT", "NEO_OPE", "NEO_AGR", "NEO_CON", "AUDIT"]
    + [f"MINI-COPE_{i}" for i in range(1, 15)]
    + [f"CVLT_{i}" for i in range(1, 14)]
    + ["leukocytes", "erythrocytes", "hemoglobin", "hematocrit", "MCV", "MCH", "MCHC", "RDW-CV",
       "platelets", "PDW", "MPV", "P-LCR", "neutrophils", "lymphocytes", "monocytes",
       "eosinophils", "basophils", "neutrophils_%", "lymphocytes_%", "monocytes_%",
       "eosinophils_%", "basophils_%", "total_cholesterol", "cholesterol_HDL",
       "non-HDL_cholesterol", "LDL_cholesterol", "triglycerides"]
)

sample = pd.read_csv(SAMPLE_CSV)[["participant_id", "Group"]].rename(columns={"Group": "gene"})

mediators = []
for prefix, path in SMRI_FILES.items():
    roi = pd.read_csv(path).drop(columns=["l???", "r???", "GROUP"], errors="ignore")
    roi["participant_id"] = roi.pop("names").astype(str).str.replace("_T1w", "", regex=False)
    mediators.append(roi.set_index("participant_id").add_prefix(f"med_{prefix}_"))
for name, (path, column) in DYNAMIC_MEDIATORS.items():  # after the sMRI columns
    mediators.append(pd.read_csv(path).set_index("participant_id")[[column]].rename(columns={column: name}))
mediators = pd.concat(mediators, axis=1, join="inner").astype(float)

clinical = pd.read_csv(PARTICIPANTS_TSV, sep="\t").set_index("participant_id")
# learning_deficits holds a malformed value ('[0. 1 ]') for one sampled subject; it is set
# to missing, so that subject drops out of the mediation models (n = 63).
covariates = clinical[COVARIATES].apply(pd.to_numeric, errors="coerce")
outcomes = clinical[OUTCOMES].astype(float).add_prefix("y_")

table = (sample.set_index("participant_id")
         .join(mediators, how="inner")
         .join(covariates)
         .join(outcomes)
         .reset_index())
assert len(table) == len(sample), "every sampled subject needs sMRI data"
table.to_csv(OUT_CSV, index=False)

print(f"Wrote {OUT_CSV}: {len(table)} subjects, {mediators.shape[1]} mediators, "
      f"{len(COVARIATES)} covariates, {len(OUTCOMES)} outcomes")
