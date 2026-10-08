"""Select the analysis sample: genotype group + complete EEG, fMRI and sMRI data.

Groups (PEARL-Neuro participants table):
  N     no APOE-e4, PICALM rs3851179 not G/G
  A+P-  APOE-e4 carrier, PICALM not G/G
  A+P+  APOE-e4 carrier, PICALM G/G
Any APOE-e2 carrier is excluded (as in src/fmri_sliding_window_fc.py).
No APOE-e4 non-carrier with PICALM G/G (A-P+) has neuroimaging data.

Run from the repository root:  python src/select_participants.py
Writes
  data/filtered_participants.tsv  phase-2 participants with a genotype group (Group column),
                                  read by the permutation and fMRI pipelines;
  data/participants_n64.csv       the analysis sample (n = 64: N 25, A+P- 23, A+P+ 16).
"""
import os
import re

import numpy as np
import pandas as pd

PARTICIPANTS_TSV = "data/participants.csv"  # OpenNeuro ds004796 participants.tsv (tab-separated)
EEG_DIR = "data/EEG_data/HFD_PSD_4sec_75overlap_6min"
FMRI_DIR = "data/fMRI_data/subject_windows_confounds_old_60sec"
SMRI_CSV = "data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv"
GROUPS_TSV = "data/filtered_participants.tsv"
OUT_CSV = "data/participants_n64.csv"
GROUP_CODE = {"N": 0, "A+P-": 1, "A+P+": 2}


def subject_ids(names):
    return {"sub-%02d" % int(re.search(r"(\d+)", str(n)).group(1)) for n in names}


df = pd.read_csv(PARTICIPANTS_TSV, sep="\t")
e4 = df["APOE_haplotype"].str.contains("e4", case=False)
e2 = df["APOE_haplotype"].str.contains("e2", case=False)
picalm_gg = df["PICALM_rs3851179"] == "G/G"

group = np.select([~e4 & ~picalm_gg, e4 & ~picalm_gg, e4 & picalm_gg], ["N", "A+P-", "A+P+"], "Excluded")
group = np.where(e2, "Excluded", group)
df["Risk_Group"] = group

phase2 = df[(df["Risk_Group"] != "Excluded") & (df["second_phase"] == 1)]
phase2.drop(columns="Risk_Group").assign(Group=phase2["Risk_Group"]).to_csv(GROUPS_TSV, sep="\t", index=False)

eeg = subject_ids(f for f in os.listdir(EEG_DIR) if f.endswith(".npz"))
fmri = subject_ids(f for f in os.listdir(FMRI_DIR) if f.endswith(".npz"))
smri = subject_ids(pd.read_csv(SMRI_CSV)["names"])

sample = df[(df["Risk_Group"] != "Excluded") & df["participant_id"].isin(eeg & fmri & smri)].copy()
sample["Group"] = sample["Risk_Group"].map(GROUP_CODE)
sample.to_csv(OUT_CSV, index=False)

print(f"Wrote {GROUPS_TSV}: n = {len(phase2)}")
print(f"Wrote {OUT_CSV}: n = {len(sample)}")
print(sample["Risk_Group"].value_counts().reindex(GROUP_CODE).to_string())
