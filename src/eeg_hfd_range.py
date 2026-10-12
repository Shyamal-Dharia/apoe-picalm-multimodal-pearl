"""Range of the band-wise Higuchi fractal dimension (HFD) values in the analysis sample.

Run from the repository root after src/EEG_feature_extraction.py:
    python src/eeg_hfd_range.py
For a sampled curve the HFD lies between 1 and 2. With narrow-band filtering and a large k_max
(82 here) the estimate can exceed 2, so this reports, per band, the share of window x channel
values above 2 over the 64 participants (a quality check, not reported in the paper).
Writes results/tables/eeg_hfd_range.csv.
"""
import glob
import os

import numpy as np
import pandas as pd

FEATURE_DIR = "data/EEG_data/HFD_PSD_4sec_75overlap_6min"
SAMPLE_CSV = "data/participants_n64.csv"
OUT_CSV = "results/tables/eeg_hfd_range.csv"
BANDS = ["delta", "theta", "alpha", "beta", "gamma"]  # order of the last axis of HFD_features

sample = set(pd.read_csv(SAMPLE_CSV)["participant_id"])
values = []
for path in sorted(glob.glob(os.path.join(FEATURE_DIR, "*.npz"))):
    sid = f"sub-{int(os.path.basename(path).split('.')[0]):02d}"
    if sid in sample:
        values.append(np.load(path, allow_pickle=True)["HFD_features"])  # windows x channels x bands
assert len(values) == len(sample), f"{len(values)} feature files for {len(sample)} participants"
hfd = np.concatenate(values)

rows = []
for b, band in enumerate(BANDS):
    v = hfd[:, :, b].ravel()
    rows.append({"band": band, "n_values": v.size, "median": np.median(v), "p05": np.percentile(v, 5),
                 "p95": np.percentile(v, 95), "pct_above_2": 100 * np.mean(v > 2),
                 "pct_below_1": 100 * np.mean(v < 1)})
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
