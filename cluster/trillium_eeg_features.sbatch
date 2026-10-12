#!/bin/bash
# EEG window features (HFD per channel and band, 4-s windows, 1-s step) for all preprocessed
# recordings, in parallel shards on one Trillium node. Input: data/EEG_data/
# resting_state_preprocessed_data_new (read-only is fine); output: EEG_FEATURES_OUT (on $SCRATCH).
# Submit from a working directory that has data/ and src/ (see README), e.g.
#   PYTHON=~/envs/mmgenes/bin/python EEG_FEATURES_OUT=$SCRATCH/MM-eeg/out sbatch --account=<def-account> cluster/trillium_eeg_features.sbatch
#SBATCH --job-name=eeg-features
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --output=logs/eeg_features_%j.log

set -euo pipefail
PY=${PYTHON:-python}
N=${N_SHARDS:-75}
mkdir -p "${EEG_FEATURES_OUT:?set EEG_FEATURES_OUT}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for i in $(seq 0 $((N - 1))); do
  SHARD=$i N_SHARDS=$N "$PY" src/EEG_feature_extraction.py > "logs/eeg_features_shard_${i}.log" 2>&1 &
done
wait
ls "$EEG_FEATURES_OUT" | wc -l
