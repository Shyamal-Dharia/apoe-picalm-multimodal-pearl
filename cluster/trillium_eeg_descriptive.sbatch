#!/bin/bash
# Dynamic EEG group comparisons (src/eeg_std_distance_multiseed.py: k-means states on all
# participants' windows, five seeds, k = 5-10; within-state HFD variability per participant).
# Runs two versions side by side on one Trillium node, each writing to its own folder:
#   min1  one window is enough (the SD of a single window is 0; reproduces the original run)
#   min2  at least two windows per state, otherwise the value is missing
# Submit from the repository root, e.g.
#   sbatch --account=<def-account> --export=ALL,PYTHON=$HOME/envs/mmeeg/bin/python,OUT_BASE=$SCRATCH/MM-eeg-desc \
#     cluster/trillium_eeg_descriptive.sbatch   (Trillium starts jobs with --export=NONE)
#SBATCH --job-name=eeg-descriptive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --output=logs/eeg_descriptive_%j.log

set -euo pipefail
PY=${PYTHON:-python}
: "${OUT_BASE:?set OUT_BASE}"
export OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48 MKL_NUM_THREADS=48 MPLBACKEND=Agg
for m in 1 2; do
  mkdir -p "$OUT_BASE/min$m"
  EEG_MIN_WINDOWS=$m EEG_RESULTS_DIR="$OUT_BASE/min$m" "$PY" src/eeg_std_distance_multiseed.py \
    > "$OUT_BASE/min$m/run.log" 2>&1 &
done
wait
ls "$OUT_BASE"/min*/
