#!/bin/bash
# Permutation tests (Table 2) on one Trillium CPU node, with the standard null: the subject
# label vector is permuted once per permutation and the whole cross-validation is rerun.
# Five observed seeds, 1,000 permutations (seed 9), four comparisons x four modalities.
#
# Setup (login node, once): a Python environment with the versions in requirements.txt,
# e.g. ~/envs/mmperm; set PYTHON to its interpreter.
# Submit from the project directory on $SCRATCH (data/ must hold the EEG, fMRI and sMRI inputs
# and filtered_participants.tsv):
#   mkdir -p results/logs
#   PYTHON=~/envs/mmperm/bin/python sbatch --account=<def-account> cluster/trillium_permutation.sbatch
# The four comparisons can run as parallel jobs (TASK=1..4, each writing results/permutation/task_N),
# then be combined:
#   (OUT_BASE, relative to src/, sets another output folder.)
#   for k in 1 2 3 4; do sbatch --account=<def-account> --export=ALL,TASK=$k,PYTHON=... cluster/trillium_permutation.sbatch; done
#   cd src && python f1_permutation_pipeline.py --assemble ../results/permutation/task_{1,2,3,4} --out-dir ../results/permutation
#SBATCH --job-name=permutation-full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/permutation_full_%x_%j.log

set -euo pipefail
PY=${PYTHON:-python}
cd src   # utils.py reads ../data
OUT=${OUT_BASE:-../results/permutation}${TASK:+/task_$TASK}
"$PY" -u f1_permutation_pipeline.py --table --seeds 9 42 123 2021 2022 --perm-seeds 9 \
  --permutations 1000 --jobs "${SLURM_CPUS_PER_TASK}" --perm-scheme full \
  ${TASK:+--tasks $TASK} --out-dir "$OUT"
