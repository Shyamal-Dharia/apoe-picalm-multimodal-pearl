#!/bin/bash
# Mediation with the four dynamic EEG/fMRI mediators (MEDIATION_SET=dynamic): 64 outcomes x 4
# mediators, a separate family from the sMRI grid with its own FDR. Outputs go to
# results/mediation/dynamic/. Every outcome runs in parallel; the grid is then combined and
# FDR-corrected, and the pairwise follow-up runs if any pair passes.
#
# Setup as in cluster/trillium_mediation.sbatch. Submit from the project directory on $SCRATCH:
#   mkdir -p results/logs
#   sbatch --account=<def-account> cluster/trillium_mediation_dynamic.sbatch
#SBATCH --job-name=mediation-dynamic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/mediation_dynamic_%j.log

set -euo pipefail
module load StdEnv/2023 r/4.3.1
export R_LIBS_USER=~/R/library
export MEDIATION_SET=dynamic
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1

n_outcomes=$(head -1 data/mediation_input_n64.csv | tr ',' '\n' | grep -c '^"\?y_')
mkdir -p results/logs/mediation_dynamic_parts
for i in $(seq 1 "$n_outcomes"); do
  MEDIATION_CORES=2 Rscript src/mediation_grid.R "$i" "$i" \
    > "results/logs/mediation_dynamic_parts/outcome_$i.log" 2>&1 &
done
wait

Rscript src/mediation_grid.R   # every part exists now, so this only combines and FDR-corrects
sig=results/mediation/dynamic/mediation_grid_significant_acme.csv
if [ "$(wc -l < "$sig")" -gt 1 ]; then
  MEDIATION_CORES=${SLURM_CPUS_PER_TASK} Rscript src/run_pairwise_significant_mediations.R
fi
