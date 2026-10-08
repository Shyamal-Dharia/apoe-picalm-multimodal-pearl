#!/bin/bash
# Mediation analysis (grid + pairwise post hoc) on one Trillium CPU node.
# About 102,600 models x 10,000 simulations; roughly 5 h on 192 cores.
# The grid writes one file per outcome and skips finished outcomes, so if the job
# hits the time limit, resubmit it and it continues where it stopped.
#
# Setup (login node, once):
#   module load StdEnv/2023 r/4.3.1          # check versions with: module spider r
#   mkdir -p ~/R/library && export R_LIBS_USER=~/R/library
#   Rscript install_r_packages.R
# Submit from the project directory on $SCRATCH:
#   mkdir -p results/logs
#   sbatch --account=<def-account> cluster/trillium_mediation.sbatch
#SBATCH --job-name=mediation-n64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/mediation_%j.log

set -euo pipefail
module load StdEnv/2023 r/4.3.1
export R_LIBS_USER=~/R/library
export MEDIATION_CORES=${SLURM_CPUS_PER_TASK}
# One BLAS thread per worker; parallelism comes from the R workers
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 FLEXIBLAS_NUM_THREADS=1

Rscript src/mediation_grid.R
if [ -f results/mediation/mediation_grid_significant_acme.csv ]; then
  Rscript src/run_pairwise_significant_mediations.R
fi
