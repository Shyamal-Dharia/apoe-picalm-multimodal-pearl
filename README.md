# MM-Mediation-Genes-AD

**An Exploration of Multimodal Neuroimaging and Clinical Associations of APOE-ε4 and PICALM rs3851179 in Healthy Middle-Aged Adults: A Cohort Study**

Analysis code for a multimodal (EEG, fMRI, sMRI, clinical) comparison of three genotype groups
in the PEARL-Neuro cohort (n = 64): non-carriers (N, n = 25), APOE-ε4 carriers without the
PICALM rs3851179 G/G genotype (A+P-, n = 23), and APOE-ε4 carriers with G/G (A+P+, n = 16).

<p align="center">
  <img src="figures/MA-v4.png" width="650">
</p>

## Setup

```bash
pip install -r requirements.txt                         # Python 3.10
Rscript install_r_packages.R                            # R 4.3.1
gcc -shared -fPIC -O2 -o src/libhfd.so src/hfd.c        # Higuchi fractal dimension
```

## Data

PEARL-Neuro, OpenNeuro [ds004796](https://openneuro.org/datasets/ds004796) (v1.0.9).

| Input | Source | Location expected by the code |
|---|---|---|
| Participants table | `participants.tsv` from the dataset | `data/participants.csv` (tab-separated) |
| Raw resting-state EEG | dataset | folder given by `PEARL_ROOT` (see step 1) |
| Preprocessed fMRI | FSL TOPUP, then fMRIPrep 25.0.0 | folder given by `FMRIPREP_DIR` (step 2) |
| sMRI ROI measures | CAT12 ROI exports (HCP-MMP1: depth, fractal dimension, gyrification, thickness; AAL3: grey-matter volume), with a `GROUP` column | `data/MRI_data/ROI_*_with_groups.csv` |

## Reproducing the results

Run every command from the repository root unless noted. Outputs go to `results/`.

| Step | Command | Produces |
|---|---|---|
| 1. EEG preprocessing and HFD features (4 s windows, 75% overlap) | `PEARL_ROOT=/path python src/EEG_raw_preprocessing.py`<br>`python src/EEG_feature_extraction.py` | `data/EEG_data/` |
| 2. fMRI sliding-window connectivity (60 s windows, 15.2 s step) | `FMRIPREP_DIR=/path python src/fmri_sliding_window_fc.py` | `data/fMRI_data/` |
| 3. Genotype groups and analysis sample (n = 64) | `python src/select_participants.py` | `data/filtered_participants.tsv`, `data/participants_n64.csv` |
| 4. Permutation tests | `cd src && python f1_permutation_pipeline.py --table --seeds 9 42 123 2021 2022 --perm-seeds 9 --permutations 1000 --jobs 8` | Table 3 |
| 5. EEG dynamic features | `python src/eeg_std_distance_multiseed.py` | Fig 6 |
| 6. fMRI dwell time and connectivity | `python src/fmri_significant_clusters_60s_75overlap.py` | Fig 7, Table 4 |
| 7. Group comparison tables | `python src/descriptive_tables.py` | Tables 1, 5, 6 |
| 8. Mediation input | `python src/build_mediation_input.py` | `data/mediation_input_n64.csv` |
| 9. Mediation | `MEDIATION_CORES=32 Rscript src/mediation_grid.R`<br>`MEDIATION_CORES=32 Rscript src/run_pairwise_significant_mediations.R` | `results/mediation/` |
| 10. Mediation tables | `src/mediation_pairwise_tables.ipynb` | Tables 7–8 |
| 11. Elbow curves | `src/k-means_plot.ipynb` | Fig 5 |

Notes
- Groups: A+ = any APOE-ε4 allele; P+ = PICALM rs3851179 G/G; APOE-ε2 carriers are excluded.
  The sample is every phase-2 participant with a group and EEG, fMRI and sMRI data.
- Step 2 regresses 6 motion parameters and the first aCompCor component (combined WM+CSF
  mask), with linear detrending and 0.01–0.1 Hz band-pass, on the Schaefer-100 (7-network)
  atlas from TemplateFlow; windows are 75 volumes with a 19-volume step (26 per run), and
  connectivity is nilearn's Ledoit-Wolf correlation.
- Step 9 fits about 102,600 models with 10,000 simulations each (about 38 CPU-days). The grid
  script writes one file per outcome and resumes after interruption.
  `cluster/trillium_mediation.sbatch` runs it on one 192-core SLURM node.
- Random seeds are fixed throughout: 9, 42, 123, 2021 and 2022 for clustering and
  classification, and one seed per mediation model, so results do not depend on core count.
- `EEG_feature_analysis.ipynb`, `fMRI_feature_analysis.ipynb`, `fMRI_viz.ipynb`,
  `final_permutation_test.ipynb` and `eeg_permutation_hfd.py` are exploratory and not needed to
  reproduce the paper.

## Citation

Will be provided upon acceptance of the manuscript.
