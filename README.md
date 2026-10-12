# MM-Mediation-Genes-AD

**Parallel Multimodal Testing of APOE-ε4 and PICALM rs3851179 Genotype Groups in Healthy Middle-Aged Adults Finds No Differences That Survive Correction**

Analysis code for a comparison of three genotype groups of the PEARL-Neuro cohort (n = 64) in
four modalities: dynamic EEG, dynamic fMRI, structural MRI and 75 non-imaging measures. The
groups are non-carriers (N, n = 25), APOE-ε4 carriers without the PICALM rs3851179 G/G genotype
(A+P-, n = 23) and APOE-ε4 carriers with G/G (A+P+, n = 16). Each modality is classified in four
group comparisons with leave-one-subject-out cross-validation and tested against a permutation
null, with one Bonferroni correction across the 16 tests. Descriptive group comparisons and
exploratory mediation models follow.

<p align="center">
  <img src="figures/MA-v4.png" width="650">
</p>

## Setup

```bash
pip install -r requirements.txt                         # Python 3.10
Rscript install_r_packages.R                            # R 4.3.1
gcc -shared -fPIC -O2 -o src/libhfd.so src/hfd.c        # Higuchi fractal dimension (optional, faster)
```

## Data

PEARL-Neuro, OpenNeuro [ds004796](https://openneuro.org/datasets/ds004796) (v1.0.9). The
repository holds code only. The scripts create the folders under `data/` and `results/`.

| Input | Source | Location expected by the code |
|---|---|---|
| Participants table | `participants.tsv` from the dataset | `data/participants.csv` (tab-separated) |
| Raw resting-state EEG | dataset | folder given by `PEARL_ROOT` (step 1) |
| Preprocessed fMRI | FSL TOPUP (AP/PA), then fMRIPrep 25.0.0 without its own distortion correction | folder given by `FMRIPREP_DIR` (steps 2 and 8) |
| sMRI ROI measures | CAT12 ROI exports (HCP-MMP1: depth, fractal dimension, gyrification, thickness; AAL3: grey-matter volume), with a `GROUP` column | `data/MRI_data/ROI_*_with_groups.csv` |

## Reproducing the results

Run every command from the repository root unless noted. Table and figure numbers follow the
manuscript.

| Step | Command | Produces |
|---|---|---|
| 1. EEG preprocessing and window features (4 s windows, 1 s step, HFD per channel and band) | `PEARL_ROOT=/path python src/EEG_raw_preprocessing.py`<br>`python src/EEG_feature_extraction.py`<br>(`cluster/trillium_eeg_features.sbatch` runs the second in parallel shards) | `data/EEG_data/` |
| 2. fMRI sliding-window connectivity (60 s windows, 15.2 s step) | `FMRIPREP_DIR=/path python src/fmri_sliding_window_fc.py` | `data/fMRI_data/` |
| 3. Genotype groups and analysis sample (n = 64) | `python src/select_participants.py` | `data/filtered_participants.tsv`, `data/participants_n64.csv`, `results/tables/sample_flow.csv` |
| 4. Classification and permutation tests (16 tests) | `cd src && python f1_permutation_pipeline.py --table --seeds 9 42 123 2021 2022 --perm-seeds 9 --permutations 1000 --jobs 8`<br>(`cluster/trillium_permutation.sbatch`, one job per comparison, then `--assemble`) | Table 2 |
| 5. Dynamic EEG states | `python src/eeg_std_distance_multiseed.py`<br>(`cluster/trillium_eeg_descriptive.sbatch`) | `results/eeg/` (data for Figure 2) |
| 6. fMRI dwell time and connectivity | `python src/fmri_significant_clusters_60s_75overlap.py` | `results/fmri/` (data for Figure 3, Table S3) |
| 7. Group comparison tables | `python src/descriptive_tables.py` | `results/tables/` (Tables 1 and 3, Table S4) |
| 8. fMRI head motion | `FMRIPREP_DIR=/path python src/fmri_motion_qc.py` | Table S2 |
| 9. Figures | `python src/paper_figures.py` | Figures 2–5 |
| 10. Elbow curves | `python src/kmeans_elbow.py` | Figure S4 |
| 11. LaTeX tables | `TEMPLATEFLOW_HOME=/path python src/latex_tables.py` | Tables 1–3, S3, S4; `results/tables/permutation_summary.csv` |
| 12. Power reference | `python src/power_reference.py` | `results/tables/minimum_detectable_effect.csv`, `replication_sample_size.csv` |
| 13. Mediation input | `python src/build_mediation_input.py` (after steps 5, 6 and 9) | `data/mediation_input_n64.csv` |
| 14. Mediation | `MEDIATION_CORES=32 Rscript src/mediation_grid.R`<br>`MEDIATION_CORES=32 MEDIATION_SET=dynamic Rscript src/mediation_grid.R`<br>`Rscript src/run_pairwise_significant_mediations.R` (planned follow-up, runs only for pairs with q < 0.05)<br>(`cluster/trillium_mediation.sbatch`, `cluster/trillium_mediation_dynamic.sbatch`) | `results/mediation/`, `results/mediation/dynamic/` |
| 15. Mediation summary | `python src/mediation_summary.py`<br>`python src/mediation_summary.py results/mediation/dynamic results/latex dynamic` | counts in the text, Table S5 |
| 16. Graphical abstract | `python src/render_nilearn_surface_parcels_lateral.py`<br>`python src/render_nilearn_surface_docs_gray.py`<br>`python src/graphical_abstract.py && python src/graphical_abstract_journal.py`<br>`python src/ga_result_grid.py` | `results/figures/graphical_abstract*`, `ga_result_grid*` |

Notes
- Groups: A+ = any APOE-ε4 allele; P+ = PICALM rs3851179 G/G; APOE-ε2 carriers are excluded.
  The sample is every phase-2 participant with a group and EEG, fMRI and sMRI data.
- EEG bands are filtered with a fifth-order Butterworth filter in second-order sections (the
  (b, a) form is numerically unstable for the delta band at 256 Hz). The dynamic EEG feature is
  the SD of a participant's window distances to a state centroid, computed only when the
  participant has at least two windows in the state (`EEG_MIN_STATE_WINDOWS` in
  `src/f1_permutation_pipeline.py`, `EEG_MIN_WINDOWS` in `src/eeg_std_distance_multiseed.py`).
- Step 2 regresses 6 motion parameters and the first aCompCor component (combined WM+CSF
  mask), with linear detrending and 0.01–0.1 Hz band-pass, on the Schaefer-100 (7-network)
  atlas from TemplateFlow; windows are 75 volumes with a 19-volume step (26 per run), and
  connectivity is nilearn's Ledoit-Wolf correlation.
- Step 4 builds the null by permuting the subject labels once per permutation and rerunning
  the cross-validation (`--perm-scheme full`, the default). Participants are processed in a
  fixed order, so the k-means initialisation and the results do not depend on Python's hash
  seed. `--perm-scheme per-fold` reproduces the April 2026 table, which shuffled each fold's
  training labels separately.
- Step 14 fits about 102,600 sMRI models and 256 dynamic models with 10,000 simulations each
  (about 38 CPU-days). The grid script writes one file per outcome and resumes after
  interruption. `cluster/trillium_mediation.sbatch` runs it on one 192-core SLURM node.
- Random seeds are fixed throughout: 9, 42, 123, 2021 and 2022 for clustering and
  classification, and one seed per mediation model, so results do not depend on core count.
- Steps 7 and 9 share the three-group post hoc test in `src/group_stats.py` (ANOVA contrasts,
  Bonferroni p over three contrasts, Tukey-adjusted CIs, Cohen's d), as in JASP.
- Step 16 also needs `data/EEG_data/channel_names.txt`, the 127 EEG channel labels in recording
  order (one per line, from any recording's `.vhdr` header).
- Not needed to reproduce the paper: the notebooks (`EEG_feature_analysis.ipynb`,
  `fMRI_feature_analysis.ipynb`, `fMRI_viz.ipynb`, `final_permutation_test.ipynb`,
  `mediation_pairwise_tables.ipynb`, `mediation_filter_significant.ipynb`, and
  `k-means_plot.ipynb`, which `src/kmeans_elbow.py` replaces), `eeg_permutation_hfd.py`,
  `eeg_hfd_range.py` (a quality check of the HFD range), `subject_deviation_profiles.py`
  (exploratory, written for an earlier draft) and the other `render_nilearn_*.py` scripts.

## Citation

Will be provided upon acceptance of the manuscript.
