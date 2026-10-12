from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import templateflow.api as tf
from nilearn import image
from nilearn.plotting import view_connectome
from scipy import ndimage
from scipy.stats import (
    f_oneway,
    kruskal,
    levene,
    mannwhitneyu,
    shapiro,
    ttest_ind,
)
from sklearn.cluster import KMeans


# ============================================================================
# Configuration
# ============================================================================

# Window geometry of src/fmri_sliding_window_fc.py: 75 volumes, step 19 volumes, TR 0.8 s
FMRI_WINDOW_SEC = 75 * 0.8               # 60 s
FMRI_STEP_SEC = 19 * 0.8                 # 15.2 s
FMRI_OVERLAP = 1.0 - 19 / 75             # 0.747

K_STATES_LIST = list(range(4, 10))
SEED_LIST = [9, 42, 123, 2021, 2022]
P_VALUE_THRESHOLD = 0.05

PARTICIPANTS_TSV = "data/filtered_participants.tsv"  # from src/select_participants.py
SAMPLE_CSV = "data/participants_n64.csv"  # analysis sample, from src/select_participants.py

FMRI_DIR_CANDIDATES = [
    Path("data/fMRI_data/subject_windows_confounds_old_60sec"),
]

# Used only to restrict to subjects with EEG data.
EEG_DIR_CANDIDATES = [
    Path("data/EEG_data/HFD_PSD_4sec_75overlap_6min"),
]

OUTPUT_ROOT = Path("results/fmri/fMRI_significant_clusters_60s_75overlap_multiseed")
COMBINED_UNIFIED_TOP_PAIRS_CSV = Path("results/fmri/fMRI_unified_top_fc_pairs_60s_75overlap_multiseed.csv")
SIGNIFICANT_CLUSTER_SUMMARY_CSV = Path("results/fmri/fMRI_significant_cluster_summary_60s_75overlap_multiseed.csv")
N_DIRECTION_TOP_PAIRS_CSV = Path("results/fmri/fMRI_top_fc_pairs_N_higher_multiseed.csv")
RISK_DIRECTION_TOP_PAIRS_CSV = Path("results/fmri/fMRI_top_fc_pairs_risk_higher_multiseed.csv")
N_DIRECTION_CONNECTOME_HTML = Path("results/fmri/fMRI_top_fc_pairs_N_higher_multiseed.html")
RISK_DIRECTION_CONNECTOME_HTML = Path("results/fmri/fMRI_top_fc_pairs_risk_higher_multiseed.html")

POSTHOC_SEED = 9
POSTHOC_K_VALUES = [7, 8]
POSTHOC_BOOTSTRAP_SAMPLES = 5000

GROUP_LABEL_TO_NAME = {
    0: "N",
    1: "A+P-",
    2: "A+P+",
}

GROUP_LABEL_TO_SAFE_NAME = {
    0: "N",
    1: "AposPminus",
    2: "AposPpos",
}

SAFE_NAME_TO_READABLE = {
    "N": "N",
    "AposPminus": "A+P-",
    "AposPpos": "A+P+",
}


# ============================================================================
# Utilities
# ============================================================================

def resolve_existing_path(path_candidates: list[Path], label: str) -> Path:
    for path in path_candidates:
        if path.exists():
            return path
    tried = "\n".join(str(p) for p in path_candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")


def extract_subject_id_from_filename(filename: str) -> str | None:
    match = re.search(r"sub-(\d+)", filename)
    if match:
        return match.group(1).zfill(2)

    match = re.match(r"(\d+)", filename)
    if match:
        return match.group(1).zfill(2)

    return None


def collect_subject_ids_from_directory(data_dir: Path, suffix: str = ".npz") -> set[str]:
    subject_ids: set[str] = set()
    for filename in os.listdir(data_dir):
        if not filename.endswith(suffix):
            continue
        subject_id = extract_subject_id_from_filename(filename)
        if subject_id is not None:
            subject_ids.add(subject_id)
    return subject_ids


def load_fmri_window_data(fmri_dir: Path, keep_subjects: set[str]) -> dict[str, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    sub_ids: list[str] = []

    label_stats: Counter[int] = Counter()

    for filename in sorted(os.listdir(fmri_dir)):
        if not filename.endswith("_windows.npz"):
            continue

        subject_id = extract_subject_id_from_filename(filename)
        if subject_id is None or subject_id not in keep_subjects:
            continue

        file_path = fmri_dir / filename
        data = np.load(file_path, allow_pickle=True)

        subject_windows = np.concatenate((data["PA"], data["AP"]), axis=0)
        subject_label = int(np.asarray(data["label"]).item())

        X_list.append(subject_windows)
        y_list.append(np.full(subject_windows.shape[0], subject_label, dtype=int))
        sub_ids.extend([subject_id] * subject_windows.shape[0])
        label_stats[subject_label] += 1

    if not X_list:
        raise ValueError(f"No fMRI window files found in {fmri_dir}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    sub_ids_array = np.asarray(sub_ids)

    print("=" * 80)
    print("Loaded fMRI window data")
    print("=" * 80)
    print(f"Directory: {fmri_dir}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Unique subjects: {len(np.unique(sub_ids_array))}")
    print(f"Subject label counts: {dict(label_stats)}")

    return {"X": X, "y": y, "sub_ids": sub_ids_array}


def compute_subject_dwell_times_fmri(
    X_data: np.ndarray,
    y_data: np.ndarray,
    sub_ids_data: np.ndarray,
    target_subjects: np.ndarray,
    kmeans_model: KMeans,
    k_states: int,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    subject_features: list[np.ndarray] = []
    subject_labels: list[int] = []
    processed_subjects: list[str] = []

    for subject in np.unique(target_subjects):
        subject_mask = sub_ids_data == subject
        if not np.any(subject_mask):
            continue

        subject_windows = X_data[subject_mask]
        subject_label = int(y_data[subject_mask][0])
        subject_states = kmeans_model.predict(subject_windows)
        subject_dwell_times = np.bincount(subject_states, minlength=k_states).astype(float) * step_sec

        subject_features.append(subject_dwell_times)
        subject_labels.append(subject_label)
        processed_subjects.append(subject)

    return np.asarray(subject_features), np.asarray(subject_labels), processed_subjects


def choose_statistical_test(groups: list[np.ndarray]) -> tuple[str, float, float]:
    if len(groups) < 2 or any(group.size == 0 for group in groups):
        return "None", 1.0, 0.0

    try:
        normality_ok = all(shapiro(group)[1] > 0.05 for group in groups if len(group) >= 3)
        equal_var_ok = levene(*groups)[1] > 0.05

        if normality_ok and equal_var_ok:
            if len(groups) == 2:
                stat, p_value = ttest_ind(*groups)
                return "t-test", float(p_value), float(stat)
            stat, p_value = f_oneway(*groups)
            return "ANOVA", float(p_value), float(stat)

        if len(groups) == 2:
            stat, p_value = mannwhitneyu(*groups)
            return "Mann-Whitney", float(p_value), float(stat)
        stat, p_value = kruskal(*groups)
        return "Kruskal-Wallis", float(p_value), float(stat)
    except Exception:
        return "None", 1.0, 0.0


def determine_dominant_group(
    subject_features: np.ndarray,
    subject_labels: np.ndarray,
    state: int,
) -> tuple[str, dict[int, float], str]:
    group_means: dict[int, float] = {}

    for label in np.unique(subject_labels):
        group_values = subject_features[subject_labels == label, state]
        group_means[int(label)] = float(np.mean(group_values))

    dominant_label = max(group_means, key=group_means.get)
    dominant_group_safe = GROUP_LABEL_TO_SAFE_NAME[dominant_label]

    means_str = "_".join(
        f"{GROUP_LABEL_TO_SAFE_NAME[label]}{group_means[label]:.1f}"
        for label in sorted(group_means)
    )

    return dominant_group_safe, group_means, means_str


def build_risk_group_dataframe(participants_path: Path, common_subjects: set[str]) -> pd.DataFrame:
    df = pd.read_csv(participants_path, sep="\t")

    conditions = [
        (~df["APOE_haplotype"].str.contains("e4", case=False))
        & (df["PICALM_rs3851179"] != "G/G"),
        (df["APOE_haplotype"].str.contains("e4", case=False))
        & (df["PICALM_rs3851179"] != "G/G"),
        (df["APOE_haplotype"].str.contains("e4", case=False))
        & (df["PICALM_rs3851179"] == "G/G"),
    ]
    choices = ["N", "A+P-", "A+P+"]
    df["Risk_Group"] = np.select(conditions, choices, default="Excluded")

    df.loc[
        (df["APOE_haplotype"].str.contains("e2", case=False))
        & (~df["APOE_haplotype"].str.contains("e4", case=False))
        & (df["Risk_Group"] == "N"),
        "Risk_Group",
    ] = "Excluded"

    study_df = df[df["Risk_Group"].isin(["N", "A+P-", "A+P+"])].copy()
    study_df["Has_EEG_fMRI"] = study_df["participant_id"].apply(
        lambda participant_id: participant_id.replace("sub-", "").zfill(2) in common_subjects
    )
    final_df = study_df[study_df["Has_EEG_fMRI"]].copy()
    final_df["Group"] = final_df["Risk_Group"].map({"N": 0, "A+P-": 1, "A+P+": 2})

    print("=" * 80)
    print("Participant filtering")
    print("=" * 80)
    print(f"participants.tsv path: {participants_path}")
    print(f"Final dataframe shape: {final_df.shape}")
    print("Class distribution:")
    print(final_df["Risk_Group"].value_counts())

    return final_df


def flat_to_matrix_coords_formula(flat_index: int, n_rois: int = 100) -> tuple[int, int]:
    i = n_rois - 2 - int(np.floor(np.sqrt(-8 * flat_index + 4 * n_rois * (n_rois - 1) - 7) / 2.0 - 0.5))
    j = flat_index + i + 1 - n_rois * (n_rois - 1) // 2 + (n_rois - i) * ((n_rois - i) - 1) // 2
    return i, j


def prettify_schaefer_label(label: str) -> str:
    return label.replace("7Networks_", "").replace("_", " ")


def load_schaefer_coordinates(n_rois: int = 100) -> tuple[np.ndarray, list[str]]:
    atlas_paths = tf.get(
        template="MNI152NLin2009cAsym",
        atlas="Schaefer2018",
        desc=f"{n_rois}Parcels7Networks",
        extension=".nii.gz",
    )
    atlas_path = atlas_paths[0] if isinstance(atlas_paths, list) else atlas_paths
    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata()

    coordinates: list[list[float]] = []
    unique_labels = np.unique(atlas_data)[1:]

    for label in unique_labels:
        region_mask = atlas_data == label
        center_of_mass_voxel = ndimage.center_of_mass(region_mask)
        center_of_mass_mni = image.coord_transform(
            center_of_mass_voxel[0],
            center_of_mass_voxel[1],
            center_of_mass_voxel[2],
            atlas_img.affine,
        )
        coordinates.append(
            [
                float(center_of_mass_mni[0]),
                float(center_of_mass_mni[1]),
                float(center_of_mass_mni[2]),
            ]
        )

    # Names from the TemplateFlow label table of the same atlas image. (nilearn's older
    # Schaefer label file has the same parcels and order but 19 outdated region names.)
    label_table = pd.read_csv(
        tf.get(template="MNI152NLin2009cAsym", atlas="Schaefer2018",
               desc=f"{n_rois}Parcels7Networks", suffix="dseg", extension=".tsv"),
        sep="\t",
    )
    atlas_labels = label_table.sort_values("index")["name"].tolist()
    return np.asarray(coordinates), atlas_labels


def aggregate_cluster_files(
    npz_files: list[Path],
    aggregation_within_cluster: str = "median",
) -> np.ndarray:
    if not npz_files:
        raise ValueError("No cluster files were provided for aggregation.")

    cluster_summaries: list[np.ndarray] = []
    for file_path in npz_files:
        data = np.load(file_path, allow_pickle=True)
        windows = data["windows"]

        if aggregation_within_cluster == "median":
            cluster_summary = np.median(windows, axis=0)
        elif aggregation_within_cluster == "mean":
            cluster_summary = np.mean(windows, axis=0)
        else:
            raise ValueError("aggregation_within_cluster must be 'median' or 'mean'")

        cluster_summaries.append(cluster_summary)

    return np.mean(np.asarray(cluster_summaries), axis=0)


def build_top_index_map(
    mean_vector: np.ndarray,
    top_n: int,
) -> tuple[list[int], dict[int, float]]:
    top_indices = np.argsort(mean_vector)[-top_n:][::-1]
    top_map = {int(idx): float(mean_vector[idx]) for idx in top_indices}
    return [int(idx) for idx in top_indices], top_map


def collect_cluster_files_from_dirs(cluster_dirs: list[Path]) -> list[Path]:
    cluster_files: list[Path] = []
    for cluster_dir in cluster_dirs:
        if cluster_dir.exists():
            cluster_files.extend(sorted(cluster_dir.glob("*.npz")))
    return cluster_files


def build_direction_table(
    top_indices: list[int],
    top_map: dict[int, float],
    atlas_labels: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for rank, flat_idx in enumerate(top_indices, start=1):
        roi_i, roi_j = flat_to_matrix_coords_formula(flat_idx, n_rois=100)
        region_1 = prettify_schaefer_label(atlas_labels[roi_i])
        region_2 = prettify_schaefer_label(atlas_labels[roi_j])
        rows.append(
            {
                "rank": rank,
                "roi_pair": f"({roi_i},{roi_j})",
                "region_names": f"{region_1} <-> {region_2}",
                "correlation_value": float(top_map[flat_idx]),
                "flat_index": flat_idx,
                "roi_i": roi_i,
                "roi_j": roi_j,
            }
        )
    return pd.DataFrame(rows)


def build_adjacency_matrix_from_top_map(
    top_indices: list[int],
    top_map: dict[int, float],
    n_rois: int = 100,
) -> np.ndarray:
    adjacency_matrix = np.zeros((n_rois, n_rois), dtype=float)
    for flat_idx in top_indices:
        roi_i, roi_j = flat_to_matrix_coords_formula(flat_idx, n_rois=n_rois)
        adjacency_matrix[roi_i, roi_j] = float(top_map[flat_idx])
        adjacency_matrix[roi_j, roi_i] = float(top_map[flat_idx])
    return adjacency_matrix


def save_direction_connectome_html(
    top_indices: list[int],
    top_map: dict[int, float],
    output_html: Path,
    edge_cmap: str,
    node_color: str = "gray",
    node_size: int = 5,
) -> None:
    coordinates, _ = load_schaefer_coordinates(n_rois=100)
    adjacency_matrix = build_adjacency_matrix_from_top_map(
        top_indices=top_indices,
        top_map=top_map,
        n_rois=100,
    )
    interactive_view = view_connectome(
        adjacency_matrix,
        coordinates,
        node_size=node_size,
        node_color=node_color,
        edge_cmap=edge_cmap,
        colorbar=False,
    )
    interactive_view.save_as_html(str(output_html))


def save_direction_outputs(
    direction_name: str,
    top_indices: list[int],
    top_map: dict[int, float],
    atlas_labels: list[str],
    output_csv: Path,
    output_html: Path,
    edge_cmap: str,
) -> pd.DataFrame:
    direction_df = build_direction_table(
        top_indices=top_indices,
        top_map=top_map,
        atlas_labels=atlas_labels,
    )
    direction_df.to_csv(output_csv, index=False)
    save_direction_connectome_html(
        top_indices=top_indices,
        top_map=top_map,
        output_html=output_html,
        edge_cmap=edge_cmap,
    )

    printable_df = direction_df.copy()
    printable_df["correlation_value"] = printable_df["correlation_value"].map(lambda value: f"{value:.4f}")

    print("=" * 80)
    print(f"{direction_name} TOP FUNCTIONAL CONNECTIVITY TABLE")
    print("=" * 80)
    print(f"Saved CSV: {output_csv}")
    print(f"Saved HTML: {output_html}")
    print(
        printable_df[
            ["rank", "roi_pair", "region_names", "correlation_value"]
        ].to_string(index=False)
    )

    return direction_df


def build_unified_direction_table(
    risk_cluster_dirs: list[Path],
    n_cluster_dirs: list[Path],
    output_csv: Path,
    top_n: int = 10,
    aggregation_within_cluster: str = "median",
    save_direction_visuals: bool = False,
) -> pd.DataFrame:
    risk_files = collect_cluster_files_from_dirs(risk_cluster_dirs)
    n_files = collect_cluster_files_from_dirs(n_cluster_dirs)

    if not risk_files:
        raise FileNotFoundError("No risk-higher cluster files found.")
    if not n_files:
        raise FileNotFoundError("No N-higher cluster files found.")

    risk_mean = aggregate_cluster_files(
        risk_files,
        aggregation_within_cluster=aggregation_within_cluster,
    )
    n_mean = aggregate_cluster_files(
        n_files,
        aggregation_within_cluster=aggregation_within_cluster,
    )

    risk_top_indices, risk_top_map = build_top_index_map(risk_mean, top_n=top_n)
    n_top_indices, n_top_map = build_top_index_map(n_mean, top_n=top_n)

    unified_indices = risk_top_indices + [idx for idx in n_top_indices if idx not in set(risk_top_indices)]

    _, atlas_labels = load_schaefer_coordinates(n_rois=100)

    if save_direction_visuals:
        save_direction_outputs(
            direction_name="RISK > N DWELL-TIME",
            top_indices=risk_top_indices,
            top_map=risk_top_map,
            atlas_labels=atlas_labels,
            output_csv=RISK_DIRECTION_TOP_PAIRS_CSV,
            output_html=RISK_DIRECTION_CONNECTOME_HTML,
            edge_cmap="Blues",
        )
        save_direction_outputs(
            direction_name="N > OTHERS DWELL-TIME",
            top_indices=n_top_indices,
            top_map=n_top_map,
            atlas_labels=atlas_labels,
            output_csv=N_DIRECTION_TOP_PAIRS_CSV,
            output_html=N_DIRECTION_CONNECTOME_HTML,
            edge_cmap="Blues",
        )

    rows: list[dict[str, object]] = []
    for rank, flat_idx in enumerate(unified_indices, start=1):
        roi_i, roi_j = flat_to_matrix_coords_formula(flat_idx, n_rois=100)
        region_1 = prettify_schaefer_label(atlas_labels[roi_i])
        region_2 = prettify_schaefer_label(atlas_labels[roi_j])

        rows.append(
            {
                "rank": rank,
                "roi_pair": f"({roi_i},{roi_j})",
                "region_names": f"{region_1} <-> {region_2}",
                "r_risk_higher_N_lt": risk_top_map.get(flat_idx, np.nan),
                "r_N_higher_N_gt": n_top_map.get(flat_idx, np.nan),
                "flat_index": flat_idx,
                "roi_i": roi_i,
                "roi_j": roi_j,
                "risk_present": flat_idx in risk_top_map,
                "n_present": flat_idx in n_top_map,
            }
        )

    unified_df = pd.DataFrame(rows)
    unified_df.to_csv(output_csv, index=False)

    printable_df = unified_df.copy()
    printable_df["r_risk_higher_N_lt"] = printable_df["r_risk_higher_N_lt"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.4f}"
    )
    printable_df["r_N_higher_N_gt"] = printable_df["r_N_higher_N_gt"].map(
        lambda value: "—" if pd.isna(value) else f"{value:.4f}"
    )

    print("=" * 80)
    print("UNIFIED TOP FUNCTIONAL CONNECTIVITY TABLE")
    print("=" * 80)
    print(f"Risk-higher files used: {len(risk_files)}")
    print(f"N-higher files used: {len(n_files)}")
    print(f"Saved CSV: {output_csv}")
    print(
        printable_df[
            ["rank", "roi_pair", "region_names", "r_risk_higher_N_lt", "r_N_higher_N_gt"]
        ].to_string(index=False)
    )

    return unified_df


def summarize_top_connectivity_pairs(
    cluster_dir: Path,
    aggregation_within_cluster: str = "median",
    top_n: int = 10,
) -> pd.DataFrame:
    npz_files = sorted(cluster_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {cluster_dir}")

    mean_windows: list[np.ndarray] = []
    for file_path in npz_files:
        data = np.load(file_path, allow_pickle=True)
        windows = data["windows"]

        if aggregation_within_cluster == "median":
            cluster_summary = np.median(windows, axis=0)
        elif aggregation_within_cluster == "mean":
            cluster_summary = np.mean(windows, axis=0)
        else:
            raise ValueError("aggregation_within_cluster must be 'median' or 'mean'")

        mean_windows.append(cluster_summary)

    mean_windows_array = np.asarray(mean_windows)
    all_mean = np.mean(mean_windows_array, axis=0)

    _, atlas_labels = load_schaefer_coordinates(n_rois=100)

    top_indices = np.argsort(all_mean)[-top_n:][::-1]

    rows: list[dict[str, object]] = []
    for rank, flat_idx in enumerate(top_indices, start=1):
        roi_i, roi_j = flat_to_matrix_coords_formula(int(flat_idx), n_rois=100)
        rows.append(
            {
                "rank": rank,
                "flat_index": int(flat_idx),
                "roi_i": int(roi_i),
                "roi_j": int(roi_j),
                "region_1": atlas_labels[roi_i],
                "region_2": atlas_labels[roi_j],
                "mean_connectivity": float(all_mean[flat_idx]),
            }
        )

    results_df = pd.DataFrame(rows)

    print("=" * 80)
    print(f"Top {top_n} FC pairs from {cluster_dir}")
    print("=" * 80)
    print(f"Total cluster files: {len(npz_files)}")
    print(f"Flattened feature length: {len(all_mean)}")
    print(f"Expected upper-triangle length for 100 ROIs: {100 * 99 // 2}")
    print(results_df.to_string(index=False))

    return results_df


def make_grouped_output_dirs(base_dir: Path) -> dict[str, Path]:
    grouped_output_dirs = {
        "N": base_dir / "N_group_higher",
        "AposPminus": base_dir / "AposPminus_group_higher",
        "AposPpos": base_dir / "AposPpos_group_higher",
    }
    for path in grouped_output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return grouped_output_dirs


def create_seed_feature_dataframe(
    final_df_template: pd.DataFrame,
    participant_to_row_index: dict[str, int],
    significant_clusters: list[dict[str, object]],
    output_csv: Path,
) -> pd.DataFrame:
    seed_df = final_df_template.copy()

    for cluster in significant_clusters:
        k_value = cluster["K"]
        state = cluster["state"]
        dominant_group = cluster["dominant_group"]
        p_value = cluster["p_value"]

        col_name = f"fMRI_K{k_value}_state{state}_{dominant_group}"
        seed_df[col_name] = np.nan
        seed_df[f"{col_name}_pval"] = p_value
        seed_df[f"{col_name}_test"] = cluster["test"]

        for subject_idx, subject_id in enumerate(cluster["processed_subjects"]):
            participant_id = f"sub-{subject_id}"
            row_index = participant_to_row_index.get(participant_id)
            if row_index is None:
                continue
            seed_df.loc[row_index, col_name] = cluster["subject_features"][subject_idx]

    seed_df.to_csv(output_csv, index=False)
    return seed_df


def bootstrap_mean_difference_ci(
    group_1: np.ndarray,
    group_2: np.ndarray,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(random_seed)
    group_1 = np.asarray(group_1, dtype=float)
    group_2 = np.asarray(group_2, dtype=float)

    boot_diffs = np.empty(n_bootstrap, dtype=float)
    for idx in range(n_bootstrap):
        sample_1 = rng.choice(group_1, size=len(group_1), replace=True)
        sample_2 = rng.choice(group_2, size=len(group_2), replace=True)
        boot_diffs[idx] = float(np.mean(sample_1) - np.mean(sample_2))

    alpha = 1.0 - confidence_level
    lower = float(np.percentile(boot_diffs, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_diffs, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def p_value_marker(p_value: float) -> str:
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "dagger"
    return ""


def select_representative_clusters_for_posthoc(
    significant_clusters: list[dict[str, object]],
    target_k_values: list[int],
) -> list[dict[str, object]]:
    selected_clusters: list[dict[str, object]] = []

    for k_value in target_k_values:
        candidates = [cluster for cluster in significant_clusters if cluster["K"] == k_value]
        if not candidates:
            continue

        selected_cluster = min(
            candidates,
            key=lambda cluster: (cluster["p_value"], cluster["state"]),
        )
        selected_clusters.append(selected_cluster)

    return selected_clusters


def build_seed_posthoc_table(
    significant_clusters: list[dict[str, object]],
    seed: int,
    output_csv: Path,
    target_k_values: list[int],
    bootstrap_samples: int = 5000,
) -> pd.DataFrame:
    selected_clusters = select_representative_clusters_for_posthoc(
        significant_clusters=significant_clusters,
        target_k_values=target_k_values,
    )

    if not selected_clusters:
        raise FileNotFoundError(
            f"No significant clusters found for requested k values {target_k_values} in seed {seed}."
        )

    pair_order = [(0, 1), (0, 2), (1, 2)]
    rows: list[dict[str, object]] = []

    for cluster in selected_clusters:
        raw_p_values: list[float] = []
        cluster_rows: list[dict[str, object]] = []

        subject_features = np.asarray(cluster["subject_features"], dtype=float)
        subject_labels = np.asarray(cluster["subject_labels"], dtype=int)

        print(
            f"Post hoc selection for seed={seed}: "
            f"k={cluster['K']}, state={cluster['state']}, p={cluster['p_value']:.6f}, "
            f"dominant_group={SAFE_NAME_TO_READABLE[cluster['dominant_group']]}"
        )

        for pair_idx, (group_1_label, group_2_label) in enumerate(pair_order):
            group_1_values = subject_features[subject_labels == group_1_label]
            group_2_values = subject_features[subject_labels == group_2_label]

            test_name, p_raw, _ = choose_statistical_test([group_1_values, group_2_values])
            mean_diff = float(np.mean(group_1_values) - np.mean(group_2_values))
            ci_lower, ci_upper = bootstrap_mean_difference_ci(
                group_1=group_1_values,
                group_2=group_2_values,
                n_bootstrap=bootstrap_samples,
                confidence_level=0.95,
                random_seed=seed * 10_000 + cluster["K"] * 100 + cluster["state"] * 10 + pair_idx,
            )

            raw_p_values.append(p_raw)
            cluster_rows.append(
                {
                    "seed": seed,
                    "K": cluster["K"],
                    "state": cluster["state"],
                    "dominant_group": SAFE_NAME_TO_READABLE[cluster["dominant_group"]],
                    "group_1": GROUP_LABEL_TO_NAME[group_1_label],
                    "group_2": GROUP_LABEL_TO_NAME[group_2_label],
                    "difference": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "p_raw": p_raw,
                    "test": test_name,
                }
            )

        adjusted_p_values = np.minimum(np.asarray(raw_p_values) * len(raw_p_values), 1.0)
        for row, adjusted_p_value in zip(cluster_rows, adjusted_p_values):
            row["p_bonf"] = float(adjusted_p_value)
            row["significance"] = p_value_marker(float(adjusted_p_value))
            rows.append(row)

    posthoc_df = pd.DataFrame(rows)
    posthoc_df.to_csv(output_csv, index=False)

    printable_df = posthoc_df.copy()
    for col_name in ["difference", "ci_lower", "ci_upper", "p_raw", "p_bonf"]:
        printable_df[col_name] = printable_df[col_name].map(lambda value: f"{value:.4f}")

    print("=" * 80)
    print("SEED-SPECIFIC POST HOC TABLE")
    print("=" * 80)
    print(f"Saved CSV: {output_csv}")
    print(
        printable_df[
            [
                "seed",
                "K",
                "state",
                "group_1",
                "group_2",
                "difference",
                "ci_lower",
                "ci_upper",
                "p_bonf",
                "significance",
            ]
        ].to_string(index=False)
    )

    return posthoc_df


def build_significant_cluster_summary(
    significant_clusters: list[dict[str, object]],
    output_csv: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster in significant_clusters:
        group_means = cluster["group_means"]
        rows.append(
            {
                "seed": cluster["seed"],
                "K": cluster["K"],
                "state": cluster["state"],
                "test": cluster["test"],
                "p_value": cluster["p_value"],
                "stat_value": cluster["stat_value"],
                "dominant_group": SAFE_NAME_TO_READABLE[cluster["dominant_group"]],
                "n_windows": cluster["n_windows"],
                "group_mean_N": group_means.get(0, np.nan),
                "group_mean_AposPminus": group_means.get(1, np.nan),
                "group_mean_AposPpos": group_means.get(2, np.nan),
                "means_str": cluster["means_str"],
            }
        )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(output_csv, index=False)
    return summary_df


def run_single_seed_analysis(
    seed: int,
    fmri_data: dict[str, np.ndarray],
    final_df_template: pd.DataFrame,
    participant_to_row_index: dict[str, int],
) -> dict[str, object]:
    print()
    print("=" * 80)
    print(f"RUNNING SEED = {seed}")
    print("=" * 80)

    significant_clusters: list[dict[str, object]] = []
    unique_subjects = np.unique(fmri_data["sub_ids"])

    for k_states in K_STATES_LIST:
        print()
        print(f"Testing K_STATES = {k_states} | seed = {seed}")
        print("=" * 60)

        kmeans = KMeans(
            n_clusters=k_states,
            random_state=seed,
            n_init="auto",
            max_iter=300,
        )
        kmeans.fit(fmri_data["X"])

        all_cluster_predictions = kmeans.predict(fmri_data["X"])

        subject_features, subject_labels, processed_subjects = compute_subject_dwell_times_fmri(
            X_data=fmri_data["X"],
            y_data=fmri_data["y"],
            sub_ids_data=fmri_data["sub_ids"],
            target_subjects=unique_subjects,
            kmeans_model=kmeans,
            k_states=k_states,
            step_sec=FMRI_STEP_SEC,
        )

        for state in range(k_states):
            groups = [
                subject_features[subject_labels == label, state]
                for label in np.unique(subject_labels)
            ]
            test_name, p_value, stat_value = choose_statistical_test(groups)
            is_significant = p_value < P_VALUE_THRESHOLD
            marker = "***" if is_significant else ""

            print(f"  State {state}: {test_name}, p={p_value:.6f} {marker}")

            if not is_significant:
                continue

            dominant_group, group_means, means_str = determine_dominant_group(
                subject_features=subject_features,
                subject_labels=subject_labels,
                state=state,
            )

            cluster_mask = all_cluster_predictions == state
            cluster_windows = fmri_data["X"][cluster_mask]

            cluster_info = {
                "seed": seed,
                "K": k_states,
                "state": state,
                "p_value": p_value,
                "stat_value": stat_value,
                "test": test_name,
                "dominant_group": dominant_group,
                "group_means": group_means,
                "means_str": means_str,
                "windows": cluster_windows,
                "n_windows": int(len(cluster_windows)),
                "subject_features": subject_features[:, state].copy(),
                "subject_labels": subject_labels.copy(),
                "processed_subjects": processed_subjects.copy(),
            }
            significant_clusters.append(cluster_info)

            readable_group = SAFE_NAME_TO_READABLE[dominant_group]
            print(f"    SIGNIFICANT: dominant group = {readable_group}")
            print(f"    Group means: {group_means}")
            print(f"    Windows: {len(cluster_windows)}")

    print()
    print("=" * 80)
    print(f"FOUND {len(significant_clusters)} SIGNIFICANT CLUSTERS FOR SEED {seed}")
    print("=" * 80)

    seed_output_root = OUTPUT_ROOT / f"seed_{seed}"
    grouped_output_dirs = make_grouped_output_dirs(seed_output_root)

    print()
    print("=" * 80)
    print(f"SAVING SIGNIFICANT CLUSTERS FOR SEED {seed}")
    print("=" * 80)

    for index, cluster in enumerate(significant_clusters, start=1):
        group_dir = grouped_output_dirs[cluster["dominant_group"]]
        filename = (
            f"cluster_seed{seed}_K{cluster['K']}_state{cluster['state']}_"
            f"{cluster['dominant_group']}_dominant_"
            f"p{cluster['p_value']:.6f}.npz"
        )
        file_path = group_dir / filename

        np.savez_compressed(
            file_path,
            windows=cluster["windows"],
            seed=cluster["seed"],
            K=cluster["K"],
            state=cluster["state"],
            p_value=cluster["p_value"],
            stat_value=cluster["stat_value"],
            test=cluster["test"],
            dominant_group=cluster["dominant_group"],
            group_means=cluster["group_means"],
            means_str=cluster["means_str"],
            n_windows=cluster["n_windows"],
            window_size_sec=FMRI_WINDOW_SEC,
            overlap=FMRI_OVERLAP,
            step_sec=FMRI_STEP_SEC,
        )

        print(f"{index:2d}. {file_path}")
        print(f"    Windows: {cluster['n_windows']}, p={cluster['p_value']:.6f}")

    seed_output_csv = seed_output_root / f"fMRI_all_significant_features_seed{seed}_60s_75overlap.csv"
    seed_df = create_seed_feature_dataframe(
        final_df_template=final_df_template,
        participant_to_row_index=participant_to_row_index,
        significant_clusters=significant_clusters,
        output_csv=seed_output_csv,
    )

    print()
    print("=" * 80)
    print(f"SUMMARY REPORT FOR SEED {seed}")
    print("=" * 80)
    print(f"Saved dataframe: {seed_output_csv}")
    print(f"Shape: {seed_df.shape}")
    print(f"Significant feature columns: {len(significant_clusters)}")

    dominant_counts = Counter(cluster["dominant_group"] for cluster in significant_clusters)
    print("\nClusters by dominant group:")
    for group_name, count in dominant_counts.items():
        print(f"  {SAFE_NAME_TO_READABLE[group_name]}: {count}")

    k_counts = Counter(cluster["K"] for cluster in significant_clusters)
    print("\nClusters by K value:")
    for k_value, count in sorted(k_counts.items()):
        print(f"  K={k_value}: {count}")

    if significant_clusters:
        p_values = [cluster["p_value"] for cluster in significant_clusters]
        print("\nP-value range:")
        print(f"  Min: {min(p_values):.6f}")
        print(f"  Max: {max(p_values):.6f}")
        print(f"  Mean: {np.mean(p_values):.6f}")

    for safe_group_name, group_dir in grouped_output_dirs.items():
        cluster_files = list(group_dir.glob("*.npz"))
        if not cluster_files:
            print()
            print(f"No significant clusters found for {SAFE_NAME_TO_READABLE[safe_group_name]}.")
            continue

        print()
        print(f"POST-HOC CONNECTIVITY SUMMARY FOR {SAFE_NAME_TO_READABLE[safe_group_name]} | seed = {seed}")
        summarize_top_connectivity_pairs(
            cluster_dir=group_dir,
            aggregation_within_cluster="median",
            top_n=10,
        )

    risk_higher_dirs = [
        grouped_output_dirs["AposPminus"],
        grouped_output_dirs["AposPpos"],
    ]
    n_higher_dirs = [grouped_output_dirs["N"]]
    seed_unified_csv = seed_output_root / f"fMRI_unified_top_fc_pairs_seed{seed}_60s_75overlap.csv"

    try:
        print()
        build_unified_direction_table(
            risk_cluster_dirs=risk_higher_dirs,
            n_cluster_dirs=n_higher_dirs,
            output_csv=seed_unified_csv,
            top_n=10,
            aggregation_within_cluster="median",
        )
    except FileNotFoundError as exc:
        print()
        print(f"Skipping unified direction table for seed {seed}: {exc}")

    if seed == POSTHOC_SEED:
        seed_posthoc_csv = seed_output_root / f"fMRI_posthoc_seed{seed}_k{'_'.join(map(str, POSTHOC_K_VALUES))}.csv"
        try:
            print()
            build_seed_posthoc_table(
                significant_clusters=significant_clusters,
                seed=seed,
                output_csv=seed_posthoc_csv,
                target_k_values=POSTHOC_K_VALUES,
                bootstrap_samples=POSTHOC_BOOTSTRAP_SAMPLES,
            )
        except FileNotFoundError as exc:
            print()
            print(f"Skipping seed-specific post hoc table for seed {seed}: {exc}")

    return {
        "seed": seed,
        "significant_clusters": significant_clusters,
        "grouped_output_dirs": grouped_output_dirs,
        "seed_output_root": seed_output_root,
        "seed_output_csv": seed_output_csv,
    }


# ============================================================================
# Main workflow
# ============================================================================

def main() -> None:
    fmri_dir = resolve_existing_path(FMRI_DIR_CANDIDATES, "60s / 75% overlap fMRI directory")
    eeg_dir = resolve_existing_path(EEG_DIR_CANDIDATES, "EEG directory")
    participants_path = Path(PARTICIPANTS_TSV)

    if not participants_path.exists():
        raise FileNotFoundError(f"Participants file not found: {participants_path}")

    fmri_subjects = collect_subject_ids_from_directory(fmri_dir, suffix=".npz")
    eeg_subjects = collect_subject_ids_from_directory(eeg_dir, suffix=".npz")
    sample_subjects = {pid.replace("sub-", "").zfill(2) for pid in pd.read_csv(SAMPLE_CSV)["participant_id"]}
    common_subjects = eeg_subjects.intersection(fmri_subjects) & sample_subjects

    print("=" * 80)
    print("Data sources")
    print("=" * 80)
    print(f"fMRI dir: {fmri_dir}")
    print(f"EEG dir: {eeg_dir}")
    print(f"Number of subjects with both EEG and fMRI data: {len(common_subjects)}")

    fmri_data = load_fmri_window_data(fmri_dir, common_subjects)
    final_df = build_risk_group_dataframe(participants_path, common_subjects)
    participant_to_row_index = dict(zip(final_df["participant_id"], final_df.index))

    print()
    print("=" * 80)
    print("MULTI-SEED ANALYSIS")
    print("=" * 80)
    print(f"Seeds: {SEED_LIST}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    seed_results: list[dict[str, object]] = []
    all_significant_clusters: list[dict[str, object]] = []

    for seed in SEED_LIST:
        seed_result = run_single_seed_analysis(
            seed=seed,
            fmri_data=fmri_data,
            final_df_template=final_df,
            participant_to_row_index=participant_to_row_index,
        )
        seed_results.append(seed_result)
        all_significant_clusters.extend(seed_result["significant_clusters"])

    print()
    print("=" * 80)
    print("COMBINED MULTI-SEED SUMMARY")
    print("=" * 80)
    print(f"Seeds completed: {len(seed_results)}")
    print(f"Total significant clusters across all seeds: {len(all_significant_clusters)}")

    if all_significant_clusters:
        summary_df = build_significant_cluster_summary(
            significant_clusters=all_significant_clusters,
            output_csv=SIGNIFICANT_CLUSTER_SUMMARY_CSV,
        )
        print(f"Saved cluster summary CSV: {SIGNIFICANT_CLUSTER_SUMMARY_CSV}")
        print(f"Summary rows: {len(summary_df)}")

        seed_counts = Counter(cluster["seed"] for cluster in all_significant_clusters)
        print("\nClusters by seed:")
        for seed, count in sorted(seed_counts.items()):
            print(f"  seed={seed}: {count}")

        dominant_counts = Counter(cluster["dominant_group"] for cluster in all_significant_clusters)
        print("\nClusters by dominant group:")
        for group_name, count in dominant_counts.items():
            print(f"  {SAFE_NAME_TO_READABLE[group_name]}: {count}")

        k_counts = Counter(cluster["K"] for cluster in all_significant_clusters)
        print("\nClusters by K value:")
        for k_value, count in sorted(k_counts.items()):
            print(f"  K={k_value}: {count}")

        p_values = [cluster["p_value"] for cluster in all_significant_clusters]
        print("\nP-value range:")
        print(f"  Min: {min(p_values):.6f}")
        print(f"  Max: {max(p_values):.6f}")
        print(f"  Mean: {np.mean(p_values):.6f}")

        risk_higher_dirs: list[Path] = []
        n_higher_dirs: list[Path] = []
        for seed_result in seed_results:
            grouped_output_dirs = seed_result["grouped_output_dirs"]
            risk_higher_dirs.extend(
                [
                    grouped_output_dirs["AposPminus"],
                    grouped_output_dirs["AposPpos"],
                ]
            )
            n_higher_dirs.append(grouped_output_dirs["N"])

        print()
        build_unified_direction_table(
            risk_cluster_dirs=risk_higher_dirs,
            n_cluster_dirs=n_higher_dirs,
            output_csv=COMBINED_UNIFIED_TOP_PAIRS_CSV,
            top_n=10,
            aggregation_within_cluster="median",
            save_direction_visuals=True,
        )
    else:
        print("No significant clusters were found across any seed.")

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
