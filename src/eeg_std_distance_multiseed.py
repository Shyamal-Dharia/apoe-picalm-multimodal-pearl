from __future__ import annotations

import glob
import math
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy import stats
from scipy.stats import f_oneway, kruskal, levene, mannwhitneyu, shapiro, ttest_ind
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.multitest import multipletests


# ============================================================================
# Configuration
# ============================================================================

SEED_LIST = [9, 42, 123, 2021, 2022]
TABLE_SEED = 9
K_STATES_LIST = list(range(5, 11))
P_VALUE_THRESHOLD = 0.05
TOP_CONSISTENT_REGION_BANDS = 10
MIN_CONSISTENT_REGION_BAND_SUPPORT = 2

EEG_DATA_DIR_CANDIDATES = [
    Path("data/EEG_data/HFD_PSD_4sec_75overlap_6min"),
    Path("../data/EEG_data/HFD_PSD_4sec_75overlap_6min"),
]
MRI_CSV_CANDIDATES = [
    Path("data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv"),
    Path("../data/MRI_data/ROI_aal3_Vgm.csv_with_groups.csv"),
]
RAW_VHDR_CANDIDATES = [
    Path("data/EEG_data/one_subject_raw/sub-01_task-rest_eeg.vhdr"),
    Path("../data/EEG_data/one_subject_raw/sub-01_task-rest_eeg.vhdr"),
]

SAMPLE_CSV = Path("data/participants_n64.csv")  # analysis sample, from src/select_participants.py
# EEG_RESULTS_DIR redirects all outputs (e.g. for a sensitivity run); default results/eeg
RESULTS_DIR = Path(os.environ.get("EEG_RESULTS_DIR", "results/eeg"))
# A participant's value for a state is the SD over their windows in that state; with fewer than
# MIN_WINDOWS windows the value is missing. With one window the SD is 0, which is not a measured
# variability, so the paper uses 2 (as src/f1_permutation_pipeline.py); 1 reproduces the run
# before 2026-10-10.
MIN_WINDOWS = int(os.environ.get("EEG_MIN_WINDOWS", "2"))
OUTPUT_ROOT = RESULTS_DIR / "eeg_std_distance_multiseed"
EEG_DIRECTION_N_HIGHER_CSV = RESULTS_DIR / "eeg_subject_mean_variability_N_higher_multiseed.csv"
EEG_DIRECTION_APM_HIGHER_CSV = RESULTS_DIR / "eeg_subject_mean_variability_A_plus_P_minus_higher_multiseed.csv"
EEG_DIRECTION_APP_HIGHER_CSV = RESULTS_DIR / "eeg_subject_mean_variability_A_plus_P_plus_higher_multiseed.csv"
EEG_DIRECTION_RISK_HIGHER_CSV = RESULTS_DIR / "eeg_subject_mean_variability_risk_higher_multiseed.csv"
EEG_DIRECTION_COMBINED_CSV = RESULTS_DIR / "eeg_subject_mean_variability_by_direction_multiseed.csv"
DIRECTION_SPLIT_FIGURE_FILENAME = "regional_variability_direction_split_consistent_multiseed.png"

GROUP_TO_LABEL = {"N": 0, "A+P-": 1, "A+P+": 2}
LABEL_TO_GROUP = {value: key for key, value in GROUP_TO_LABEL.items()}
DIRECTION_GROUPS = list(GROUP_TO_LABEL.keys())
GROUP_TO_SAFE_NAME = {
    "N": "N",
    "A+P-": "A_plus_P_minus",
    "A+P+": "A_plus_P_plus",
}
GROUP_TO_DIRECTION_LABEL = {
    "N": "N higher",
    "A+P-": "A+P- higher",
    "A+P+": "A+P+ higher",
}

REGION_COLORS = {
    "Frontal": "#99CCFF",
    "Temporal": "#FFBBDD",
    "Central": "#99EEAA",
    "Parietal": "#EADDAA",
    "Occipital": "#D8BFD8",
}
SUPPORT_GROUP_COLORS = {
    "N": "#4C78A8",
    "A+P-": "#F58518",
    "A+P+": "#E45756",
}

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


# ============================================================================
# General helpers
# ============================================================================


def resolve_existing_path(path_candidates: list[Path], label: str) -> Path:
    for path in path_candidates:
        if path.exists():
            return path
    tried = "\n".join(str(p) for p in path_candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")


def file_to_sid(path_or_name: str, pad: int = 2) -> str:
    base = os.path.splitext(os.path.basename(path_or_name))[0]
    digits = "".join(ch for ch in base if ch.isdigit())
    if not digits:
        if base.startswith("sub-"):
            tail = "".join(ch for ch in base.split("-", 1)[1] if ch.isdigit())
            digits = tail or "0"
        else:
            digits = "0"
    return f"sub-{str(int(digits)).zfill(pad)}"


def get_column_case_insensitive(df: pd.DataFrame, target_name_lower: str) -> str:
    for column in df.columns:
        if column.lower() == target_name_lower:
            return column
    raise KeyError(f"Column '{target_name_lower}' not found. Available: {list(df.columns)}")


def choose_statistical_test(groups: list[np.ndarray]) -> tuple[str, float, float]:
    if len(groups) < 2 or any(group.size == 0 for group in groups):
        return "None", 1.0, 0.0

    try:
        normality_ok = all(shapiro(group)[1] > 0.05 for group in groups if len(group) >= 3)
        equal_var_ok = levene(*groups)[1] > 0.05

        if normality_ok and equal_var_ok:
            if len(groups) == 2:
                stat_value, p_value = ttest_ind(*groups)
                return "t-test", float(p_value), float(stat_value)
            stat_value, p_value = f_oneway(*groups)
            return "ANOVA", float(p_value), float(stat_value)

        if len(groups) == 2:
            stat_value, p_value = mannwhitneyu(*groups)
            return "Mann-Whitney", float(p_value), float(stat_value)
        stat_value, p_value = kruskal(*groups)
        return "Kruskal-Wallis", float(p_value), float(stat_value)
    except Exception:
        return "None", 1.0, 0.0


def determine_dominant_group(feature_values: np.ndarray, labels: np.ndarray) -> tuple[str, dict[int, float]]:
    group_means: dict[int, float] = {}
    for label in np.unique(labels):
        values = feature_values[labels == label]
        group_means[int(label)] = float(np.mean(values)) if len(values) else float("nan")
    dominant_label = max(group_means, key=group_means.get)
    return LABEL_TO_GROUP[dominant_label], group_means


def significance_marker(p_value: float) -> str:
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "dagger"
    return ""


def group_to_safe_name(group_name: str) -> str:
    return GROUP_TO_SAFE_NAME[group_name]


# ============================================================================
# EEG state feature analysis
# ============================================================================


def load_subject_labels(mri_csv_path: Path, eeg_data_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    eeg_subjects = {file_to_sid(file_name) for file_name in os.listdir(eeg_data_dir) if file_name.endswith(".npz")}
    eeg_subjects &= set(pd.read_csv(SAMPLE_CSV)["participant_id"])

    mri_df = pd.read_csv(mri_csv_path)
    names_col = get_column_case_insensitive(mri_df, "names")
    group_col = get_column_case_insensitive(mri_df, "group")

    name_series = mri_df[names_col].astype(str)
    subject_raw = name_series.str.extract(r"(sub-\d+)", expand=False).fillna(
        name_series.str.replace("_T1w", "", regex=False)
    )
    subject_num = subject_raw.str.extract(r"(\d+)", expand=False)
    mri_df["subject_id"] = subject_num.apply(
        lambda digits: f"sub-{str(int(digits)).zfill(2)}"
        if pd.notna(digits) and str(digits).isdigit()
        else np.nan
    )
    mri_df["numeric_label"] = mri_df[group_col].map(GROUP_TO_LABEL)

    mri_df_eeg = mri_df[mri_df["subject_id"].isin(eeg_subjects)].dropna(
        subset=["subject_id", "numeric_label"]
    ).copy()
    participant_df = mri_df_eeg[["subject_id", group_col, "numeric_label"]].rename(
        columns={group_col: "Risk_Group", "subject_id": "participant_id", "numeric_label": "Group"}
    ).reset_index(drop=True)

    risk_group_map = dict(zip(mri_df_eeg["subject_id"], mri_df_eeg["numeric_label"]))
    return participant_df, risk_group_map


def load_eeg_features(feature_dir: Path, risk_group_map: dict[str, int]) -> dict[str, np.ndarray]:
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    all_sub_ids: list[int] = []

    print("\nLoading EEG window features...")
    for npz_path in sorted(glob.glob(str(feature_dir / "*.npz"))):
        sid = file_to_sid(npz_path, pad=2)
        if sid not in risk_group_map:
            continue

        label = int(risk_group_map[sid])
        data = np.load(npz_path, allow_pickle=True)
        hfd = data["HFD_features"]
        n_windows = hfd.shape[0]
        subject_num = int(sid.split("-")[1])

        for window_idx in range(n_windows):
            all_features.append(hfd[window_idx].flatten())
            all_labels.append(label)
            all_sub_ids.append(subject_num)

        print(f"  Loaded {n_windows:4d} windows from {sid} ({LABEL_TO_GROUP[label]})")

    return {
        "X": np.asarray(all_features),
        "y": np.asarray(all_labels),
        "sub_ids": np.asarray(all_sub_ids),
    }


def compute_std_distance_to_centroid_only(
    X_data: np.ndarray,
    y_data: np.ndarray,
    sub_ids_data: np.ndarray,
    kmeans_model: KMeans,
    k_states: int,
) -> list[dict[str, object]]:
    centroids = kmeans_model.cluster_centers_
    all_preds = kmeans_model.predict(X_data)

    subject_features: list[dict[str, object]] = []
    for subject in np.unique(sub_ids_data):
        mask = sub_ids_data == subject
        if not np.any(mask):
            continue

        subject_windows = X_data[mask]
        subject_preds = all_preds[mask]
        subject_label = int(y_data[mask][0])

        std_dist = np.full(k_states, np.nan)
        for state in range(k_states):
            state_windows = subject_windows[subject_preds == state]
            if state_windows.shape[0] < max(MIN_WINDOWS, 1):
                continue
            distances = np.linalg.norm(state_windows - centroids[state], axis=1)
            std_dist[state] = np.std(distances)

        subject_features.append(
            {
                "subject_id": int(subject),
                "label": subject_label,
                "std_distance_to_centroid": std_dist,
            }
        )

    return subject_features


def perform_posthoc_comparisons(
    corrected_significant_features: list[dict[str, object]],
    labels_map: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    comparisons = [
        (0, 1, "N", "A+P-"),
        (0, 2, "N", "A+P+"),
        (1, 2, "A+P-", "A+P+"),
    ]

    for feature in corrected_significant_features:
        values = np.asarray(feature["feature_values"])
        subject_ids = np.asarray(feature["subject_ids"])
        labels_array = np.asarray([labels_map.get(f"sub-{int(sid):02d}", -1) for sid in subject_ids])

        valid_mask = ~np.isnan(values) & (labels_array != -1)
        values_valid = values[valid_mask]
        labels_valid = labels_array[valid_mask]

        if len(values_valid) == 0:
            continue

        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(values_valid.reshape(-1, 1)).flatten()

        for label_1, label_2, group_1, group_2 in comparisons:
            group1_values = scaled_values[labels_valid == label_1]
            group2_values = scaled_values[labels_valid == label_2]
            if len(group1_values) < 2 or len(group2_values) < 2:
                continue

            _, p_value = ttest_ind(group1_values, group2_values)
            mean_diff = float(np.mean(group1_values) - np.mean(group2_values))
            se_diff = np.sqrt(
                np.var(group1_values, ddof=1) / len(group1_values)
                + np.var(group2_values, ddof=1) / len(group2_values)
            )
            df_value = len(group1_values) + len(group2_values) - 2
            t_crit = stats.t.ppf(0.975, df_value)
            ci_lower = float(mean_diff - t_crit * se_diff)
            ci_upper = float(mean_diff + t_crit * se_diff)

            rows.append(
                {
                    "seed": int(feature["seed"]),
                    "K": int(feature["K"]),
                    "state": int(feature["state"]),
                    "feature_type": feature["feature_type"],
                    "dominant_group": feature["dominant_group"],
                    "group1": group_1,
                    "group2": group_2,
                    "mean_diff": mean_diff,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "df": int(df_value),
                    "p_raw": float(p_value),
                }
            )

    if not rows:
        return pd.DataFrame()

    posthoc_df = pd.DataFrame(rows)
    reject, pvals_corrected, _, _ = multipletests(posthoc_df["p_raw"], alpha=P_VALUE_THRESHOLD, method="fdr_bh")
    posthoc_df["p_fdr_bh"] = pvals_corrected
    posthoc_df["significant_after_correction"] = reject
    posthoc_df["significance"] = posthoc_df["p_fdr_bh"].map(significance_marker)
    return posthoc_df


def export_significant_state_npzs(
    corrected_significant_features: list[dict[str, object]],
    eeg_data: dict[str, np.ndarray],
    kmeans_models: dict[int, KMeans],
    scaled_X_by_k: dict[int, np.ndarray],
    labels_map: dict[str, int],
    output_dir: Path,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for feature in corrected_significant_features:
        k_states = int(feature["K"])
        state = int(feature["state"])
        kmeans = kmeans_models[k_states]
        X_scaled = scaled_X_by_k[k_states]
        preds = kmeans.predict(X_scaled)

        subject_windows_list: list[np.ndarray] = []
        subject_labels: list[int] = []
        subject_partids: list[str] = []

        for subject in np.unique(eeg_data["sub_ids"]):
            mask = (eeg_data["sub_ids"] == subject) & (preds == state)
            if not np.any(mask):
                continue

            subject_windows = eeg_data["X"][mask]
            n_windows_for_state = subject_windows.shape[0]
            subject_windows_reshaped = subject_windows.reshape(n_windows_for_state, 127, 5)
            subject_windows_list.append(subject_windows_reshaped)

            participant_id = f"sub-{int(subject):02d}"
            subject_partids.append(participant_id)
            subject_labels.append(int(labels_map[participant_id]))

        if not subject_windows_list:
            continue

        labels_out = np.asarray(subject_labels, dtype=int)
        partids_out = np.asarray(subject_partids, dtype=object)
        X_subjects_windows = np.asarray(subject_windows_list, dtype=object)

        output_path = output_dir / f"significant_K{k_states}_state{state}.npz"
        np.savez_compressed(
            output_path,
            X_subjects_windows=X_subjects_windows,
            labels=labels_out,
            participant_ids=partids_out,
            seed=np.int32(feature["seed"]),
            K=np.int32(k_states),
            state=np.int32(state),
            p_value=np.float64(feature["p_value"]),
            p_value_corrected=np.float64(feature["p_value_corrected"]),
            dominant_group=np.asarray(feature["dominant_group"]),
        )

        window_counts = [windows.shape[0] for windows in subject_windows_list]
        rows.append(
            {
                "seed": int(feature["seed"]),
                "K": k_states,
                "state": state,
                "dominant_group": feature["dominant_group"],
                "n_subjects": len(subject_windows_list),
                "total_windows": int(sum(window_counts)),
                "min_windows_per_subject": int(min(window_counts)),
                "max_windows_per_subject": int(max(window_counts)),
                "mean_windows_per_subject": float(np.mean(window_counts)),
                "npz_path": str(output_path),
            }
        )

    return pd.DataFrame(rows)


def summarize_subject_variability_by_direction(
    participant_df: pd.DataFrame,
    significant_features: list[dict[str, object]],
) -> pd.DataFrame:
    participant_base = participant_df[["participant_id", "Risk_Group", "Group"]].copy()
    participant_ids = participant_base["participant_id"].tolist()
    value_map: dict[str, list[float]] = {participant_id: [] for participant_id in participant_ids}

    for feature in significant_features:
        subject_ids = np.asarray(feature["subject_ids"])
        feature_values = np.asarray(feature["feature_values"], dtype=float)

        for subject_id, value in zip(subject_ids, feature_values):
            if np.isnan(value):
                continue
            participant_id = f"sub-{int(subject_id):02d}"
            if participant_id in value_map:
                value_map[participant_id].append(float(value))

    mean_values: list[float] = []
    median_values: list[float] = []
    cluster_counts: list[int] = []
    for participant_id in participant_ids:
        values = value_map[participant_id]
        if values:
            mean_values.append(float(np.mean(values)))
            median_values.append(float(np.median(values)))
            cluster_counts.append(len(values))
        else:
            mean_values.append(np.nan)
            median_values.append(np.nan)
            cluster_counts.append(0)

    summary_df = participant_base.copy()
    summary_df["mean_variability"] = mean_values
    summary_df["median_variability"] = median_values
    summary_df["n_clusters_used"] = cluster_counts
    return summary_df


def run_seed_analysis(
    seed: int,
    eeg_data: dict[str, np.ndarray],
    labels_map: dict[str, int],
    participant_df: pd.DataFrame,
    seed_output_dir: Path,
) -> dict[str, object]:
    print("\n" + "=" * 80)
    print(f"RUNNING SEED {seed}")
    print("=" * 80)

    seed_output_dir.mkdir(parents=True, exist_ok=True)
    npz_output_dir = seed_output_dir / "significant_state_npz"

    kmeans_models: dict[int, KMeans] = {}
    scaled_X_by_k: dict[int, np.ndarray] = {}
    all_features_for_wide: list[dict[str, object]] = []
    all_state_results: list[dict[str, object]] = []

    for k_states in K_STATES_LIST:
        print(f"\nAnalyzing K = {k_states} for seed {seed}")
        print("-" * 40)

        X_scaled = eeg_data["X"] / 2
        kmeans = KMeans(n_clusters=k_states, random_state=seed, n_init=10, max_iter=300)
        kmeans.fit(X_scaled)

        kmeans_models[k_states] = kmeans
        scaled_X_by_k[k_states] = X_scaled

        subject_features = compute_std_distance_to_centroid_only(
            X_scaled,
            eeg_data["y"],
            eeg_data["sub_ids"],
            kmeans,
            k_states,
        )
        if not subject_features:
            continue

        labels = np.asarray([subject["label"] for subject in subject_features])
        subject_ids = np.asarray([subject["subject_id"] for subject in subject_features])
        std_distance_matrix = np.vstack([subject["std_distance_to_centroid"] for subject in subject_features])

        for state in range(k_states):
            values = std_distance_matrix[:, state]
            feature_name = f"EEG_K{k_states}_std_distance_to_centroid_state{state}"

            for subject_id, value in zip(subject_ids, values):
                all_features_for_wide.append(
                    {
                        "seed": seed,
                        "subject_id_num": int(subject_id),
                        "participant_id": f"sub-{int(subject_id):02d}",
                        "feature_name": feature_name,
                        "value": value,
                        "K": k_states,
                        "state": state,
                    }
                )

            valid_mask = ~np.isnan(values)
            if not np.any(valid_mask):
                continue

            valid_values = values[valid_mask]
            valid_labels = labels[valid_mask]
            if len(np.unique(valid_labels)) < 2:
                continue

            groups = [valid_values[valid_labels == label] for label in np.unique(valid_labels)]
            test_name, p_value, stat_value = choose_statistical_test(groups)
            dominant_group, group_means = determine_dominant_group(valid_values, valid_labels)

            all_state_results.append(
                {
                    "seed": seed,
                    "K": k_states,
                    "feature_type": "std_distance_to_centroid",
                    "state": state,
                    "p_value": p_value,
                    "test": test_name,
                    "stat_value": stat_value,
                    "dominant_group": dominant_group,
                    "group_means": group_means,
                    "feature_values": values,
                    "subject_ids": subject_ids.copy(),
                    "raw_significant": p_value < P_VALUE_THRESHOLD,
                }
            )

            if p_value < P_VALUE_THRESHOLD:
                print(f"  ✓ State {state}: p={p_value:.6f} ({dominant_group} higher)")

    wide_df = pd.DataFrame(all_features_for_wide)
    wide_df.to_csv(seed_output_dir / f"eeg_subject_features_seed{seed}.csv", index=False)

    all_results_df = pd.DataFrame(
        [
            {
                "seed": result["seed"],
                "K": result["K"],
                "state": result["state"],
                "feature_type": result["feature_type"],
                "test": result["test"],
                "stat_value": result["stat_value"],
                "p_value": result["p_value"],
                "dominant_group": result["dominant_group"],
                "N_mean": result["group_means"].get(0, np.nan),
                "A+P-_mean": result["group_means"].get(1, np.nan),
                "A+P+_mean": result["group_means"].get(2, np.nan),
                "raw_significant": result["raw_significant"],
            }
            for result in all_state_results
        ]
    )
    all_results_df.to_csv(seed_output_dir / f"eeg_state_test_summary_seed{seed}.csv", index=False)

    raw_sig_results = [result for result in all_state_results if result["raw_significant"]]
    if raw_sig_results:
        reject, pvals_corrected, _, _ = multipletests(
            [result["p_value"] for result in raw_sig_results],
            alpha=P_VALUE_THRESHOLD,
            method="fdr_bh",
        )
        for idx, result in enumerate(raw_sig_results):
            result["p_value_corrected"] = float(pvals_corrected[idx])
            result["significant_after_correction"] = bool(reject[idx])
    else:
        for result in raw_sig_results:
            result["p_value_corrected"] = np.nan
            result["significant_after_correction"] = False

    corrected_significant_features = [
        result for result in raw_sig_results if result.get("significant_after_correction", False)
    ]

    corrected_summary_df = pd.DataFrame(
        [
            {
                "seed": result["seed"],
                "K": result["K"],
                "state": result["state"],
                "feature_type": result["feature_type"],
                "test": result["test"],
                "stat_value": result["stat_value"],
                "p_value": result["p_value"],
                "p_value_corrected": result["p_value_corrected"],
                "dominant_group": result["dominant_group"],
                "N_mean": result["group_means"].get(0, np.nan),
                "A+P-_mean": result["group_means"].get(1, np.nan),
                "A+P+_mean": result["group_means"].get(2, np.nan),
            }
            for result in corrected_significant_features
        ]
    )
    corrected_summary_path = seed_output_dir / f"eeg_significant_state_summary_seed{seed}.csv"
    corrected_summary_df.to_csv(corrected_summary_path, index=False)

    npz_summary_df = export_significant_state_npzs(
        corrected_significant_features=corrected_significant_features,
        eeg_data=eeg_data,
        kmeans_models=kmeans_models,
        scaled_X_by_k=scaled_X_by_k,
        labels_map=labels_map,
        output_dir=npz_output_dir,
    )
    npz_summary_df.to_csv(seed_output_dir / f"eeg_significant_npz_summary_seed{seed}.csv", index=False)

    participant_to_row = {row["participant_id"]: idx for idx, row in participant_df.iterrows()}
    feature_df = participant_df.copy()
    for result in corrected_significant_features:
        column_name = f"EEG_K{result['K']}_std_distance_to_centroid_state{result['state']}"
        feature_df[column_name] = np.nan
        feature_df[f"{column_name}_pval"] = result["p_value"]
        feature_df[f"{column_name}_pval_fdr_bh"] = result["p_value_corrected"]
        feature_df[f"{column_name}_dominant_group"] = result["dominant_group"]

        for subject_id, value in zip(result["subject_ids"], result["feature_values"]):
            participant_id = f"sub-{int(subject_id):02d}"
            row_index = participant_to_row.get(participant_id)
            if row_index is not None:
                feature_df.loc[row_index, column_name] = value

    feature_df.to_csv(seed_output_dir / f"eeg_significant_features_seed{seed}.csv", index=False)

    posthoc_df = perform_posthoc_comparisons(corrected_significant_features, labels_map)
    if not posthoc_df.empty:
        posthoc_df.to_csv(seed_output_dir / f"eeg_posthoc_seed{seed}.csv", index=False)

    print(f"\nSeed {seed} raw-significant states: {len(raw_sig_results)}")
    print(f"Seed {seed} FDR-surviving states: {len(corrected_significant_features)}")

    return {
        "seed": seed,
        "corrected_significant_features": corrected_significant_features,
        "corrected_summary_path": corrected_summary_path,
        "npz_dir": npz_output_dir,
        "posthoc_path": seed_output_dir / f"eeg_posthoc_seed{seed}.csv",
    }


# ============================================================================
# Regional variability analysis on pooled significant states
# ============================================================================


def extract_alpha_prefix(ch_name: str) -> str:
    letters = "".join(char for char in ch_name if char.isalpha())
    return letters.upper()


def infer_region_from_name(ch_name: str) -> str:
    prefix = extract_alpha_prefix(ch_name)
    if not prefix:
        return "Frontal"
    if prefix.startswith(("FP", "AF", "FC", "FT")) or prefix.startswith("F"):
        return "Frontal"
    if prefix.startswith("TP") or prefix.startswith("T"):
        return "Temporal"
    if prefix.startswith("CP") or prefix.startswith("C"):
        return "Central"
    if prefix.startswith("PO") or prefix.startswith("P"):
        return "Parietal"
    if prefix.startswith("O"):
        return "Occipital"
    for char, region in (
        ("F", "Frontal"),
        ("T", "Temporal"),
        ("C", "Central"),
        ("P", "Parietal"),
        ("O", "Occipital"),
    ):
        if char in prefix:
            return region
    return "Frontal"


def collect_groupwise_squared_differences(npz_files: list[Path]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    N_list: list[np.ndarray] = []
    A_plus_P_minus_list: list[np.ndarray] = []
    A_plus_P_plus_list: list[np.ndarray] = []

    for file_path in npz_files:
        data = np.load(file_path, allow_pickle=True)
        subject_windows_list = data["X_subjects_windows"]
        labels = data["labels"]
        centroid = np.mean(np.vstack(subject_windows_list), axis=0, keepdims=True)

        for subject_idx in range(subject_windows_list.shape[0]):
            subject_windows = subject_windows_list[subject_idx]
            label = int(labels[subject_idx])
            squared_diff = (subject_windows - centroid) ** 2

            if label == 0:
                N_list.append(squared_diff)
            elif label == 1:
                A_plus_P_minus_list.append(squared_diff)
            elif label == 2:
                A_plus_P_plus_list.append(squared_diff)

    return N_list, A_plus_P_minus_list, A_plus_P_plus_list


def analyze_regional_differences(
    N_list: list[np.ndarray],
    A_plus_P_minus_list: list[np.ndarray],
    A_plus_P_plus_list: list[np.ndarray],
    ch_names: list[str],
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, list[int]]]:
    channel_to_region: dict[int, str] = {}
    region_channels: dict[str, list[int]] = defaultdict(list)

    for channel_idx, ch_name in enumerate(ch_names):
        region = infer_region_from_name(ch_name)
        channel_to_region[channel_idx] = region
        region_channels[region].append(channel_idx)

    regional_subject_data: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"N": [], "A+P-": [], "A+P+": []})
    )

    for group_name, group_list in (
        ("N", N_list),
        ("A+P-", A_plus_P_minus_list),
        ("A+P+", A_plus_P_plus_list),
    ):
        for subject_data in group_list:
            for region, channel_indices in region_channels.items():
                for band_idx in range(subject_data.shape[2]):
                    band_name = BAND_NAMES[band_idx] if band_idx < len(BAND_NAMES) else f"band_{band_idx}"
                    total_windows = subject_data.shape[0]
                    region_band_avg = np.sqrt(
                        np.sum(subject_data[:, channel_indices, band_idx].reshape(total_windows, -1), axis=1)
                    )
                    regional_subject_data[region][band_name][group_name].append(float(np.std(region_band_avg)))

    regional_results: dict[str, dict[str, dict[str, object]]] = {}
    for region in regional_subject_data:
        regional_results[region] = {}
        for band in regional_subject_data[region]:
            n_values = np.asarray(regional_subject_data[region][band]["N"])
            apm_values = np.asarray(regional_subject_data[region][band]["A+P-"])
            app_values = np.asarray(regional_subject_data[region][band]["A+P+"])

            regional_results[region][band] = {
                "N_mean_std": float(np.mean(n_values)),
                "N_std_of_std": float(np.std(n_values)),
                "N_values": n_values,
                "A+P-_mean_std": float(np.mean(apm_values)),
                "A+P-_std_of_std": float(np.std(apm_values)),
                "A+P-_values": apm_values,
                "A+P+_mean_std": float(np.mean(app_values)),
                "A+P+_std_of_std": float(np.std(app_values)),
                "A+P+_values": app_values,
                "N_n": int(len(n_values)),
                "A+P-_n": int(len(apm_values)),
                "A+P+_n": int(len(app_values)),
            }

    return regional_results, region_channels


def statistical_testing(regional_results: dict[str, dict[str, dict[str, object]]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []

    for region, bands_data in regional_results.items():
        for band, values in bands_data.items():
            n_vals = np.asarray(values["N_values"])
            apm_vals = np.asarray(values["A+P-_values"])
            app_vals = np.asarray(values["A+P+_values"])
            groups = [n_vals, apm_vals, app_vals]

            normality_ok = all(shapiro(group)[1] > 0.05 for group in groups if len(group) >= 3)
            equal_var_ok = levene(*groups)[1] > 0.05

            if normality_ok and equal_var_ok:
                stat_value, p_value = f_oneway(*groups)
                test_name = "ANOVA"
            else:
                stat_value, p_value = kruskal(*groups)
                test_name = "Kruskal-Wallis"

            means = {
                "N": values["N_mean_std"],
                "A+P-": values["A+P-_mean_std"],
                "A+P+": values["A+P+_mean_std"],
            }

            results.append(
                {
                    "region": region,
                    "band": band,
                    "p_value": float(p_value),
                    "test": test_name,
                    "highest_variability_group": max(means, key=means.get),
                    "N_mean_std": values["N_mean_std"],
                    "N_std_of_std": values["N_std_of_std"],
                    "N_n": values["N_n"],
                    "A+P-_mean_std": values["A+P-_mean_std"],
                    "A+P-_std_of_std": values["A+P-_std_of_std"],
                    "A+P-_n": values["A+P-_n"],
                    "A+P+_mean_std": values["A+P+_mean_std"],
                    "A+P+_std_of_std": values["A+P+_std_of_std"],
                    "A+P+_n": values["A+P+_n"],
                    "effect_size": float(stat_value),
                }
            )

    if not results:
        return results

    reject, pvals_corrected, _, _ = multipletests(
        [result["p_value"] for result in results],
        alpha=P_VALUE_THRESHOLD,
        method="fdr_bh",
    )
    for idx, result in enumerate(results):
        result["p_value_corrected"] = float(pvals_corrected[idx])
        result["significant_after_correction"] = bool(reject[idx])
    return results


def run_regional_analysis_for_npz_files(
    npz_files: list[Path],
    ch_names: list[str],
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, list[int]], pd.DataFrame]:
    if not npz_files:
        return {}, {}, pd.DataFrame()

    N_list, A_plus_P_minus_list, A_plus_P_plus_list = collect_groupwise_squared_differences(npz_files)
    regional_results, region_channels = analyze_regional_differences(
        N_list,
        A_plus_P_minus_list,
        A_plus_P_plus_list,
        ch_names,
    )
    all_regional_results_df = pd.DataFrame(statistical_testing(regional_results))
    return regional_results, region_channels, all_regional_results_df


def build_region_band_seed_support(
    seed_regional_results_by_seed: dict[int, pd.DataFrame],
    total_seeds: int,
) -> pd.DataFrame:
    combined_frames: list[pd.DataFrame] = []

    for seed, seed_df in seed_regional_results_by_seed.items():
        if seed_df is None or seed_df.empty:
            continue
        df = seed_df.copy()
        if "seed" not in df.columns:
            df["seed"] = seed
        combined_frames.append(df)

    if not combined_frames:
        return pd.DataFrame(
            columns=[
                "region",
                "band",
                "seed_support",
                "seed_support_fraction",
                "support_seeds",
                "majority_highest_variability_group",
                "majority_group_support",
                "majority_group_support_fraction",
                "majority_group_seeds",
                "median_p_value_corrected_majority_group",
                "median_p_value_corrected_any",
                "N_seed_support",
                "N_seeds",
                "A_plus_P_minus_seed_support",
                "A_plus_P_minus_seeds",
                "A_plus_P_plus_seed_support",
                "A_plus_P_plus_seeds",
            ]
        )

    combined_df = pd.concat(combined_frames, ignore_index=True)
    significant_df = combined_df[combined_df["significant_after_correction"] == True].copy()
    if significant_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (region, band), group_df in significant_df.groupby(["region", "band"], sort=False):
        seed_support = int(group_df["seed"].nunique())
        support_seeds = sorted(int(seed) for seed in group_df["seed"].unique())
        direction_support: dict[str, object] = {}

        for direction_group in DIRECTION_GROUPS:
            direction_seed_values = sorted(
                int(seed)
                for seed in group_df.loc[
                    group_df["highest_variability_group"] == direction_group,
                    "seed",
                ].unique()
            )
            safe_name = group_to_safe_name(direction_group)
            direction_support[f"{safe_name}_seed_support"] = len(direction_seed_values)
            direction_support[f"{safe_name}_seeds"] = ",".join(str(seed) for seed in direction_seed_values)

        group_support_series = (
            group_df.groupby("highest_variability_group")["seed"]
            .nunique()
            .sort_values(ascending=False)
        )
        majority_group = str(group_support_series.index[0])
        majority_group_support = int(group_support_series.iloc[0])
        majority_group_df = group_df[group_df["highest_variability_group"] == majority_group]
        majority_group_seeds = sorted(int(seed) for seed in majority_group_df["seed"].unique())

        rows.append(
            {
                "region": region,
                "band": band,
                "seed_support": seed_support,
                "seed_support_fraction": seed_support / total_seeds,
                "support_seeds": ",".join(str(seed) for seed in support_seeds),
                "majority_highest_variability_group": majority_group,
                "majority_group_support": majority_group_support,
                "majority_group_support_fraction": majority_group_support / total_seeds,
                "majority_group_seeds": ",".join(str(seed) for seed in majority_group_seeds),
                "median_p_value_corrected_majority_group": float(majority_group_df["p_value_corrected"].median()),
                "median_p_value_corrected_any": float(group_df["p_value_corrected"].median()),
                **direction_support,
            }
        )

    support_df = pd.DataFrame(rows)
    support_df = support_df.sort_values(
        by=[
            "majority_group_support",
            "seed_support",
            "median_p_value_corrected_majority_group",
            "median_p_value_corrected_any",
            "region",
            "band",
        ],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    support_df["selected_for_consistent_top10_figure"] = False
    eligible_idx = support_df.index[support_df["majority_group_support"] >= MIN_CONSISTENT_REGION_BAND_SUPPORT].tolist()
    if eligible_idx:
        selected_idx = eligible_idx[:TOP_CONSISTENT_REGION_BANDS]
    else:
        selected_idx = support_df.index[:TOP_CONSISTENT_REGION_BANDS].tolist()
    support_df.loc[selected_idx, "selected_for_consistent_top10_figure"] = True
    return support_df


def add_significance_bracket(ax: plt.Axes, x1: float, x2: float, y: float, p_value: float) -> None:
    height = y * 0.04 if y > 0 else 0.04
    ax.plot([x1, x1, x2, x2], [y, y + height, y + height, y], lw=1.5, c="black")

    if p_value < 0.001:
        sig_text = "***"
    elif p_value < 0.01:
        sig_text = "**"
    elif p_value < 0.05:
        sig_text = "*"
    else:
        sig_text = "ns"

    ax.text((x1 + x2) / 2, y + height * 1.2, sig_text, ha="center", va="bottom", fontsize=10)


def create_visualizations(
    all_results_df: pd.DataFrame,
    corrected_results_df: pd.DataFrame,
    region_channels: dict[str, list[int]],
    regional_results: dict[str, dict[str, dict[str, object]]],
    raw_info: mne.Info,
    output_dir: Path,
    *,
    generate_summary_plots: bool = True,
    barplot_filename: str = "regional_variability_barplot_topomap_multiseed.png",
    support_lookup: dict[tuple[str, str], int] | None = None,
    support_denominator: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if generate_summary_plots and not all_results_df.empty:
        pivot_p = all_results_df.pivot(index="region", columns="band", values="p_value_corrected")
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_p,
            annot=True,
            cmap="viridis_r",
            fmt=".4f",
            cbar_kws={"label": "FDR-corrected p-value"},
        )
        plt.title("Regional EEG Variability Differences Across All Seeds")
        plt.tight_layout()
        plt.savefig(output_dir / "regional_variability_significance_heatmap_multiseed.png", dpi=300, bbox_inches="tight")
        plt.close()

        top_df = all_results_df.sort_values("p_value_corrected").head(20)
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(top_df)), -np.log10(top_df["p_value_corrected"].clip(lower=1e-12)))
        plt.axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="FDR p=0.05")
        plt.xlabel("Region-Band Combinations")
        plt.ylabel("-log10(FDR-corrected p-value)")
        plt.title("Top Regional EEG Variability Differences Across All Seeds")
        plt.xticks(
            range(len(top_df)),
            [f"{row['region']}\n{row['band']}" for _, row in top_df.iterrows()],
            rotation=45,
            ha="right",
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "top_significant_variability_regions_multiseed.png", dpi=300, bbox_inches="tight")
        plt.close()

    if corrected_results_df.empty:
        return

    band_order = [band for band in BAND_NAMES if band in corrected_results_df["band"].unique()]
    results_by_band = {
        band: corrected_results_df[corrected_results_df["band"] == band].sort_values("p_value_corrected")
        for band in band_order
    }

    n_bands = len(results_by_band)
    n_rows = int(math.ceil(n_bands / 2))
    fig = plt.figure(figsize=(24, 4.5 * max(n_rows, 1)))
    gs = fig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 0.1, 0.9], hspace=0.4, wspace=0.25)
    ax_topo = fig.add_subplot(gs[:, 3])

    mne.viz.plot_sensors(raw_info, ch_type="eeg", show_names=False, show=False, axes=ax_topo)
    sensors_collection = ax_topo.collections[0]
    inferred_regions = [infer_region_from_name(ch) for ch in raw_info["ch_names"]]
    sensor_colors = [REGION_COLORS.get(region, "#E0E0E0") for region in inferred_regions]
    sensors_collection.set_facecolors(sensor_colors)
    sensors_collection.set_alpha(0.9)
    sensors_collection.set_edgecolors("black")
    sensors_collection.set_linewidth(0.8)
    sensors_collection.set_sizes([450] * len(raw_info["ch_names"]))

    for idx, (band, band_df) in enumerate(results_by_band.items()):
        row = idx // 2
        col = idx % 2
        if n_bands % 2 == 1 and idx == n_bands - 1:
            inner_gs = GridSpecFromSubplotSpec(
                1,
                3,
                subplot_spec=gs[row, :2],
                width_ratios=[0.2, 1.0, 0.2],
                wspace=0.0,
            )
            ax_bar = fig.add_subplot(inner_gs[0, 1])
        else:
            ax_bar = fig.add_subplot(gs[row, col])

        regions = band_df["region"].values
        x_pos = np.arange(len(regions))
        bar_width = 0.25

        for region_idx, (_, row_data) in enumerate(band_df.iterrows()):
            region = row_data["region"]
            base_color = REGION_COLORS.get(region, "#CCCCCC")

            ax_bar.bar(
                x_pos[region_idx] - bar_width,
                row_data["N_mean_std"],
                bar_width,
                yerr=row_data["N_std_of_std"],
                color=base_color,
                alpha=1.0,
                capsize=4,
                error_kw={"linewidth": 1.5, "elinewidth": 1.5},
                edgecolor="black",
                linewidth=1.5,
                label="N" if region_idx == 0 else "",
            )
            ax_bar.bar(
                x_pos[region_idx],
                row_data["A+P-_mean_std"],
                bar_width,
                yerr=row_data["A+P-_std_of_std"],
                color=base_color,
                alpha=0.8,
                capsize=4,
                error_kw={"linewidth": 1.5, "elinewidth": 1.5},
                edgecolor="black",
                linewidth=1.5,
                hatch="////",
                label="A+P-" if region_idx == 0 else "",
            )
            ax_bar.bar(
                x_pos[region_idx] + bar_width,
                row_data["A+P+_mean_std"],
                bar_width,
                yerr=row_data["A+P+_std_of_std"],
                color=base_color,
                alpha=0.6,
                capsize=4,
                error_kw={"linewidth": 1.5, "elinewidth": 1.5},
                edgecolor="black",
                linewidth=1.5,
                hatch="xxxx",
                label="A+P+" if region_idx == 0 else "",
            )

            n_vals = regional_results[region][band]["N_values"]
            apm_vals = regional_results[region][band]["A+P-_values"]
            app_vals = regional_results[region][band]["A+P+_values"]
            y_vals = [
                row_data["N_mean_std"] + row_data["N_std_of_std"],
                row_data["A+P-_mean_std"] + row_data["A+P-_std_of_std"],
                row_data["A+P+_mean_std"] + row_data["A+P+_std_of_std"],
            ]
            y_max = max(y_vals)

            _, p_n_apm = mannwhitneyu(n_vals, apm_vals, alternative="two-sided")
            if p_n_apm < 0.05:
                add_significance_bracket(ax_bar, x_pos[region_idx] - bar_width, x_pos[region_idx], y_max * 1.12, p_n_apm)

            _, p_n_app = mannwhitneyu(n_vals, app_vals, alternative="two-sided")
            if p_n_app < 0.05:
                add_significance_bracket(
                    ax_bar,
                    x_pos[region_idx] - bar_width,
                    x_pos[region_idx] + bar_width,
                    y_max * 1.28,
                    p_n_app,
                )

            _, p_apm_app = mannwhitneyu(apm_vals, app_vals, alternative="two-sided")
            if p_apm_app < 0.05:
                add_significance_bracket(
                    ax_bar,
                    x_pos[region_idx],
                    x_pos[region_idx] + bar_width,
                    y_max * 1.44,
                    p_apm_app,
                )

        if idx == 0:
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=1.5, label="N"),
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=1.5, hatch="////", label="A+P-"),
                plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", linewidth=1.5, hatch="xxxx", label="A+P+"),
            ]
            legend = ax_bar.legend(handles=legend_elements, loc="best", fontsize=12, frameon=True, edgecolor="black", facecolor="white", framealpha=1)
            legend.get_frame().set_linewidth(1.0)

        ax_bar.set_xlim(-0.5, len(regions) - 0.5)
        ax_bar.set_ylabel(f"{band.capitalize()} Variability", fontsize=14)
        ax_bar.set_xticks(x_pos)
        if support_lookup is not None and support_denominator:
            x_labels = [
                f"{region}\n({support_lookup.get((region, band), 0)}/{support_denominator})"
                for region in regions
            ]
        else:
            x_labels = list(regions)
        ax_bar.set_xticklabels(x_labels, fontsize=18)
        ax_bar.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.8)
        ax_bar.spines["top"].set_visible(False)
        ax_bar.spines["right"].set_visible(False)
        ax_bar.spines["left"].set_linewidth(1.0)
        ax_bar.spines["bottom"].set_linewidth(1.0)
        ax_bar.tick_params(labelsize=18, width=1.0, length=5)

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08, wspace=0.25, hspace=0.4)
    plt.savefig(output_dir / barplot_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_direction_split_consistency_figure(
    consistent_results_df: pd.DataFrame,
    output_dir: Path,
    *,
    filename: str = DIRECTION_SPLIT_FIGURE_FILENAME,
    support_denominator: int | None = None,
) -> None:
    if consistent_results_df.empty:
        return

    direction_order = [
        direction_group
        for direction_group in DIRECTION_GROUPS
        if direction_group in consistent_results_df["majority_highest_variability_group"].unique()
    ]
    if not direction_order:
        return

    max_rows = max(
        int((consistent_results_df["majority_highest_variability_group"] == direction_group).sum())
        for direction_group in direction_order
    )
    fig_height = max(4.5, 1.0 * max_rows + 1.8)
    fig, axes = plt.subplots(1, len(direction_order), figsize=(6.5 * len(direction_order), fig_height))
    axes_array = np.atleast_1d(axes)
    denominator_value = support_denominator if support_denominator is not None else 0
    support_columns = [
        ("N", "N_seed_support"),
        ("A+P-", "A_plus_P_minus_seed_support"),
        ("A+P+", "A_plus_P_plus_seed_support"),
    ]

    for panel_idx, direction_group in enumerate(direction_order):
        ax = axes_array[panel_idx]
        panel_df = consistent_results_df[
            consistent_results_df["majority_highest_variability_group"] == direction_group
        ].copy()
        panel_df = panel_df.sort_values(
            by=[
                "majority_group_support",
                "seed_support",
                "median_p_value_corrected_majority_group",
                "p_value_corrected",
                "band",
                "region",
            ],
            ascending=[False, False, True, True, True, True],
        ).reset_index(drop=True)

        if panel_df.empty:
            ax.axis("off")
            continue

        y_pos = np.arange(len(panel_df))
        bar_height = 0.68
        left_values = np.zeros(len(panel_df), dtype=float)

        for group_name, support_col in support_columns:
            counts = panel_df[support_col].to_numpy(dtype=float)
            ax.barh(
                y_pos,
                counts,
                left=left_values,
                height=bar_height,
                color=SUPPORT_GROUP_COLORS[group_name],
                alpha=0.9,
                edgecolor="black",
                linewidth=1.0,
                label=group_name if panel_idx == 0 else "",
            )

            for idx, count in enumerate(counts):
                if count <= 0:
                    continue
                x_center = left_values[idx] + count / 2.0
                ax.text(
                    x_center,
                    y_pos[idx],
                    str(int(count)),
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if count >= 1.5 else "black",
                    fontweight="bold",
                )
            left_values = left_values + counts

        y_labels = []
        for row in panel_df.itertuples(index=False):
            y_labels.append(
                (
                    f"{row.region} / {row.band}\n"
                    f"majority {int(row.majority_group_support)}/{support_denominator if support_denominator is not None else '?'}"
                )
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Seed support count", fontsize=11)
        ax.set_title(GROUP_TO_DIRECTION_LABEL[direction_group], fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3, linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", length=0)
        if denominator_value > 0:
            ax.set_xlim(0, denominator_value)
            ax.set_xticks(np.arange(0, denominator_value + 1, 1))
        else:
            ax.set_xlim(0, max(float(np.nanmax(left_values)), 1.0))

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=SUPPORT_GROUP_COLORS["N"], edgecolor="black", linewidth=1.0, label="N"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SUPPORT_GROUP_COLORS["A+P-"], edgecolor="black", linewidth=1.0, label="A+P-"),
        plt.Rectangle((0, 0), 1, 1, facecolor=SUPPORT_GROUP_COLORS["A+P+"], edgecolor="black", linewidth=1.0, label="A+P+"),
    ]
    axes_array[0].legend(handles=legend_elements, loc="lower right", fontsize=10, frameon=True)
    fig.suptitle("Directional seed support for consistent region-bands", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    eeg_data_dir = resolve_existing_path(EEG_DATA_DIR_CANDIDATES, "EEG HFD/PSD directory")
    mri_csv_path = resolve_existing_path(MRI_CSV_CANDIDATES, "MRI groups CSV")
    raw_vhdr_path = resolve_existing_path(RAW_VHDR_CANDIDATES, "reference EEG raw file")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EEG STD_DISTANCE_TO_CENTROID MULTISEED ANALYSIS")
    print("=" * 80)
    print(f"EEG data dir: {eeg_data_dir}")
    print(f"MRI CSV: {mri_csv_path}")
    print(f"Seeds: {SEED_LIST}")

    participant_df, risk_group_map = load_subject_labels(mri_csv_path, eeg_data_dir)
    print(f"\nSubjects with both EEG and MRI labels: {len(risk_group_map)}")
    for group_name, label in GROUP_TO_LABEL.items():
        count = sum(1 for value in risk_group_map.values() if value == label)
        print(f"  {group_name} (label={label}): {count} subjects")

    eeg_data = load_eeg_features(eeg_data_dir, risk_group_map)
    print("\nEEG Data Summary:")
    print(f"  Windows: {eeg_data['X'].shape[0]}")
    print(f"  Features per window: {eeg_data['X'].shape[1]}")
    print(f"  Unique subjects: {len(np.unique(eeg_data['sub_ids']))}")

    all_seed_summary_frames: list[pd.DataFrame] = []
    seed_results: dict[int, dict[str, object]] = {}

    for seed in SEED_LIST:
        seed_output_dir = OUTPUT_ROOT / f"seed_{seed}"
        seed_result = run_seed_analysis(
            seed=seed,
            eeg_data=eeg_data,
            labels_map=risk_group_map,
            participant_df=participant_df,
            seed_output_dir=seed_output_dir,
        )
        seed_results[seed] = seed_result

        summary_path = seed_result["corrected_summary_path"]
        if summary_path.exists():
            summary_df = pd.read_csv(summary_path)
            if not summary_df.empty:
                all_seed_summary_frames.append(summary_df)

    if all_seed_summary_frames:
        combined_summary_df = pd.concat(all_seed_summary_frames, ignore_index=True)
    else:
        combined_summary_df = pd.DataFrame(
            columns=[
                "seed",
                "K",
                "state",
                "feature_type",
                "test",
                "stat_value",
                "p_value",
                "p_value_corrected",
                "dominant_group",
                "N_mean",
                "A+P-_mean",
                "A+P+_mean",
            ]
        )
    combined_summary_path = OUTPUT_ROOT / "eeg_significant_state_summary_all_seeds.csv"
    combined_summary_df.to_csv(combined_summary_path, index=False)

    all_corrected_significant_features: list[dict[str, object]] = []
    for seed in SEED_LIST:
        all_corrected_significant_features.extend(seed_results[seed]["corrected_significant_features"])

    direction_feature_map = {
        direction_group: [
            feature
            for feature in all_corrected_significant_features
            if feature["dominant_group"] == direction_group
        ]
        for direction_group in DIRECTION_GROUPS
    }
    direction_csv_map = {
        "N": EEG_DIRECTION_N_HIGHER_CSV,
        "A+P-": EEG_DIRECTION_APM_HIGHER_CSV,
        "A+P+": EEG_DIRECTION_APP_HIGHER_CSV,
    }
    direction_summary_map: dict[str, pd.DataFrame] = {}

    for direction_group in DIRECTION_GROUPS:
        direction_summary_df = summarize_subject_variability_by_direction(
            participant_df=participant_df,
            significant_features=direction_feature_map[direction_group],
        )
        direction_summary_df["higher_variability_group"] = direction_group
        direction_summary_df["direction"] = GROUP_TO_DIRECTION_LABEL[direction_group]
        direction_summary_df.to_csv(direction_csv_map[direction_group], index=False)
        direction_summary_map[direction_group] = direction_summary_df

    risk_higher_summary_df = pd.concat(
        [direction_summary_map["A+P-"], direction_summary_map["A+P+"]],
        ignore_index=True,
    )
    risk_higher_summary_df["direction_family"] = "risk_higher"
    risk_higher_summary_df.to_csv(EEG_DIRECTION_RISK_HIGHER_CSV, index=False)

    combined_direction_df = pd.concat(
        [direction_summary_map[direction_group] for direction_group in DIRECTION_GROUPS],
        ignore_index=True,
    )
    combined_direction_df.to_csv(EEG_DIRECTION_COMBINED_CSV, index=False)

    table_seed_result = seed_results.get(TABLE_SEED)
    if table_seed_result and table_seed_result["posthoc_path"].exists():
        print(f"\nSeed {TABLE_SEED} post hoc table:")
        print(f"  {table_seed_result['posthoc_path']}")
    else:
        print(f"\nNo post hoc table was generated for seed {TABLE_SEED}.")

    raw = mne.io.read_raw_brainvision(raw_vhdr_path, preload=True, verbose=False)
    ch_names = raw.info["ch_names"]

    seed_regional_results_by_seed: dict[int, pd.DataFrame] = {}
    for seed in SEED_LIST:
        seed_npz_dir = OUTPUT_ROOT / f"seed_{seed}" / "significant_state_npz"
        seed_npz_files = sorted(seed_npz_dir.glob("*.npz")) if seed_npz_dir.exists() else []
        _, _, seed_regional_results_df = run_regional_analysis_for_npz_files(seed_npz_files, ch_names)
        if not seed_regional_results_df.empty:
            seed_regional_results_df["seed"] = seed
        seed_regional_results_by_seed[seed] = seed_regional_results_df

        seed_output_dir = OUTPUT_ROOT / f"seed_{seed}"
        seed_regional_results_df.to_csv(
            seed_output_dir / f"regional_variability_statistical_results_seed{seed}.csv",
            index=False,
        )
        if not seed_regional_results_df.empty:
            seed_regional_results_df[
                seed_regional_results_df["significant_after_correction"] == True
            ].copy().to_csv(
                seed_output_dir / f"regional_variability_statistical_results_seed{seed}_significant_only.csv",
                index=False,
            )

    seed_support_df = build_region_band_seed_support(
        seed_regional_results_by_seed=seed_regional_results_by_seed,
        total_seeds=len(SEED_LIST),
    )
    seed_support_path = OUTPUT_ROOT / "region_band_seed_support_multiseed.csv"
    seed_support_df.to_csv(seed_support_path, index=False)

    selected_consistent_support_df = seed_support_df[
        seed_support_df["selected_for_consistent_top10_figure"] == True
    ].copy()
    selected_consistent_support_path = OUTPUT_ROOT / "region_band_seed_support_top10_consistent_multiseed.csv"
    selected_consistent_support_df.to_csv(selected_consistent_support_path, index=False)

    pooled_npz_files: list[Path] = []
    for seed in SEED_LIST:
        seed_npz_dir = OUTPUT_ROOT / f"seed_{seed}" / "significant_state_npz"
        if seed_npz_dir.exists():
            pooled_npz_files.extend(sorted(seed_npz_dir.glob("*.npz")))

    pooled_output_dir = OUTPUT_ROOT / "pooled_across_seeds"
    pooled_output_dir.mkdir(parents=True, exist_ok=True)

    if not pooled_npz_files:
        print("\nNo significant NPZ files found across seeds. Skipping pooled regional analysis.")
        print(f"\nCombined summary saved to: {combined_summary_path}")
        return

    print("\n" + "=" * 80)
    print("POOLED REGIONAL VARIABILITY ANALYSIS ACROSS ALL SEEDS")
    print("=" * 80)
    print(f"Pooled significant state files: {len(pooled_npz_files)}")

    regional_results, region_channels, all_regional_results_df = run_regional_analysis_for_npz_files(
        pooled_npz_files,
        ch_names,
    )
    all_regional_results_df.to_csv(
        pooled_output_dir / "regional_variability_statistical_results_multiseed.csv",
        index=False,
    )

    corrected_regional_results_df = all_regional_results_df[
        all_regional_results_df["significant_after_correction"] == True
    ].copy()
    corrected_regional_results_df.to_csv(
        pooled_output_dir / "regional_variability_statistical_results_multiseed_significant_only.csv",
        index=False,
    )

    create_visualizations(
        all_results_df=all_regional_results_df,
        corrected_results_df=corrected_regional_results_df,
        region_channels=region_channels,
        regional_results=regional_results,
        raw_info=raw.info,
        output_dir=pooled_output_dir,
    )

    consistent_plot_df = all_regional_results_df.merge(
        selected_consistent_support_df[
            [
                "region",
                "band",
                "seed_support",
                "N_seed_support",
                "A_plus_P_minus_seed_support",
                "A_plus_P_plus_seed_support",
                "majority_highest_variability_group",
                "majority_group_support",
                "median_p_value_corrected_majority_group",
            ]
        ],
        on=["region", "band"],
        how="inner",
    )
    consistent_plot_df = consistent_plot_df.sort_values(
        by=[
            "majority_group_support",
            "seed_support",
            "median_p_value_corrected_majority_group",
            "p_value_corrected",
            "band",
            "region",
        ],
        ascending=[False, False, True, True, True, True],
    ).reset_index(drop=True)
    consistent_plot_df.to_csv(
        pooled_output_dir / "regional_variability_top10_consistent_multiseed.csv",
        index=False,
    )

    if not consistent_plot_df.empty:
        consistency_support_lookup = {
            (row["region"], row["band"]): int(row["majority_group_support"])
            for _, row in selected_consistent_support_df.iterrows()
        }
        create_visualizations(
            all_results_df=all_regional_results_df,
            corrected_results_df=consistent_plot_df,
            region_channels=region_channels,
            regional_results=regional_results,
            raw_info=raw.info,
            output_dir=pooled_output_dir,
            generate_summary_plots=False,
            barplot_filename="regional_variability_barplot_topomap_multiseed_consistent_top10.png",
            support_lookup=consistency_support_lookup,
            support_denominator=len(SEED_LIST),
        )
        create_direction_split_consistency_figure(
            consistent_results_df=consistent_plot_df,
            output_dir=pooled_output_dir,
            support_denominator=len(SEED_LIST),
        )

    print("\nSaved outputs:")
    print(f"  Combined state summary: {combined_summary_path}")
    print(f"  N-higher subject variability: {EEG_DIRECTION_N_HIGHER_CSV}")
    print(f"  A+P--higher subject variability: {EEG_DIRECTION_APM_HIGHER_CSV}")
    print(f"  A+P+-higher subject variability: {EEG_DIRECTION_APP_HIGHER_CSV}")
    print(f"  Risk-higher subject variability: {EEG_DIRECTION_RISK_HIGHER_CSV}")
    print(f"  Combined subject variability: {EEG_DIRECTION_COMBINED_CSV}")
    print(f"  Seed 9 post hoc table: {OUTPUT_ROOT / f'seed_{TABLE_SEED}' / f'eeg_posthoc_seed{TABLE_SEED}.csv'}")
    print(f"  Region-band seed support: {seed_support_path}")
    print(f"  Top consistent region-bands: {selected_consistent_support_path}")
    print(f"  Pooled regional stats: {pooled_output_dir / 'regional_variability_statistical_results_multiseed.csv'}")
    print(f"  Pooled figure: {pooled_output_dir / 'regional_variability_barplot_topomap_multiseed.png'}")
    print(
        "  Consistency-filtered figure: "
        f"{pooled_output_dir / 'regional_variability_barplot_topomap_multiseed_consistent_top10.png'}"
    )
    print(
        "  Direction-split consistency figure: "
        f"{pooled_output_dir / DIRECTION_SPLIT_FIGURE_FILENAME}"
    )


if __name__ == "__main__":
    main()
