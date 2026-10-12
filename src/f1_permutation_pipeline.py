import argparse
import os
import random
import warnings
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import f_oneway, kruskal, levene, mannwhitneyu, shapiro, ttest_ind
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from threadpoolctl import threadpool_limits

from utils import (
    EEG_DATA_DIR,
    FMRI_DATA_DIR,
    SMRI_DATA_DIR,
    extract_psychometric_features,
    get_X_y,
    load_and_filter_all_modalities,
)

warnings.filterwarnings("ignore")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "permutation")


MODALITIES = ["sMRI", "Psychometric", "fMRI", "EEG"]

WIN_STEP = 19  # fMRI window step in volumes (src/fmri_sliding_window_fc.py)
TR = 0.8
FMRI_STEP_SEC = WIN_STEP * TR  # 15.2 s; features are rescaled before classification, so F1 does not depend on it

EEG_WINDOW_SIZE_SEC = 4.0
EEG_OVERLAP_PERCENT = 0.75
EEG_STEP_SEC = EEG_WINDOW_SIZE_SEC * (1 - EEG_OVERLAP_PERCENT)
SEED_LIST = [9, 42, 123, 2021, 2022]
K_STATES_LIST_FMRI = list(range(4, 10))
K_STATES_LIST_EEG = list(range(5, 11))

NORMALITY_ALPHA = 0.05
VARIANCE_ALPHA = 0.05
MIN_SAMPLE_SIZE = 3
P_VALUE_SMOOTHING = 1e-10
WEIGHT_POWER = 2.0

XGB_DEVICE = "cuda"

# How the null is built. "full": permute the subject label vector once per permutation and
# rerun the cross-validation with it, so each fold's training and held-out labels come from
# the same permutation (Ojala & Garriga, 2010). "per-fold": shuffle each fold's training labels
# independently and score against the true held-out labels (the scheme used for the April 2026
# table; it can give a narrower null). Set with --perm-scheme.
PERM_SCHEME = "full"
XGB_N_JOBS = 1
DEFAULT_PERM_JOBS = min(8, os.cpu_count() or 1)

_PERM_WORKER_FOLD_CACHE = None
_PERM_WORKER_SEED_VAL = None
_PERM_WORKER_XGB_DEVICE = None

TASKS = [
    ("N", "A+P-"),
    ("N", "A+P+"),
    ("A+P-", "A+P+"),
    ("N", "A+P-", "A+P+"),
]

TASK_LABELS = {
    ("N", "A+P-"): "N vs A+P-",
    ("N", "A+P+"): "N vs A+P+",
    ("A+P-", "A+P+"): "A+P- vs A+P+",
    ("N", "A+P-", "A+P+"): "N vs (A+P- and A+P+)",
}

TASK_ORDER = [
    "N vs A+P-",
    "N vs A+P+",
    "A+P- vs A+P+",
    "N vs (A+P- and A+P+)",
]

MODALITY_LABELS = {
    "EEG": "EEG",
    "fMRI": "fMRI",
    "sMRI": "sMRI",
    "Psychometric": "Clinical",
}

MODALITY_ORDER = ["EEG", "fMRI", "sMRI", "Clinical"]


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_indices_for_subjects(sub_ids, target_subjects):
    return np.isin(sub_ids, target_subjects)


def _as_int_labels(y):
    return np.asarray(y, dtype=int)


def _normalize_seeds(seeds):
    if isinstance(seeds, (int, np.integer)):
        return (int(seeds),)
    return tuple(seeds)


def _normalize_jobs(n_jobs):
    if n_jobs in (None, 0):
        return 1
    if n_jobs < 0:
        return max(1, os.cpu_count() or 1)
    return max(1, int(n_jobs))


def _resolve_participants_tsv():
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "filtered_participants.tsv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    raise FileNotFoundError("Could not find filtered_participants.tsv")


def _resolve_path(path_like):
    candidates = [
        os.path.abspath(path_like),
        os.path.abspath(os.path.join(os.path.dirname(__file__), path_like)),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.abspath(path_like)


def _subject_group_mapping(participants_tsv_path):
    df = pd.read_csv(participants_tsv_path, sep="\t")
    mapping = {}
    for _, row in df.iterrows():
        participant_id = str(row["participant_id"])
        if participant_id.startswith("sub-"):
            subject_id = participant_id.replace("sub-", "").zfill(2)
            mapping[subject_id] = row["Group"]
    return mapping


def _label_mapping_for_classes(classes):
    if classes == ("all",):
        return {"N": 0, "A+P-": 1, "A+P+": 2}
    if len(classes) == 2:
        return {classes[0]: 0, classes[1]: 1}
    if len(classes) == 3:
        return {classes[0]: 0, classes[1]: 1, classes[2]: 1}
    raise ValueError(f"Unsupported classes: {classes}")


def _resolve_eeg_subject_file(eeg_data_dir, subject_id):
    subject_str = str(subject_id).zfill(2)
    candidates = [os.path.join(eeg_data_dir, f"{int(subject_str)}.npz"), os.path.join(eeg_data_dir, f"{subject_str}.npz")]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def extract_eeg_window_level_features(eeg_data_dir, common_subjects, participants_tsv_path, classes):
    group_map = _subject_group_mapping(participants_tsv_path)
    label_mapping = _label_mapping_for_classes(classes)

    X, y, sub_ids = [], [], []
    for subject_id in sorted(common_subjects):
        group = group_map.get(subject_id)
        if group is None:
            continue
        if classes != ("all",) and group not in classes:
            continue

        eeg_path = _resolve_eeg_subject_file(eeg_data_dir, subject_id)
        if eeg_path is None:
            continue

        eeg_data = np.load(eeg_path, allow_pickle=True)
        if "HFD_features" not in eeg_data:
            raise KeyError(f"'HFD_features' not found in EEG file: {eeg_path}")

        # Match the old EEG permutation workflow: use per-window HFD features.
        hfd_features = np.asarray(eeg_data["HFD_features"], dtype=np.float32)
        flattened_windows = hfd_features.reshape(hfd_features.shape[0], -1)

        X.append(flattened_windows)
        y.append(np.full(flattened_windows.shape[0], label_mapping[group], dtype=int))
        sub_ids.extend([subject_id] * flattened_windows.shape[0])

    if len(X) == 0:
        return np.array([]), np.array([]), np.array([])

    return np.concatenate(X, axis=0), np.concatenate(y, axis=0), np.asarray(sub_ids)


def _log(verbose, message):
    if verbose:
        print(message)


def prepare_multimodal_data(classes, verbose=True):
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes.")

    participants_tsv = _resolve_participants_tsv()
    eeg_data_dir = _resolve_path(EEG_DATA_DIR)
    filtered_eeg, filtered_fmri, filtered_smri, filtered_psychometric, common_subjects = load_and_filter_all_modalities(
        eeg_data_dir,
        _resolve_path(FMRI_DATA_DIR),
        _resolve_path(SMRI_DATA_DIR),
        participants_tsv,
        verbose=verbose,
    )
    # Fixed subject order: common_subjects is a set, whose iteration order changes between runs
    # (string hashing), and k-means++ initialization depends on the order of the windows.
    common_subjects = sorted(common_subjects)

    X_eeg, y_eeg, sub_ids_eeg = extract_eeg_window_level_features(
        eeg_data_dir, common_subjects, participants_tsv, classes
    )
    X_fmri, y_fmri, sub_ids_fmri = get_X_y(filtered_fmri, common_subjects, modality="fMRI", classes=classes)
    X_smri, y_smri, sub_ids_smri = get_X_y(filtered_smri, common_subjects, modality="sMRI", classes=classes)
    X_psychometric, y_psychometric, sub_ids_psychometric = extract_psychometric_features(
        filtered_psychometric, common_subjects, classes=classes
    )

    multimodal_data = {
        "eeg": {"X": X_eeg, "y": _as_int_labels(y_eeg), "sub_ids": sub_ids_eeg},
        "fmri": {"X": X_fmri, "y": _as_int_labels(y_fmri), "sub_ids": sub_ids_fmri},
        "smri": {"X": X_smri, "y": _as_int_labels(y_smri), "sub_ids": sub_ids_smri},
        "psychometric": {
            "X": X_psychometric,
            "y": _as_int_labels(y_psychometric),
            "sub_ids": sub_ids_psychometric,
        },
    }

    for key in ["eeg", "fmri", "smri", "psychometric"]:
        sort_idx = multimodal_data[key]["sub_ids"].argsort(kind="stable")  # keeps each subject's windows in time order
        multimodal_data[key]["sub_ids"] = multimodal_data[key]["sub_ids"][sort_idx]
        multimodal_data[key]["X"] = multimodal_data[key]["X"][sort_idx]
        multimodal_data[key]["y"] = _as_int_labels(multimodal_data[key]["y"][sort_idx])

    unique_subjects = multimodal_data["smri"]["sub_ids"]
    subject_labels = multimodal_data["smri"]["y"]

    loo = LeaveOneOut()
    dummy_X = np.zeros((len(unique_subjects), 1))

    fold_splits = []
    for train_idx, test_idx in loo.split(dummy_X, subject_labels):
        fold_splits.append((unique_subjects[train_idx], unique_subjects[test_idx]))

    return multimodal_data, fold_splits


def build_task_context(classes, verbose=True):
    multimodal_data, fold_splits = prepare_multimodal_data(classes, verbose=verbose)

    folds = []
    for fold_index, (train_subjects, test_subjects) in enumerate(fold_splits, start=1):
        fold_entry = {
            "fold_index": fold_index,
            "train_subjects": train_subjects,
            "test_subjects": test_subjects,
        }

        for modality_key, modality_data in multimodal_data.items():
            train_idx = get_indices_for_subjects(modality_data["sub_ids"], train_subjects)
            test_idx = get_indices_for_subjects(modality_data["sub_ids"], test_subjects)

            fold_entry[modality_key] = {
                "X_train": modality_data["X"][train_idx],
                "y_train": _as_int_labels(modality_data["y"][train_idx]),
                "X_test": modality_data["X"][test_idx],
                "y_test": _as_int_labels(modality_data["y"][test_idx]),
                "train_sub_ids": modality_data["sub_ids"][train_idx],
                "test_sub_ids": modality_data["sub_ids"][test_idx],
            }

        fold_entry["true_labels"] = fold_entry["smri"]["y_test"]
        folds.append(fold_entry)

    return {
        "classes": tuple(classes),
        "multimodal_data": multimodal_data,
        "folds": folds,
        "subject_ids": multimodal_data["smri"]["sub_ids"],
        "subject_labels": _as_int_labels(multimodal_data["smri"]["y"]),
    }


def test_normality_and_variance(groups):
    if len(groups) < 2:
        return False
    if any(len(g) < MIN_SAMPLE_SIZE for g in groups):
        return False

    normality_pvals = [shapiro(g)[1] if len(g) >= 3 else 0.0 for g in groups]
    if not all(p > NORMALITY_ALPHA for p in normality_pvals):
        return False

    try:
        return levene(*groups)[1] > VARIANCE_ALPHA
    except Exception:
        return False


def choose_statistical_test_adaptive(groups):
    groups = [np.asarray(g) for g in groups if len(g) > 0]
    if len(groups) < 2:
        return 1.0

    pooled = np.concatenate(groups)
    if np.std(pooled) <= 1e-10:
        return 1.0

    try:
        if test_normality_and_variance(groups):
            _, p_val = ttest_ind(*groups) if len(groups) == 2 else f_oneway(*groups)
        else:
            _, p_val = mannwhitneyu(*groups) if len(groups) == 2 else kruskal(*groups)
        return float(p_val)
    except Exception:
        return 1.0


def compute_pvalue_weights(p_values, smoothing=P_VALUE_SMOOTHING, power=WEIGHT_POWER):
    p_values = np.asarray(p_values, dtype=float) + smoothing
    inverse_p = (1.0 / p_values) ** power
    total = inverse_p.sum()
    if total <= 0 or not np.isfinite(total):
        return np.ones_like(inverse_p) / len(inverse_p)
    return inverse_p / total


def weighted_prediction(probabilities, weights):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)
    weighted_prob = np.average(probabilities, axis=1, weights=weights)
    return (weighted_prob > 0.5).astype(int), weighted_prob


def compute_subject_dwell_times(X_data, y_data, sub_ids_data, target_subjects, kmeans_model, k_states, step_sec):
    subject_features, subject_labels = [], []

    for subject in target_subjects:
        mask = sub_ids_data == subject
        if not np.any(mask):
            continue
        subject_windows = X_data[mask]
        subject_states = kmeans_model.predict(subject_windows)
        dwell_times = np.bincount(subject_states, minlength=k_states) * step_sec
        subject_features.append(dwell_times)
        subject_labels.append(y_data[mask][0])

    return np.asarray(subject_features), _as_int_labels(subject_labels)


def compute_eeg_subject_features(X_data, y_data, sub_ids_data, target_subjects, kmeans_model, k_states, step_sec):
    subject_features, subject_labels = [], []

    for subject in target_subjects:
        mask = sub_ids_data == subject
        if not np.any(mask):
            continue
        subject_eeg = X_data[mask]
        subject_eeg_flat = subject_eeg.reshape(subject_eeg.shape[0], -1)
        subject_states = kmeans_model.predict(subject_eeg_flat)
        dwell_times = np.bincount(subject_states, minlength=k_states) * step_sec
        subject_features.append(dwell_times)
        subject_labels.append(y_data[mask][0])

    return np.asarray(subject_features), _as_int_labels(subject_labels)


# A participant's EEG value for a state is the SD of the distances of their windows in that state
# to its centroid. With a single window the SD is 0, which is not a measured variability, so a
# state needs at least this many windows; otherwise the value is missing (as with no window).
EEG_MIN_STATE_WINDOWS = 2


def compute_eeg_std_distance_features(X_data, y_data, sub_ids_data, target_subjects, kmeans_model, k_states):
    centroids = kmeans_model.cluster_centers_
    subject_features, subject_labels = [], []

    for subject in target_subjects:
        mask = sub_ids_data == subject
        if not np.any(mask):
            continue

        subject_windows = X_data[mask]
        subject_states = kmeans_model.predict(subject_windows)
        std_dist = np.full(k_states, np.nan, dtype=np.float32)

        for state in range(k_states):
            state_windows = subject_windows[subject_states == state]
            if state_windows.shape[0] >= EEG_MIN_STATE_WINDOWS:
                distances = np.linalg.norm(state_windows - centroids[state], axis=1)
                std_dist[state] = float(np.std(distances))

        subject_features.append(std_dist)
        subject_labels.append(y_data[mask][0])

    return np.asarray(subject_features, dtype=np.float32), _as_int_labels(subject_labels)


def _get_scale_pos_weight(y):
    y = _as_int_labels(y)
    counts = np.bincount(y, minlength=2)
    return counts[0] / counts[1] if counts[1] > 0 else 1.0


def _predict_xgb_binary(X_train, y_train, X_test, seed, device=None):
    y_train = _as_int_labels(y_train)
    if len(np.unique(y_train)) < 2:
        return np.full(X_test.shape[0], int(y_train[0]), dtype=int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kwargs = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=_get_scale_pos_weight(y_train),
        tree_method="hist",
        n_jobs=XGB_N_JOBS,
        verbosity=0,
    )
    device = XGB_DEVICE if device is None else device

    try:
        model = xgb.XGBClassifier(device=device, **kwargs)
        model.fit(X_train_scaled, y_train)
    except Exception:
        model = xgb.XGBClassifier(device="cpu", **kwargs)
        model.fit(X_train_scaled, y_train)

    prob = model.predict_proba(X_test_scaled)[:, 1]
    return (prob >= 0.5).astype(int)


def _prepare_binary_xgb_cache(X_train, y_train, X_test):
    y_train = _as_int_labels(y_train)
    test_size = X_test.shape[0]
    cache = {
        "y_train": y_train,
        "test_size": test_size,
        "majority": int(np.bincount(y_train).argmax()) if y_train.size else 0,
        "has_two_classes": len(np.unique(y_train)) >= 2 if y_train.size else False,
        "X_train_scaled": None,
        "X_test_scaled": None,
        "scale_pos_weight": 1.0,
    }

    if not cache["has_two_classes"]:
        return cache

    scaler = StandardScaler()
    cache["X_train_scaled"] = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    cache["X_test_scaled"] = scaler.transform(X_test).astype(np.float32, copy=False)
    cache["scale_pos_weight"] = _get_scale_pos_weight(y_train)
    return cache


def _predict_xgb_binary_from_cache(cache, seed, y_train_override=None, device=None):
    y_train = cache["y_train"] if y_train_override is None else _as_int_labels(y_train_override)

    if y_train.size == 0:
        return np.zeros(cache["test_size"], dtype=int)
    if not cache["has_two_classes"]:
        return np.full(cache["test_size"], cache["majority"], dtype=int)

    kwargs = dict(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=seed,
        scale_pos_weight=_get_scale_pos_weight(y_train),  # from the (possibly permuted) labels
        tree_method="hist",
        n_jobs=XGB_N_JOBS,
        verbosity=0,
    )
    device = XGB_DEVICE if device is None else device

    try:
        model = xgb.XGBClassifier(device=device, **kwargs)
        model.fit(cache["X_train_scaled"], y_train)
    except Exception:
        model = xgb.XGBClassifier(device="cpu", **kwargs)
        model.fit(cache["X_train_scaled"], y_train)

    prob = model.predict_proba(cache["X_test_scaled"])[:, 1]
    return (prob >= 0.5).astype(int)


def _build_state_feature_cache(
    X_train,
    y_train,
    X_test,
    train_sub_ids,
    test_sub_ids,
    train_subjects,
    test_subjects,
    seed,
    k_states_list,
    step_sec,
    subject_feature_fn,
    flatten_for_kmeans=False,
):
    y_train = _as_int_labels(y_train)
    cache = {
        "majority": int(np.bincount(y_train).argmax()) if y_train.size else 0,
        "test_size": len(test_subjects),
        "labels": np.array([], dtype=int),
        "y_train_subjects": np.array([], dtype=int),
        "features_by_k": {},
        "has_subject_classes": False,
    }

    if y_train.size == 0 or len(np.unique(y_train)) < 2:
        return cache

    kmeans_input = X_train.reshape(X_train.shape[0], -1) if flatten_for_kmeans else X_train
    y_train_subjects_ref = None

    for k_states in k_states_list:
        kmeans = KMeans(n_clusters=k_states, random_state=seed, n_init="auto")
        kmeans.fit(kmeans_input)

        X_train_subjects, y_train_subjects = subject_feature_fn(
            X_train,
            y_train,
            train_sub_ids,
            train_subjects,
            kmeans,
            k_states,
            step_sec,
        )
        X_test_subjects, _ = subject_feature_fn(
            X_test,
            np.zeros_like(test_sub_ids, dtype=int),
            test_sub_ids,
            test_subjects,
            kmeans,
            k_states,
            step_sec,
        )

        if X_train_subjects.size == 0 or X_test_subjects.size == 0:
            continue

        cache["features_by_k"][k_states] = {
            "X_train": np.asarray(X_train_subjects, dtype=np.float32),
            "X_test": np.asarray(X_test_subjects, dtype=np.float32),
        }

        if y_train_subjects_ref is None:
            y_train_subjects_ref = _as_int_labels(y_train_subjects)

    if y_train_subjects_ref is None:
        return cache

    cache["y_train_subjects"] = y_train_subjects_ref
    cache["labels"] = np.unique(y_train_subjects_ref)
    cache["has_subject_classes"] = len(cache["labels"]) >= 2
    return cache


def _build_eeg_cache(fold_entry, seed, verbose=False, total_folds=None):
    X_train = fold_entry["eeg"]["X_train"]
    y_train = fold_entry["eeg"]["y_train"]
    X_test = fold_entry["eeg"]["X_test"]
    fold_label = (
        f"{fold_entry['fold_index']}/{total_folds}"
        if total_folds is not None
        else str(fold_entry["fold_index"])
    )

    _log(verbose, f"Preparing EEG cache for fold {fold_label}")

    cache = {
        "mode": "empty",
        "majority": int(np.bincount(y_train).argmax()) if y_train.size else 0,
        "test_size": X_test.shape[0] if X_test.ndim > 0 else len(fold_entry["test_subjects"]),
    }

    if y_train.size == 0:
        return cache

    if X_train.ndim == 2:
        legacy_cache = {
            "majority": cache["majority"],
            "test_size": len(fold_entry["test_subjects"]),
            "y_train_subjects": np.array([], dtype=int),
            "features_by_k": {},
            "has_subject_classes": False,
        }

        y_train_subjects_ref = None
        for k_states in K_STATES_LIST_EEG:
            if X_train.shape[0] < k_states:
                continue

            try:
                kmeans = KMeans(n_clusters=k_states, random_state=seed, n_init=10, max_iter=300)
                kmeans.fit(X_train)
            except Exception:
                continue

            X_train_subjects, y_train_subjects = compute_eeg_std_distance_features(
                X_train,
                y_train,
                fold_entry["eeg"]["train_sub_ids"],
                fold_entry["train_subjects"],
                kmeans,
                k_states,
            )
            X_test_subjects, _ = compute_eeg_std_distance_features(
                X_test,
                np.zeros_like(fold_entry["eeg"]["test_sub_ids"], dtype=int),
                fold_entry["eeg"]["test_sub_ids"],
                fold_entry["test_subjects"],
                kmeans,
                k_states,
            )

            if X_train_subjects.size == 0 or X_test_subjects.size == 0:
                continue

            legacy_cache["features_by_k"][k_states] = {
                "X_train": X_train_subjects,
                "X_test": X_test_subjects,
            }

            if y_train_subjects_ref is None:
                y_train_subjects_ref = y_train_subjects

        cache.update(
            {
                "mode": "legacy_state",
                "legacy": legacy_cache,
            }
        )

        if y_train_subjects_ref is not None:
            cache["legacy"]["y_train_subjects"] = y_train_subjects_ref
            cache["legacy"]["has_subject_classes"] = len(np.unique(y_train_subjects_ref)) >= 2

        return cache

    if X_train.ndim == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        cache.update(
            {
                "mode": "direct",
                "binary": _prepare_binary_xgb_cache(X_train_flat, y_train, X_test_flat),
            }
        )
        return cache

    tril_i, tril_j = np.tril_indices(X_train.shape[1], k=-1)
    X_train_tri = X_train[:, tril_i, tril_j, :]
    X_test_tri = X_test[:, tril_i, tril_j, :]

    cache.update(
        {
            "mode": "state",
            "state": _build_state_feature_cache(
                X_train_tri,
                y_train,
                X_test_tri,
                fold_entry["eeg"]["train_sub_ids"],
                fold_entry["eeg"]["test_sub_ids"],
                fold_entry["train_subjects"],
                fold_entry["test_subjects"],
                seed,
                K_STATES_LIST_EEG,
                EEG_STEP_SEC,
                compute_eeg_subject_features,
                flatten_for_kmeans=True,
            ),
        }
    )
    return cache


def build_seed_fold_cache(task_context, seed, verbose=True):
    seed_fold_cache = []
    total_folds = len(task_context["folds"])

    for fold_entry in task_context["folds"]:
        _log(verbose, f"Building fold cache {fold_entry['fold_index']}/{total_folds} for seed {seed}")
        fold_cache = {
            "fold_index": fold_entry["fold_index"],
            "true_labels": fold_entry["true_labels"],
            "smri": _prepare_binary_xgb_cache(
                fold_entry["smri"]["X_train"],
                fold_entry["smri"]["y_train"],
                fold_entry["smri"]["X_test"],
            ),
            "psychometric": _prepare_binary_xgb_cache(
                fold_entry["psychometric"]["X_train"],
                fold_entry["psychometric"]["y_train"],
                fold_entry["psychometric"]["X_test"],
            ),
            "fmri": _build_state_feature_cache(
                fold_entry["fmri"]["X_train"],
                fold_entry["fmri"]["y_train"],
                fold_entry["fmri"]["X_test"],
                fold_entry["fmri"]["train_sub_ids"],
                fold_entry["fmri"]["test_sub_ids"],
                fold_entry["train_subjects"],
                fold_entry["test_subjects"],
                seed,
                K_STATES_LIST_FMRI,
                FMRI_STEP_SEC,
                compute_subject_dwell_times,
            ),
            "eeg": _build_eeg_cache(
                fold_entry,
                seed,
                verbose=verbose,
                total_folds=total_folds,
            ),
        }
        subject_ids = task_context["subject_ids"]
        subject_labels = task_context["subject_labels"]
        test_position = int(np.flatnonzero(subject_ids == fold_entry["test_subjects"][0])[0])
        fold_cache["test_position"] = test_position
        fold_cache["subject_labels"] = subject_labels
        # Every modality must list its training labels in subject order, so that one permuted
        # label vector can be applied to all of them.
        expected = np.delete(subject_labels, test_position)
        for name, labels in _training_label_arrays(fold_cache).items():
            assert np.array_equal(labels, expected), f"fold {fold_entry['fold_index']}: {name} labels out of order"
        seed_fold_cache.append(fold_cache)

    return seed_fold_cache


def _training_label_arrays(fold_cache):
    arrays = {
        "sMRI": fold_cache["smri"]["y_train"],
        "Psychometric": fold_cache["psychometric"]["y_train"],
        "fMRI": fold_cache["fmri"]["y_train_subjects"],
    }
    eeg_cache = fold_cache["eeg"]
    if eeg_cache["mode"] == "legacy_state":
        arrays["EEG"] = eeg_cache["legacy"]["y_train_subjects"]
    elif eeg_cache["mode"] == "state":
        arrays["EEG"] = eeg_cache["state"]["y_train_subjects"]
    return {name: _as_int_labels(a) for name, a in arrays.items() if len(a)}


def _predict_state_ensemble_from_cache(cache, y_train_subjects, seed):
    y_train_subjects = _as_int_labels(y_train_subjects)

    if not cache["features_by_k"] or not cache["has_subject_classes"]:
        return np.full(cache["test_size"], cache["majority"], dtype=int)

    per_k_probs = []
    per_k_pvals = []
    label_values = cache["labels"]

    for k_states, feature_data in cache["features_by_k"].items():
        Xtr = feature_data["X_train"]
        Xte = feature_data["X_test"]

        pvals = []
        for state in range(k_states):
            groups = [Xtr[y_train_subjects == label, state] for label in label_values]
            pvals.append(choose_statistical_test_adaptive(groups))

        best_state_idx = int(np.argmin(pvals))
        per_k_pvals.append(pvals[best_state_idx])

        clf = make_pipeline(
            MinMaxScaler(),
            LogisticRegression(penalty=None, class_weight="balanced", random_state=seed),
        )
        clf.fit(Xtr[:, [best_state_idx]], y_train_subjects)
        prob = clf.predict_proba(Xte[:, [best_state_idx]])[:, 1]
        per_k_probs.append(prob)

    if not per_k_probs:
        return np.full(cache["test_size"], cache["majority"], dtype=int)

    weights = compute_pvalue_weights(per_k_pvals)
    prob_matrix = np.column_stack(per_k_probs)
    pred, _ = weighted_prediction(prob_matrix, weights)
    return pred.astype(int)


def _predict_eeg_legacy_state_from_cache(cache, y_train_subjects, seed):
    y_train_subjects = _as_int_labels(y_train_subjects)

    if not cache["features_by_k"] or not cache["has_subject_classes"]:
        return np.full(cache["test_size"], cache["majority"], dtype=int)

    per_k_probs = []
    per_k_pvals = []

    for k_states, feature_data in cache["features_by_k"].items():
        Xtr = feature_data["X_train"]
        Xte = feature_data["X_test"]

        pvals = []
        for state in range(k_states):
            vals = Xtr[:, state]
            valid_mask = ~np.isnan(vals)
            if np.sum(valid_mask) < 2:
                pvals.append(1.0)
                continue
            groups = [vals[(y_train_subjects == label) & valid_mask] for label in np.unique(y_train_subjects)]
            pvals.append(choose_statistical_test_adaptive(groups))

        p_array = np.asarray(pvals, dtype=float)
        if p_array.size == 0 or np.all(np.isnan(p_array)) or np.all(p_array >= 1.0):
            continue

        best_state = int(np.nanargmin(p_array))
        best_p = float(p_array[best_state])
        train_vals = Xtr[:, best_state]
        valid_train_mask = ~np.isnan(train_vals)
        if np.sum(valid_train_mask) < 2:
            continue

        test_vals = Xte[:, best_state]
        if np.any(np.isnan(test_vals)):
            continue

        Xtr_state = train_vals[valid_train_mask].reshape(-1, 1)
        ytr = y_train_subjects[valid_train_mask]

        if len(np.unique(ytr)) < 2:
            prob = np.full(Xte.shape[0], 0.5, dtype=float)
        else:
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty=None, class_weight="balanced", random_state=seed),
            )
            clf.fit(Xtr_state, ytr)
            prob = clf.predict_proba(test_vals.reshape(-1, 1))[:, 1]

        per_k_probs.append(prob)
        per_k_pvals.append(best_p)

    if not per_k_probs:
        return np.full(cache["test_size"], cache["majority"], dtype=int)

    weights = compute_pvalue_weights(per_k_pvals)
    prob_matrix = np.column_stack(per_k_probs)
    pred, _ = weighted_prediction(prob_matrix, weights)
    return pred.astype(int)


def predict_smri_fold(X_train, y_train, X_test, seed):
    return _predict_xgb_binary(X_train, y_train, X_test, seed)


def predict_psychometric_fold(X_train, y_train, X_test, seed):
    return _predict_xgb_binary(X_train, y_train, X_test, seed)


def predict_fmri_fold(X_train, y_train, X_test, train_subjects, test_subjects, train_sub_ids, test_sub_ids, seed):
    y_train = _as_int_labels(y_train)
    majority = int(np.bincount(y_train).argmax())

    if len(np.unique(y_train)) < 2:
        return np.full(len(test_subjects), majority, dtype=int)

    X_train_subjects_all = {}
    X_test_subjects_all = {}
    y_train_subjects_ref = None

    for k_states in K_STATES_LIST_FMRI:
        kmeans = KMeans(n_clusters=k_states, random_state=seed, n_init="auto")
        kmeans.fit(X_train)

        X_train_subjects, y_train_subjects = compute_subject_dwell_times(
            X_train, y_train, train_sub_ids, train_subjects, kmeans, k_states, FMRI_STEP_SEC
        )
        X_test_subjects, _ = compute_subject_dwell_times(
            X_test, np.zeros_like(test_sub_ids, dtype=int), test_sub_ids, test_subjects, kmeans, k_states, FMRI_STEP_SEC
        )

        if X_train_subjects.size == 0 or X_test_subjects.size == 0:
            continue

        X_train_subjects_all[k_states] = X_train_subjects
        X_test_subjects_all[k_states] = X_test_subjects

        if y_train_subjects_ref is None:
            y_train_subjects_ref = y_train_subjects.copy()

    if not X_train_subjects_all:
        return np.full(len(test_subjects), majority, dtype=int)

    per_k_probs = []
    per_k_pvals = []

    for k_states in X_train_subjects_all:
        Xtr = X_train_subjects_all[k_states]
        Xte = X_test_subjects_all[k_states]

        pvals = []
        for state in range(k_states):
            groups = [Xtr[y_train_subjects_ref == label, state] for label in np.unique(y_train_subjects_ref)]
            pvals.append(choose_statistical_test_adaptive(groups))

        best_state_idx = int(np.argmin(pvals))
        per_k_pvals.append(pvals[best_state_idx])

        if len(np.unique(y_train_subjects_ref)) < 2:
            prob = np.full(Xte.shape[0], 0.5)
        else:
            clf = make_pipeline(
                MinMaxScaler(),
                LogisticRegression(penalty=None, class_weight="balanced", random_state=seed),
            )
            clf.fit(Xtr[:, [best_state_idx]], y_train_subjects_ref)
            prob = clf.predict_proba(Xte[:, [best_state_idx]])[:, 1]

        per_k_probs.append(prob)

    if not per_k_probs:
        return np.full(len(test_subjects), majority, dtype=int)

    weights = compute_pvalue_weights(per_k_pvals)
    prob_matrix = np.column_stack(per_k_probs)
    pred, _ = weighted_prediction(prob_matrix, weights)
    return pred.astype(int)


def predict_eeg_fold(X_train, y_train, X_test, train_subjects, test_subjects, train_sub_ids, test_sub_ids, seed):
    y_train = _as_int_labels(y_train)
    if y_train.size == 0:
        return np.zeros(len(test_subjects), dtype=int)
    majority = int(np.bincount(y_train).argmax())

    # Current HFD/PSD EEG data are subject-level arrays: (subjects, channels, features).
    # Handle those directly with a standard classifier.
    if X_train.ndim == 3:
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        return _predict_xgb_binary(X_train_flat, y_train, X_test_flat, seed)

    if len(np.unique(y_train)) < 2:
        return np.full(len(test_subjects), majority, dtype=int)

    tril_i, tril_j = np.tril_indices(X_train.shape[1], k=-1)
    X_train = X_train[:, tril_i, tril_j, :]
    X_test = X_test[:, tril_i, tril_j, :]

    X_train_subjects_all = {}
    X_test_subjects_all = {}
    y_train_subjects_ref = None

    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    for k_states in K_STATES_LIST_EEG:
        kmeans = KMeans(n_clusters=k_states, random_state=seed, n_init="auto")
        kmeans.fit(X_train_flat)

        X_train_subjects, y_train_subjects = compute_eeg_subject_features(
            X_train, y_train, train_sub_ids, train_subjects, kmeans, k_states, EEG_STEP_SEC
        )
        X_test_subjects, _ = compute_eeg_subject_features(
            X_test, np.zeros_like(test_sub_ids, dtype=int), test_sub_ids, test_subjects, kmeans, k_states, EEG_STEP_SEC
        )

        if X_train_subjects.size == 0 or X_test_subjects.size == 0:
            continue

        X_train_subjects_all[k_states] = X_train_subjects
        X_test_subjects_all[k_states] = X_test_subjects

        if y_train_subjects_ref is None:
            y_train_subjects_ref = y_train_subjects.copy()

    if not X_train_subjects_all:
        return np.full(len(test_subjects), majority, dtype=int)

    per_k_probs = []
    per_k_pvals = []

    for k_states in X_train_subjects_all:
        Xtr = X_train_subjects_all[k_states]
        Xte = X_test_subjects_all[k_states]

        pvals = []
        for state in range(k_states):
            groups = [Xtr[y_train_subjects_ref == label, state] for label in np.unique(y_train_subjects_ref)]
            pvals.append(choose_statistical_test_adaptive(groups))

        best_state_idx = int(np.argmin(pvals))
        per_k_pvals.append(pvals[best_state_idx])

        if len(np.unique(y_train_subjects_ref)) < 2:
            prob = np.full(Xte.shape[0], 0.5)
        else:
            clf = make_pipeline(
                MinMaxScaler(),
                LogisticRegression(penalty=None, class_weight="balanced", random_state=seed),
            )
            clf.fit(Xtr[:, [best_state_idx]], y_train_subjects_ref)
            prob = clf.predict_proba(Xte[:, [best_state_idx]])[:, 1]

        per_k_probs.append(prob)

    if not per_k_probs:
        return np.full(len(test_subjects), majority, dtype=int)

    weights = compute_pvalue_weights(per_k_pvals)
    prob_matrix = np.column_stack(per_k_probs)
    pred, _ = weighted_prediction(prob_matrix, weights)
    return pred.astype(int)


def run_observed_analysis(seed_val, classes, task_context=None, seed_fold_cache=None, verbose=True):
    set_all_seeds(seed_val)
    if task_context is None:
        task_context = build_task_context(classes, verbose=verbose)
    if seed_fold_cache is None:
        seed_fold_cache = build_seed_fold_cache(task_context, seed_val)

    observed_predictions = {mod: [] for mod in MODALITIES}
    true_labels = []

    for fold_cache in seed_fold_cache:
        _log(verbose, f"Observed fold {fold_cache['fold_index']}/{len(seed_fold_cache)}")
        true_labels.extend(fold_cache["true_labels"].tolist())

        pred_smri = _predict_xgb_binary_from_cache(fold_cache["smri"], seed_val)
        pred_psych = _predict_xgb_binary_from_cache(fold_cache["psychometric"], seed_val)
        pred_fmri = _predict_state_ensemble_from_cache(
            fold_cache["fmri"],
            fold_cache["fmri"]["y_train_subjects"],
            seed_val,
        )

        if fold_cache["eeg"]["mode"] == "empty":
            pred_eeg = np.zeros(fold_cache["eeg"]["test_size"], dtype=int)
        elif fold_cache["eeg"]["mode"] == "legacy_state":
            pred_eeg = _predict_eeg_legacy_state_from_cache(
                fold_cache["eeg"]["legacy"],
                fold_cache["eeg"]["legacy"]["y_train_subjects"],
                seed_val,
            )
        elif fold_cache["eeg"]["mode"] == "direct":
            pred_eeg = _predict_xgb_binary_from_cache(fold_cache["eeg"]["binary"], seed_val)
        else:
            pred_eeg = _predict_state_ensemble_from_cache(
                fold_cache["eeg"]["state"],
                fold_cache["eeg"]["state"]["y_train_subjects"],
                seed_val,
            )

        observed_predictions["sMRI"].extend(pred_smri.tolist())
        observed_predictions["Psychometric"].extend(pred_psych.tolist())
        observed_predictions["fMRI"].extend(pred_fmri.tolist())
        observed_predictions["EEG"].extend(pred_eeg.tolist())

    observed_f1_results = {
        mod: float(f1_score(true_labels, observed_predictions[mod], average="macro", zero_division=0))
        for mod in MODALITIES
    }

    return {
        "true_labels": true_labels,
        "predictions": observed_predictions,
        "observed_f1_results": observed_f1_results,
    }


def _init_permutation_worker(seed_fold_cache, seed_val, xgb_device):
    global _PERM_WORKER_FOLD_CACHE, _PERM_WORKER_SEED_VAL, _PERM_WORKER_XGB_DEVICE
    _PERM_WORKER_FOLD_CACHE = seed_fold_cache
    _PERM_WORKER_SEED_VAL = seed_val
    _PERM_WORKER_XGB_DEVICE = xgb_device


def _run_single_permutation_with_cache(seed_val, perm_idx, seed_fold_cache, xgb_device):
    with threadpool_limits(limits=1):
        rng = np.random.default_rng(seed_val * 1_000_003 + perm_idx)
        permutation_predictions = {mod: [] for mod in MODALITIES}
        permutation_true_labels = []
        model_seed_base = seed_val * 100_000 + perm_idx
        full = PERM_SCHEME == "full"
        if full:
            y_perm = rng.permutation(seed_fold_cache[0]["subject_labels"])

        def train_labels(original):
            """Permuted training labels for one fold and modality."""
            if full:
                return np.delete(y_perm, fold_cache["test_position"])
            return rng.permutation(original)

        for fold_idx, fold_cache in enumerate(seed_fold_cache):
            if full:
                permutation_true_labels.append(int(y_perm[fold_cache["test_position"]]))
            else:
                permutation_true_labels.extend(fold_cache["true_labels"].tolist())

            smri_cache = fold_cache["smri"]
            psych_cache = fold_cache["psychometric"]
            fmri_cache = fold_cache["fmri"]
            eeg_cache = fold_cache["eeg"]

            y_train_perm = train_labels(smri_cache["y_train"])
            y_pred_perm = _predict_xgb_binary_from_cache(
                smri_cache,
                model_seed_base + fold_idx,
                y_train_perm,
                device=xgb_device,
            )
            permutation_predictions["sMRI"].extend(y_pred_perm.tolist())

            y_train_perm = train_labels(psych_cache["y_train"])
            y_pred_perm = _predict_xgb_binary_from_cache(
                psych_cache,
                model_seed_base + 10_000 + fold_idx,
                y_train_perm,
                device=xgb_device,
            )
            permutation_predictions["Psychometric"].extend(y_pred_perm.tolist())

            y_train_perm = train_labels(fmri_cache["y_train_subjects"])
            perm_pred = _predict_state_ensemble_from_cache(
                fmri_cache,
                y_train_perm,
                model_seed_base + 20_000 + fold_idx,
            )
            permutation_predictions["fMRI"].extend(np.asarray(perm_pred, dtype=int).tolist())

            if eeg_cache["mode"] == "empty":
                permutation_predictions["EEG"].extend(np.zeros(eeg_cache["test_size"], dtype=int).tolist())
            elif eeg_cache["mode"] == "legacy_state":
                eeg_legacy_cache = eeg_cache["legacy"]
                y_train_perm = train_labels(eeg_legacy_cache["y_train_subjects"])
                perm_pred = _predict_eeg_legacy_state_from_cache(
                    eeg_legacy_cache,
                    y_train_perm,
                    model_seed_base + 30_000 + fold_idx,
                )
                permutation_predictions["EEG"].extend(np.asarray(perm_pred, dtype=int).tolist())
            elif eeg_cache["mode"] == "direct":
                eeg_binary_cache = eeg_cache["binary"]
                if full:
                    raise NotImplementedError("full permutation is not implemented for the window-level EEG mode")
                y_train_perm = rng.permutation(eeg_binary_cache["y_train"])
                y_pred_perm = _predict_xgb_binary_from_cache(
                    eeg_binary_cache,
                    model_seed_base + 40_000 + fold_idx,
                    y_train_perm,
                    device=xgb_device,
                )
                permutation_predictions["EEG"].extend(y_pred_perm.tolist())
            else:
                eeg_state_cache = eeg_cache["state"]
                y_train_perm = train_labels(eeg_state_cache["y_train_subjects"])
                perm_pred = _predict_state_ensemble_from_cache(
                    eeg_state_cache,
                    y_train_perm,
                    model_seed_base + 50_000 + fold_idx,
                )
                permutation_predictions["EEG"].extend(np.asarray(perm_pred, dtype=int).tolist())

        permutation_f1_results = {}
        for mod in MODALITIES:
            preds = np.asarray(permutation_predictions[mod], dtype=int)
            f1 = f1_score(permutation_true_labels, preds, average="macro", zero_division=0)
            permutation_f1_results[mod] = float(f1)

        return permutation_f1_results


def _run_single_permutation_job(perm_idx):
    return _run_single_permutation_with_cache(
        seed_val=_PERM_WORKER_SEED_VAL,
        perm_idx=perm_idx,
        seed_fold_cache=_PERM_WORKER_FOLD_CACHE,
        xgb_device=_PERM_WORKER_XGB_DEVICE,
    )


def run_permutation_analysis_from_scratch(
    seed_val,
    classes,
    n_permutations=1000,
    task_context=None,
    seed_fold_cache=None,
    n_jobs=1,
    verbose=True,
):
    set_all_seeds(seed_val)
    if task_context is None:
        task_context = build_task_context(classes, verbose=verbose)
    if seed_fold_cache is None:
        seed_fold_cache = build_seed_fold_cache(task_context, seed_val)
    n_jobs = min(_normalize_jobs(n_jobs), n_permutations)
    perm_xgb_device = "cpu" if n_jobs > 1 else XGB_DEVICE
    _log(
        verbose,
        f"Running {n_permutations} permutations for seed {seed_val} with jobs={n_jobs} and XGBoost device={perm_xgb_device}",
    )

    if n_jobs == 1:
        permutation_results = [
            _run_single_permutation_with_cache(
                seed_val=seed_val,
                perm_idx=perm_idx,
                seed_fold_cache=seed_fold_cache,
                xgb_device=perm_xgb_device,
            )
            for perm_idx in range(n_permutations)
        ]
    else:
        mp_context = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")
        chunksize = max(1, n_permutations // (n_jobs * 4))
        with ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp_context,
            initializer=_init_permutation_worker,
            initargs=(seed_fold_cache, seed_val, perm_xgb_device),
        ) as executor:
            permutation_results = list(
                executor.map(_run_single_permutation_job, range(n_permutations), chunksize=chunksize)
            )

    permutation_f1_results = {
        mod: [float(result[mod]) for result in permutation_results]
        for mod in MODALITIES
    }

    return {
        "permutation_true_labels": None,
        "permutation_predictions": None,
        "permutation_f1_results": permutation_f1_results,
    }


def run_f1_pipeline_from_scratch(
    classes,
    seeds=(9, 42, 123, 256, 512),
    permutation_seeds=None,
    n_permutations=1000,
    n_jobs=1,
    verbose=True,
):
    observed_seeds = _normalize_seeds(seeds)
    permutation_seeds = observed_seeds if permutation_seeds is None else _normalize_seeds(permutation_seeds)
    task_context = build_task_context(classes, verbose=verbose)
    observed_runs = {mod: [] for mod in MODALITIES}
    permutation_runs = {mod: [] for mod in MODALITIES}
    seed_fold_cache_by_seed = {}

    def get_seed_fold_cache(seed):
        if seed not in seed_fold_cache_by_seed:
            seed_fold_cache_by_seed[seed] = build_seed_fold_cache(task_context, seed, verbose=verbose)
        return seed_fold_cache_by_seed[seed]

    for seed in observed_seeds:
        seed_fold_cache = get_seed_fold_cache(seed)
        _log(verbose, "\n" + "=" * 80)
        _log(verbose, f"SEED {seed}: OBSERVED")
        _log(verbose, "=" * 80)
        obs = run_observed_analysis(
            seed,
            classes,
            task_context=task_context,
            seed_fold_cache=seed_fold_cache,
            verbose=verbose,
        )

        for mod in MODALITIES:
            observed_runs[mod].append(obs["observed_f1_results"][mod])

    for seed in permutation_seeds:
        seed_fold_cache = get_seed_fold_cache(seed)
        _log(verbose, "\n" + "=" * 80)
        _log(verbose, f"SEED {seed}: PERMUTATION")
        _log(verbose, "=" * 80)
        perm = run_permutation_analysis_from_scratch(
            seed_val=seed,
            classes=classes,
            n_permutations=n_permutations,
            task_context=task_context,
            seed_fold_cache=seed_fold_cache,
            n_jobs=n_jobs,
            verbose=verbose,
        )

        for mod in MODALITIES:
            permutation_runs[mod].append(perm["permutation_f1_results"][mod])

    summary_rows = []

    for mod in MODALITIES:
        obs_vals = np.asarray(observed_runs[mod], dtype=float)
        perm_matrix = np.asarray(permutation_runs[mod], dtype=float)
        perm_mean_dist = perm_matrix.mean(axis=0)

        observed_mean = float(obs_vals.mean())
        observed_std = float(np.std(obs_vals, ddof=1)) if len(obs_vals) > 1 else 0.0

        perm_mean = float(perm_mean_dist.mean())
        perm_std = float(np.std(perm_mean_dist, ddof=1)) if len(perm_mean_dist) > 1 else 0.0

        p_value = float((np.sum(perm_mean_dist >= observed_mean) + 1) / (len(perm_mean_dist) + 1))
        effect_size = float((observed_mean - perm_mean) / perm_std) if perm_std > 0 else 0.0

        summary_rows.append(
            {
                "Modality": mod,
                "Observed_Mean_F1": observed_mean,
                "Observed_Std_F1": observed_std,
                "Observed_N_Runs": len(observed_seeds),
                "Permutation_Mean_F1": perm_mean,
                "Permutation_Std_F1": perm_std,
                "Permutation_N_Runs": len(permutation_seeds),
                "p_value": p_value,
                "Effect_Size": effect_size,
                "Significant": p_value < 0.05,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    _log(verbose, "\n" + "=" * 120)
    _log(
        verbose,
        f"FINAL F1-MACRO SUMMARY (observed runs={len(observed_seeds)}, permutation runs={len(permutation_seeds)})",
    )
    _log(verbose, "=" * 120)
    display_df = summary_df.copy()
    for col in [
        "Observed_Mean_F1",
        "Observed_Std_F1",
        "Permutation_Mean_F1",
        "Permutation_Std_F1",
        "p_value",
        "Effect_Size",
    ]:
        display_df[col] = display_df[col].map(lambda x: f"{x:.4f}")
    display_df["Significance"] = display_df["Significant"].map(lambda x: "***" if x else "ns")
    _log(
        verbose,
        display_df[
            [
                "Modality",
                "Observed_Mean_F1",
                "Observed_Std_F1",
                "Observed_N_Runs",
                "Permutation_Mean_F1",
                "Permutation_Std_F1",
                "Permutation_N_Runs",
                "p_value",
                "Significance",
                "Effect_Size",
            ]
        ].to_string(index=False)
    )

    return summary_df, {
        "observed_runs": observed_runs,
        "permutation_runs": permutation_runs,
        "observed_seeds": observed_seeds,
        "permutation_seeds": permutation_seeds,
    }


def p_marker(p_value):
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    if p_value < 0.10:
        return "t"
    return ""


def build_publication_f1_table(
    seeds=(9, 42, 123, 256, 512),
    permutation_seeds=None,
    n_permutations=1000,
    n_jobs=1,
    verbose=True,
    tasks=None,
):
    observed_seeds = _normalize_seeds(seeds)
    permutation_seeds = observed_seeds if permutation_seeds is None else _normalize_seeds(permutation_seeds)
    all_rows = []
    raw_results = {}

    for task in (TASKS if tasks is None else [TASKS[i - 1] for i in tasks]):
        task_label = TASK_LABELS[task]

        _log(verbose, "\n" + "=" * 80)
        _log(verbose, f"RUNNING: {task_label}")
        _log(verbose, "=" * 80)

        summary_df, raw = run_f1_pipeline_from_scratch(
            classes=task,
            seeds=observed_seeds,
            permutation_seeds=permutation_seeds,
            n_permutations=n_permutations,
            n_jobs=n_jobs,
            verbose=verbose,
        )
        raw_results[task_label] = raw

        df = summary_df.copy()
        df["Group Comparison"] = task_label
        df["Modality"] = df["Modality"].map(MODALITY_LABELS).fillna(df["Modality"])
        df["Observed F1"] = df.apply(
            lambda row: f'{row["Observed_Mean_F1"]:.3f} +/- {row["Observed_Std_F1"]:.3f}',
            axis=1,
        )
        df["p-value"] = df["p_value"].apply(lambda value: f"{value:.3f}{p_marker(value)}")
        df["Observed_Mean_F1_raw"] = df["Observed_Mean_F1"]
        df["Null_Mean_F1_raw"] = df["Permutation_Mean_F1"]
        df["Null_SD_F1_raw"] = df["Permutation_Std_F1"]
        df["p_value_raw"] = df["p_value"]
        df["Observed_N_Runs_raw"] = df["Observed_N_Runs"]
        df["Permutation_N_Runs_raw"] = df["Permutation_N_Runs"]

        all_rows.append(
            df[
                [
                    "Group Comparison",
                    "Modality",
                    "Observed F1",
                    "p-value",
                    "Observed_N_Runs_raw",
                    "Permutation_N_Runs_raw",
                    "Observed_Mean_F1_raw",
                    "p_value_raw",
                    "Null_Mean_F1_raw",
                    "Null_SD_F1_raw",
                ]
            ]
        )

    table_df, pretty_df = _publication_tables(pd.concat(all_rows, ignore_index=True))
    return table_df, pretty_df, raw_results


def _publication_tables(table_df):
    """Sort the raw rows in table order and build the pretty table."""
    table_df["Group Comparison"] = pd.Categorical(
        table_df["Group Comparison"],
        categories=TASK_ORDER,
        ordered=True,
    )
    table_df["Modality"] = pd.Categorical(
        table_df["Modality"],
        categories=MODALITY_ORDER,
        ordered=True,
    )
    table_df = table_df.sort_values(["Group Comparison", "Modality"]).reset_index(drop=True)

    pretty_df = table_df[["Group Comparison", "Modality", "Observed F1", "p-value"]].copy()
    pretty_df["Group Comparison"] = pretty_df["Group Comparison"].astype(str)
    pretty_df.loc[pretty_df["Group Comparison"].duplicated(), "Group Comparison"] = ""

    return table_df, pretty_df


def style_publication_table(table_df):
    display_df = table_df.copy()
    best_idx = display_df.groupby("Group Comparison")["Observed_Mean_F1_raw"].idxmax().tolist()

    def highlight_rows(row):
        styles = [""] * len(row)
        observed_col = display_df.columns.get_loc("Observed F1")
        p_col = display_df.columns.get_loc("p-value")

        if row.name in best_idx:
            styles[observed_col] = "font-weight: bold;"
        if row["p_value_raw"] < 0.10:
            styles[p_col] = "font-weight: bold;"
        return styles

    shown = display_df[["Group Comparison", "Modality", "Observed F1", "p-value"]].copy()
    shown["Group Comparison"] = shown["Group Comparison"].astype(str)
    shown.loc[shown["Group Comparison"].duplicated(), "Group Comparison"] = ""

    return shown.style.hide(axis="index").apply(highlight_rows, axis=1)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run matched F1 permutation analyses from scratch.")
    parser.add_argument("--classes", nargs="+", help="Class labels, e.g. N A+P-")
    parser.add_argument("--seeds", nargs="+", type=int, default=[9, 42])
    parser.add_argument(
        "--perm-seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional permutation seeds. If omitted, reuse --seeds.",
    )
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_PERM_JOBS,
        help="Number of parallel workers for permutation runs. Use 1 to disable parallelism.",
    )
    parser.add_argument("--table", action="store_true", help="Run all task comparisons and print a publication table.")
    parser.add_argument("--perm-scheme", choices=["full", "per-fold"], default="full",
                        help="Null construction (see PERM_SCHEME). Default: full.")
    parser.add_argument("--out-dir", default=None, help="Output folder (default: results/permutation).")
    parser.add_argument("--tasks", nargs="+", type=int, choices=[1, 2, 3, 4], default=None,
                        help="With --table, run only these comparisons: 1 N vs A+P-, 2 N vs A+P+, "
                             "3 A+P- vs A+P+, 4 N vs all carriers (to run them as parallel jobs).")
    parser.add_argument("--assemble", nargs="+", default=None, metavar="DIR",
                        help="Combine the raw tables written by --tasks runs in these folders into "
                             "--out-dir, then exit.")
    return parser.parse_args()


def main():
    global PERM_SCHEME, RESULTS_DIR
    args = _parse_args()
    PERM_SCHEME = args.perm_scheme
    if args.out_dir:
        RESULTS_DIR = os.path.abspath(args.out_dir)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Permutation scheme: {PERM_SCHEME}; output: {RESULTS_DIR}")

    if args.assemble:
        parts = [pd.read_csv(os.path.join(d, "publication_f1_permutation_table_raw.csv")) for d in args.assemble]
        table_df, pretty_df = _publication_tables(pd.concat(parts, ignore_index=True))
        table_df.to_csv(os.path.join(RESULTS_DIR, "publication_f1_permutation_table_raw.csv"), index=False)
        pretty_df.to_csv(os.path.join(RESULTS_DIR, "publication_f1_permutation_table_pretty.csv"), index=False)
        print(pretty_df.to_string(index=False))
        return

    if args.table:
        table_df, pretty_df, _ = build_publication_f1_table(
            seeds=tuple(args.seeds),
            permutation_seeds=None if args.perm_seeds is None else tuple(args.perm_seeds),
            n_permutations=args.permutations,
            n_jobs=args.jobs,
            tasks=args.tasks,
        )
        print(pretty_df.to_string(index=False))
        table_df.to_csv(os.path.join(RESULTS_DIR, "publication_f1_permutation_table_raw.csv"), index=False)
        pretty_df.to_csv(os.path.join(RESULTS_DIR, "publication_f1_permutation_table_pretty.csv"), index=False)
        return

    if not args.classes or len(args.classes) < 2:
        raise SystemExit("Provide at least 2 classes, e.g. --classes 'A+P-' 'A+P+'")

    summary_df, _ = run_f1_pipeline_from_scratch(
        classes=tuple(args.classes),
        seeds=tuple(args.seeds),
        permutation_seeds=None if args.perm_seeds is None else tuple(args.perm_seeds),
        n_permutations=args.permutations,
        n_jobs=args.jobs,
    )
    summary_df.to_csv(os.path.join(RESULTS_DIR, "matched_f1_permutation_summary.csv"), index=False)


if __name__ == "__main__":
    main()
