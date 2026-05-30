from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr

from .schema import assert_group_disjoint


def _safe_stratified_n_splits(y, requested: int) -> int:
    """Choose a valid number of stratified folds for imbalanced labels."""
    counts = pd.Series(y).value_counts(dropna=False)
    if len(counts) < 2:
        raise ValueError("Classification target has fewer than two classes.")
    max_valid = int(counts.min())
    if max_valid < 2:
        raise ValueError(
            "At least one class has fewer than two samples; I cannot run stratified cross-validation."
        )
    return max(2, min(int(requested), max_valid))


def classification_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{key}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{key}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def majority_class_baseline(y_true, prefix: str = "") -> tuple[dict[str, float], np.ndarray]:
    """Score an always-predict-majority classifier for imbalanced visual labels."""
    y_true = np.asarray(y_true)
    majority = pd.Series(y_true).mode(dropna=False).iloc[0]
    y_pred = np.full_like(y_true, majority)
    metrics = classification_metrics(y_true, y_pred, prefix=prefix)
    return metrics, y_pred


def shuffled_label_baseline(y_true, seed: int = 42, prefix: str = "") -> tuple[dict[str, float], np.ndarray]:
    """Score a label-shuffled null baseline with the same class frequencies."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = rng.permutation(y_true)
    metrics = classification_metrics(y_true, y_pred, prefix=prefix)
    return metrics, y_pred


def regression_metrics(y_true, y_pred, prefix: str = "") -> dict[str, float]:
    key = f"{prefix}_" if prefix else ""
    return {
        f"{key}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{key}rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{key}r2": float(r2_score(y_true, y_pred)),
    }


def grouped_stratified_splits(X, y, groups=None, n_splits: int = 5, seed: int = 42):
    """Generate leakage-aware splits. GroupKFold takes priority when groups are provided."""
    X = np.asarray(X)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        for train_idx, test_idx in splitter.split(X, y, groups):
            assert_group_disjoint(groups[train_idx], groups[test_idx])
            yield train_idx, test_idx
    else:
        valid_splits = _safe_stratified_n_splits(y, n_splits)
        splitter = StratifiedKFold(n_splits=valid_splits, shuffle=True, random_state=seed)
        yield from splitter.split(X, y)


def grouped_regression_splits(X, y, groups=None, n_splits: int = 5, seed: int = 42):
    """Generate leakage-aware splits for regression targets."""
    X = np.asarray(X)
    y = np.asarray(y)
    if groups is not None:
        groups = np.asarray(groups)
        splitter = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
        for train_idx, test_idx in splitter.split(X, y, groups):
            assert_group_disjoint(groups[train_idx], groups[test_idx])
            yield train_idx, test_idx
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        yield from splitter.split(X, y)


def save_confusion_matrix_table(y_true, y_pred, labels=None) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels]) if labels is not None else pd.DataFrame(cm)


def circular_shift_null_spearman(
    x,
    y,
    n_perm: int = 1000,
    min_shift: int = 60,
    seed: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Estimate a temporal circular-shift null for autocorrelated movie features.

    Natural movie frames are temporally autocorrelated, so ordinary frame-wise
    Spearman p-values are optimistic. I preserve the stimulus time-series
    structure by circularly shifting one variable and comparing the observed
    correlation to the shifted null distribution.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4:
        raise ValueError("I need at least four finite samples for a circular-shift null test.")

    observed, _ = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    n = len(y)
    min_shift = int(min(max(1, min_shift), max(1, n // 2 - 1)))
    high = max(min_shift + 1, n - min_shift)
    null = []
    for _ in range(int(n_perm)):
        shift = int(rng.integers(min_shift, high)) if high > min_shift else min_shift
        rho, _ = spearmanr(x, np.roll(y, shift))
        null.append(float(rho))
    null = np.asarray(null, dtype=float)
    p_empirical = float((np.sum(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1))
    return float(observed), p_empirical, null


def stimulus_neural_circular_shift_table(
    frame_features: pd.DataFrame,
    neural_col: str,
    stimulus_cols: list[str],
    n_perm: int = 1000,
    min_shift: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a circular-shift null table for neural-summary/stimulus-feature correlations."""
    rows = []
    for col in stimulus_cols:
        rho, p_emp, null = circular_shift_null_spearman(
            frame_features[neural_col],
            frame_features[col],
            n_perm=n_perm,
            min_shift=min_shift,
            seed=seed,
        )
        rows.append({
            "neural_feature": neural_col,
            "stimulus_feature": col,
            "spearman_r": rho,
            "circular_shift_p": p_emp,
            "null_mean": float(np.nanmean(null)),
            "null_std": float(np.nanstd(null)),
            "n_permutations": int(n_perm),
            "min_shift_frames": int(min_shift),
        })
    return pd.DataFrame(rows)


def bootstrap_mean(values, n_boot: int = 200, seed: int = 42) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def make_contiguous_frame_blocks(n_samples: int, n_blocks: int = 5) -> np.ndarray:
    """Assign movie frames to contiguous blocks for autocorrelation-aware CV.

    Natural movie frames are temporally autocorrelated. I use contiguous movie
    blocks as groups so that regression models are evaluated on held-out movie
    segments rather than random neighboring frames.
    """
    if n_blocks < 2:
        raise ValueError("I need at least two blocks for cross-validation.")
    if n_samples < 2:
        raise ValueError("I need at least two samples to build contiguous blocks.")

    n_blocks = min(int(n_blocks), int(n_samples))
    edges = np.linspace(0, n_samples, n_blocks + 1, dtype=int)
    groups = np.empty(n_samples, dtype=int)
    for block_id in range(n_blocks):
        groups[edges[block_id]: edges[block_id + 1]] = block_id
    return groups


def frame_time_features(frame_index: np.ndarray, n_harmonics: int = 5) -> np.ndarray:
    """Build linear and Fourier frame-index features as a temporal baseline."""
    frame_index = np.asarray(frame_index, dtype=float).reshape(-1)
    denominator = max(float(np.nanmax(frame_index)), 1.0)
    t = frame_index / denominator

    columns = [t]
    for k in range(1, int(n_harmonics) + 1):
        columns.append(np.sin(2.0 * np.pi * k * t))
        columns.append(np.cos(2.0 * np.pi * k * t))
    return np.column_stack(columns)


def _safe_spearman_for_regression(y_true, y_pred) -> float:
    """Return Spearman correlation or NaN when either vector is constant."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred, nan_policy="omit")
    return float(rho)


def evaluate_feature_regression(
    representations: dict[str, np.ndarray],
    feature_table: pd.DataFrame,
    targets: list[str],
    groups: np.ndarray,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """Evaluate whether representations predict continuous real movie features.

    This function is used for notebook 05 and script-based execution. It uses
    GroupKFold over contiguous movie-frame blocks so that temporally adjacent
    natural movie frames do not leak across train/test folds.
    """
    rows: list[dict[str, object]] = []
    groups = np.asarray(groups)

    if len(groups) != len(feature_table):
        raise ValueError(
            f"Group vector has length {len(groups)}, but feature table has {len(feature_table)} rows."
        )

    for representation_name, X in representations.items():
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if X.shape[0] != len(feature_table):
            raise ValueError(
                f"{representation_name} has {X.shape[0]} rows, "
                f"but the feature table has {len(feature_table)} rows."
            )

        for target in targets:
            if target not in feature_table.columns:
                continue

            y = pd.to_numeric(feature_table[target], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
            X_valid = X[valid]
            y_valid = y[valid]
            groups_valid = groups[valid]

            unique_groups = np.unique(groups_valid)
            if len(unique_groups) < 2:
                continue

            cv = GroupKFold(n_splits=len(unique_groups))
            fold_rows: list[dict[str, float]] = []

            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_valid, y_valid, groups_valid)):
                model = make_pipeline(StandardScaler(), Ridge(alpha=float(alpha)))
                model.fit(X_valid[train_idx], y_valid[train_idx])
                pred = model.predict(X_valid[test_idx])

                fold_rows.append({
                    "fold": float(fold_idx),
                    "r2": float(r2_score(y_valid[test_idx], pred)),
                    "mae": float(mean_absolute_error(y_valid[test_idx], pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_valid[test_idx], pred))),
                    "spearman_r": _safe_spearman_for_regression(y_valid[test_idx], pred),
                    "n_train": float(len(train_idx)),
                    "n_test": float(len(test_idx)),
                })

            fold_df = pd.DataFrame(fold_rows)
            rows.append({
                "representation": representation_name,
                "target": target,
                "cv_strategy": "contiguous_movie_block_groupkfold",
                "n_samples": int(valid.sum()),
                "n_features": int(X_valid.shape[1]),
                "mean_r2": float(fold_df["r2"].mean()),
                "std_r2": float(fold_df["r2"].std()),
                "mean_mae": float(fold_df["mae"].mean()),
                "std_mae": float(fold_df["mae"].std()),
                "mean_rmse": float(fold_df["rmse"].mean()),
                "std_rmse": float(fold_df["rmse"].std()),
                "mean_spearman_r": float(fold_df["spearman_r"].mean()),
                "std_spearman_r": float(fold_df["spearman_r"].std()),
            })

    return pd.DataFrame(rows)


def latent_gain_over_temporal_baseline(
    latent_vs_baseline: pd.DataFrame,
    temporal_representations: tuple[str, ...] = (
        "frame_index_linear_baseline",
        "frame_index_fourier_baseline",
    ),
    scalar_baseline: str = "population_l2_norm_baseline",
) -> pd.DataFrame:
    """Compare the best latent representation against the best temporal baseline."""
    is_temporal = latent_vs_baseline["representation"].isin(list(temporal_representations))

    best_temporal = (
        latent_vs_baseline[is_temporal]
        .sort_values(["target", "mean_r2"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
        .rename(columns={
            "representation": "best_temporal_baseline",
            "mean_r2": "best_temporal_mean_r2",
            "mean_mae": "best_temporal_mean_mae",
            "mean_spearman_r": "best_temporal_mean_spearman_r",
        })
    )

    best_latent = (
        latent_vs_baseline[
            (~is_temporal)
            & (latent_vs_baseline["representation"] != scalar_baseline)
        ]
        .sort_values(["target", "mean_r2"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
        .rename(columns={
            "representation": "best_latent_representation",
            "mean_r2": "best_latent_mean_r2",
            "mean_mae": "best_latent_mean_mae",
            "mean_spearman_r": "best_latent_mean_spearman_r",
        })
    )

    comparison = best_latent.merge(best_temporal, on="target", how="left")
    comparison["delta_r2_latent_minus_temporal"] = (
        comparison["best_latent_mean_r2"] - comparison["best_temporal_mean_r2"]
    )
    comparison["delta_spearman_latent_minus_temporal"] = (
        comparison["best_latent_mean_spearman_r"] - comparison["best_temporal_mean_spearman_r"]
    )
    return comparison
