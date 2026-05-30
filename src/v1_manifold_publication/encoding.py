from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.decomposition import PCA
from .splits import group_kfold_indices, contiguous_blocks
from .metrics import regression_row, safe_pearson


def ridge_population_encoding(stimulus_features, neural_response, groups=None, alphas=(0.1, 1, 10, 100), n_splits=5):
    """Predict neural population responses from stimulus features.

    stimulus_features: [frames, feature_dim]
    neural_response: [frames, cells] repeat-averaged or trial-level aligned with features
    """
    X = np.asarray(stimulus_features, dtype=float)
    Y = np.asarray(neural_response, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X, Y = X[mask], Y[mask]
    groups = np.asarray(groups)[mask] if groups is not None else contiguous_blocks(len(X), n_blocks=n_splits)
    rows = []
    preds = np.full_like(Y, np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(group_kfold_indices(groups, n_splits=n_splits)):
        model = MultiOutputRegressor(make_pipeline(StandardScaler(), RidgeCV(alphas=list(alphas))))
        model.fit(X[train_idx], Y[train_idx])
        pred = model.predict(X[test_idx])
        preds[test_idx] = pred
        population_metrics = regression_row(Y[test_idx].ravel(), pred.ravel(), prefix="population_")
        cell_rs = [safe_pearson(Y[test_idx, c], pred[:, c]) for c in range(Y.shape[1])]
        rows.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_cells": Y.shape[1],
            "mean_cell_pearson_r": float(np.nanmean(cell_rs)),
            "median_cell_pearson_r": float(np.nanmedian(cell_rs)),
            **population_metrics,
        })
    return pd.DataFrame(rows), preds


def feature_matrix_from_table(features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return features[feature_cols].to_numpy(dtype=float)


def noise_corrected_score(score: float, noise_ceiling: float | None) -> float:
    if noise_ceiling is None or not np.isfinite(noise_ceiling) or noise_ceiling <= 0:
        return np.nan
    return float(score / noise_ceiling)


def _population_encoding_estimator(model_name: str, alphas=(0.1, 1, 10, 100), random_state=42):
    if model_name == "ridge_cv":
        return MultiOutputRegressor(make_pipeline(StandardScaler(), RidgeCV(alphas=list(alphas))))
    if model_name == "elastic_net":
        return MultiOutputRegressor(make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9],
                alphas=np.logspace(-3, 1, 8),
                cv=3,
                random_state=random_state,
                max_iter=5000,
            ),
        ))
    if model_name == "kernel_ridge_rbf":
        return make_pipeline(StandardScaler(), KernelRidge(alpha=1.0, kernel="rbf"))
    if model_name == "mean_response_baseline":
        return None
    raise ValueError(f"Unknown encoding model: {model_name}")


def evaluate_population_encoding_cv(
    stimulus_features,
    neural_response,
    groups=None,
    model_name: str = "ridge_cv",
    alphas=(0.1, 1, 10, 100),
    n_splits=5,
    random_state=42,
    noise_ceiling: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate stimulus-to-population encoding with publication-grade folds."""
    X = np.asarray(stimulus_features, dtype=float)
    Y = np.asarray(neural_response, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    mask = np.all(np.isfinite(X), axis=1) & np.all(np.isfinite(Y), axis=1)
    X, Y = X[mask], Y[mask]
    groups = np.asarray(groups)[mask] if groups is not None else contiguous_blocks(len(X), n_blocks=n_splits)
    rows = []
    preds = np.full_like(Y, np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(group_kfold_indices(groups, n_splits=n_splits)):
        if model_name == "mean_response_baseline":
            pred = np.repeat(Y[train_idx].mean(axis=0, keepdims=True), len(test_idx), axis=0)
        else:
            model = _population_encoding_estimator(
                model_name,
                alphas=alphas,
                random_state=random_state,
            )
            model.fit(X[train_idx], Y[train_idx])
            pred = model.predict(X[test_idx])
        preds[test_idx] = pred
        population_metrics = regression_row(Y[test_idx].ravel(), pred.ravel(), prefix="population_")
        cell_rs = [safe_pearson(Y[test_idx, c], pred[:, c]) for c in range(Y.shape[1])]
        rows.append({
            "fold": fold,
            "model": model_name,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_cells": Y.shape[1],
            "mean_cell_pearson_r": float(np.nanmean(cell_rs)),
            "median_cell_pearson_r": float(np.nanmedian(cell_rs)),
            "noise_corrected_population_r2": noise_corrected_score(
                population_metrics.get("population_r2", np.nan),
                noise_ceiling,
            ),
            **population_metrics,
        })
    return pd.DataFrame(rows), preds


def reduced_rank_population_encoding(
    stimulus_features,
    neural_response,
    groups=None,
    response_rank: int = 10,
    alphas=(0.1, 1, 10, 100),
    n_splits=5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Low-rank stimulus-to-population encoding via PCA response bottleneck."""
    X = np.asarray(stimulus_features, dtype=float)
    Y = np.asarray(neural_response, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    groups = np.asarray(groups) if groups is not None else contiguous_blocks(len(X), n_blocks=n_splits)
    rows = []
    preds = np.full_like(Y, np.nan, dtype=float)
    for fold, (train_idx, test_idx) in enumerate(group_kfold_indices(groups, n_splits=n_splits)):
        rank = max(1, min(int(response_rank), Y.shape[1], len(train_idx) - 1))
        pca = PCA(n_components=rank)
        Y_train_low = pca.fit_transform(Y[train_idx])
        model = make_pipeline(StandardScaler(), MultiOutputRegressor(RidgeCV(alphas=list(alphas))))
        model.fit(X[train_idx], Y_train_low)
        pred_low = model.predict(X[test_idx])
        pred = pca.inverse_transform(pred_low)
        preds[test_idx] = pred
        rows.append({
            "fold": fold,
            "model": "reduced_rank_ridge",
            "response_rank": rank,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_cells": Y.shape[1],
            **regression_row(Y[test_idx].ravel(), pred.ravel(), prefix="population_"),
        })
    return pd.DataFrame(rows), preds
