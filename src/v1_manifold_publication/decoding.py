from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from .splits import contiguous_blocks, group_kfold_indices, kfold_indices
from .metrics import regression_row, classification_row
from .nulls import shuffled_labels, circular_shift, block_permutation


def evaluate_regression_cv(X, y, groups=None, model_name="ridge_cv", alphas=(0.1,1,10), n_splits=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    groups = np.asarray(groups)[mask] if groups is not None else None
    if model_name == "ridge_cv":
        estimator = make_pipeline(StandardScaler(), RidgeCV(alphas=list(alphas)))
    elif model_name == "random_forest":
        estimator = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1, min_samples_leaf=3)
    elif model_name == "mean_baseline":
        estimator = DummyRegressor(strategy="mean")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    splitter = group_kfold_indices(groups, n_splits=n_splits) if groups is not None else kfold_indices(len(y), n_splits=n_splits, random_state=random_state)
    rows, preds = [], np.full(len(y), np.nan)
    for fold, (train_idx, test_idx) in enumerate(splitter):
        estimator.fit(X[train_idx], y[train_idx])
        pred = estimator.predict(X[test_idx])
        preds[test_idx] = pred
        rows.append({"fold": fold, "model": model_name, "n_train": len(train_idx), "n_test": len(test_idx), **regression_row(y[test_idx], pred)})
    return pd.DataFrame(rows), preds


def evaluate_classification_cv(X, y, groups=None, model_name="logistic_l2", n_splits=5, random_state=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    mask = pd.Series(y).notna().to_numpy() & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    groups = np.asarray(groups)[mask] if groups is not None else None
    if len(np.unique(y)) < 2:
        return pd.DataFrame(), np.array([])
    if model_name == "logistic_l2":
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced_subsample", n_jobs=-1)
    elif model_name == "majority_baseline":
        estimator = DummyClassifier(strategy="most_frequent")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
    splitter = group_kfold_indices(groups, n_splits=n_splits) if groups is not None else kfold_indices(len(y), n_splits=n_splits, random_state=random_state)
    rows, preds = [], np.empty(len(y), dtype=object)
    for fold, (train_idx, test_idx) in enumerate(splitter):
        estimator.fit(X[train_idx], y[train_idx])
        pred = estimator.predict(X[test_idx])
        preds[test_idx] = pred
        rows.append({"fold": fold, "model": model_name, "n_train": len(train_idx), "n_test": len(test_idx), **classification_row(y[test_idx], pred)})
    return pd.DataFrame(rows), preds


def evaluate_with_nulls(X, y, groups, task="regression", n_permutations=1000, random_state=42, model_name=None):
    rng = np.random.default_rng(random_state)
    if task == "regression":
        base_model = model_name or "ridge_cv"
        real, _ = evaluate_regression_cv(X, y, groups=groups, model_name=base_model, random_state=random_state)
        metric = "r2"
        evaluator = evaluate_regression_cv
    else:
        base_model = model_name or "logistic_l2"
        real, _ = evaluate_classification_cv(X, y, groups=groups, model_name=base_model, random_state=random_state)
        metric = "balanced_accuracy"
        evaluator = evaluate_classification_cv
    null_rows = []
    for p in range(int(n_permutations)):
        for null_name, y_null in [
            ("shuffle", shuffled_labels(y, rng)),
            ("circular_shift", circular_shift(y, rng=rng)[0]),
            ("block_permutation", block_permutation(y, rng=rng)),
        ]:
            df, _ = evaluator(X, y_null, groups=groups, model_name=base_model, random_state=random_state + p)
            if not df.empty:
                null_rows.append({"permutation": p, "null": null_name, metric: df[metric].mean()})
    return real, pd.DataFrame(null_rows)


def movie_block_groups(n_frames, n_blocks=5):
    return contiguous_blocks(n_frames, n_blocks=n_blocks)


def summarize_real_vs_null(real: pd.DataFrame, nulls: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Summarize observed performance against empirical null distributions."""
    if real.empty:
        return pd.DataFrame()
    observed = float(real[metric].mean())
    rows = []
    if nulls.empty or metric not in nulls.columns:
        return pd.DataFrame([{
            "metric": metric,
            "observed": observed,
            "null": "none",
            "null_mean": np.nan,
            "null_std": np.nan,
            "delta_observed_minus_null": np.nan,
            "empirical_p_greater": np.nan,
            "n_null": 0,
        }])
    for null_name, sub in nulls.groupby("null", dropna=False):
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        p = float((np.sum(values >= observed) + 1) / (len(values) + 1))
        rows.append({
            "metric": metric,
            "observed": observed,
            "null": null_name,
            "null_mean": float(np.mean(values)),
            "null_std": float(np.std(values)),
            "delta_observed_minus_null": float(observed - np.mean(values)),
            "empirical_p_greater": p,
            "n_null": int(len(values)),
        })
    return pd.DataFrame(rows)
