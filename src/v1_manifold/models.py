from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier, RidgeCV, ElasticNetCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

from .evaluation import (
    classification_metrics, regression_metrics, grouped_stratified_splits, grouped_regression_splits,
    majority_class_baseline, shuffled_label_baseline,
)


def as_2d_feature_matrix(X, y=None):
    """Validate or reshape model inputs into a 2D sample-by-feature matrix.

    This protects decoding/evaluation code from accidentally receiving metadata
    arrays from an embeddings ``.npz`` file, such as PCA variance spectra. A
    one-dimensional vector is accepted only when it has one value per sample, in
    which case I treat it as a single-feature representation.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        if y is not None and X.shape[0] == len(y):
            X = X.reshape(-1, 1)
        else:
            raise ValueError(
                f"Expected a 2D feature matrix, but got a 1D array with shape {X.shape}. "
                "This usually means a metadata array from an embeddings file was passed "
                "as if it were a frame-level embedding."
            )

    if X.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, but got shape {X.shape}.")

    if y is not None and X.shape[0] != len(y):
        raise ValueError(
            f"Feature/target mismatch: X has {X.shape[0]} samples, but y has {len(y)} labels."
        )

    return X


def valid_embedding_names(embeddings, n_samples: int, exclude=("frame",)) -> list[str]:
    """Return only frame-level 2D embeddings from a loaded ``npz`` object.

    The notebook that creates embeddings also stores metadata arrays such as
    explained variance curves and scalar dimensionality estimates. I only pass
    arrays with shape ``[n_movie_frames, n_latent_dimensions]`` to decoders.
    """
    names: list[str] = []
    for name in embeddings.files:
        if name in exclude:
            continue
        arr = np.asarray(embeddings[name])
        if arr.ndim == 2 and arr.shape[0] == n_samples:
            names.append(name)
    return names


def describe_embedding_file(embeddings, n_samples: int) -> pd.DataFrame:
    """Summarize arrays stored in an embeddings ``.npz`` file."""
    rows = []
    valid = set(valid_embedding_names(embeddings, n_samples=n_samples))
    for name in embeddings.files:
        arr = np.asarray(embeddings[name])
        rows.append({
            "array_name": name,
            "shape": str(tuple(arr.shape)),
            "ndim": arr.ndim,
            "is_valid_frame_embedding": name in valid,
        })
    return pd.DataFrame(rows)


def baseline_decoders(random_state: int = 42) -> dict[str, object]:
    """Define strong classical baselines before deep models."""
    return {
        "ridge_classifier": Pipeline([
            ("scale", StandardScaler()),
            ("clf", RidgeClassifier(class_weight="balanced")),
        ]),
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None, random_state=random_state)),
        ]),
        "linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LinearSVC(class_weight="balanced", random_state=random_state, dual="auto")),
        ]),
        "random_forest": RandomForestClassifier(n_estimators=80, class_weight="balanced", random_state=random_state, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
    }


def evaluate_decoders(
    X,
    y,
    groups=None,
    n_splits: int = 5,
    random_state: int = 42,
    include_null_baselines: bool = True,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    """Evaluate classical decoders with imbalance-aware null baselines."""
    y = np.asarray(y)
    X = as_2d_feature_matrix(X, y)
    if len(np.unique(y)) < 2:
        raise ValueError("Classification target must contain at least two classes.")

    rows = []
    predictions: dict[str, np.ndarray] = {}
    fitted: dict[str, object] = {}
    split_iter = list(grouped_stratified_splits(X, y, groups=groups, n_splits=n_splits, seed=random_state))

    if include_null_baselines:
        metrics, y_pred = majority_class_baseline(y)
        metrics.update({
            "model": "majority_class_baseline",
            "n_samples": len(y),
            "n_features": X.shape[1],
            "is_null_baseline": True,
        })
        rows.append(metrics)
        predictions["majority_class_baseline"] = y_pred
        fitted["majority_class_baseline"] = {"strategy": "predict_global_majority_class"}

        metrics, y_pred = shuffled_label_baseline(y, seed=random_state)
        metrics.update({
            "model": "shuffled_label_baseline",
            "n_samples": len(y),
            "n_features": X.shape[1],
            "is_null_baseline": True,
        })
        rows.append(metrics)
        predictions["shuffled_label_baseline"] = y_pred
        fitted["shuffled_label_baseline"] = {"strategy": "random_permutation_of_labels", "seed": random_state}

        shuffled_model = clone(baseline_decoders(random_state)["ridge_classifier"])
        y_perm = np.random.default_rng(random_state).permutation(y)
        y_pred = cross_val_predict(shuffled_model, X, y_perm, cv=split_iter)
        metrics = classification_metrics(y, y_pred)
        metrics.update({
            "model": "ridge_classifier_trained_on_shuffled_labels",
            "n_samples": len(y),
            "n_features": X.shape[1],
            "is_null_baseline": True,
        })
        rows.append(metrics)
        predictions["ridge_classifier_trained_on_shuffled_labels"] = y_pred
        shuffled_model.fit(X, y_perm)
        fitted["ridge_classifier_trained_on_shuffled_labels"] = shuffled_model

    for name, model in baseline_decoders(random_state).items():
        y_pred = cross_val_predict(model, X, y, cv=split_iter)
        metrics = classification_metrics(y, y_pred)
        metrics.update({
            "model": name,
            "n_samples": len(y),
            "n_features": X.shape[1],
            "is_null_baseline": False,
        })
        rows.append(metrics)
        predictions[name] = y_pred
        model.fit(X, y)
        fitted[name] = model

    return pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False), predictions, fitted


def baseline_regressors(random_state: int = 42) -> dict[str, object]:
    """Define classical regressors for continuous visual-feature decoding."""
    return {
        "ridge_cv": Pipeline([
            ("scale", StandardScaler()),
            ("reg", RidgeCV(alphas=np.logspace(-3, 3, 13))),
        ]),
        "elastic_net": Pipeline([
            ("scale", StandardScaler()),
            ("reg", ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], alphas=np.logspace(-3, 1, 8), cv=3, random_state=random_state, max_iter=5000)),
        ]),
        "random_forest_regressor": RandomForestRegressor(n_estimators=120, random_state=random_state, n_jobs=1),
        "gradient_boosting_regressor": GradientBoostingRegressor(random_state=random_state),
    }


def evaluate_regressors(X, y, groups=None, n_splits: int = 5, random_state: int = 42) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    """Evaluate continuous stimulus-feature decoders with leakage-aware splits."""
    y = np.asarray(y, dtype=float)
    X = as_2d_feature_matrix(X, y)
    rows = []
    predictions = {}
    fitted = {}
    split_iter = list(grouped_regression_splits(X, y, groups=groups, n_splits=n_splits, seed=random_state))
    for name, model in baseline_regressors(random_state).items():
        y_pred = cross_val_predict(model, X, y, cv=split_iter)
        metrics = regression_metrics(y, y_pred)
        metrics.update({"model": name, "n_samples": len(y), "n_features": X.shape[1]})
        rows.append(metrics)
        predictions[name] = y_pred
        model.fit(X, y)
        fitted[name] = model
    return pd.DataFrame(rows).sort_values("r2", ascending=False), predictions, fitted


def save_fitted_models(models: dict[str, object], out_dir: str | Path, prefix: str) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, model in models.items():
        path = out_dir / f"{prefix}_{name}.joblib"
        joblib.dump(model, path)
        paths.append(path)
    return paths


class TorchDRNN:
    """A compact recurrent model for next-step latent-state prediction."""

    def __init__(self, input_size: int, hidden_size: int = 64, lr: float = 1e-3, device: str = "cpu"):
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("PyTorch is required for TorchDRNN.") from exc
        self.torch = torch
        self.nn = nn
        self.device = torch.device(device)
        self.model = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True).to(self.device)
        self.head = nn.Linear(hidden_size, input_size).to(self.device)
        self.optim = torch.optim.Adam(list(self.model.parameters()) + list(self.head.parameters()), lr=lr)
        self.loss_fn = nn.MSELoss()

    def _make_sequences(self, X: np.ndarray, sequence_length: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        xs, ys = [], []
        for i in range(0, len(X) - sequence_length):
            xs.append(X[i:i + sequence_length])
            ys.append(X[i + sequence_length])
        return np.stack(xs), np.stack(ys)

    def fit(self, X: np.ndarray, sequence_length: int = 30, epochs: int = 20, batch_size: int = 32) -> pd.DataFrame:
        torch = self.torch
        xs, ys = self._make_sequences(X, sequence_length)
        dataset = torch.utils.data.TensorDataset(torch.tensor(xs), torch.tensor(ys))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        rows = []
        for epoch in range(epochs):
            losses = []
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                self.optim.zero_grad()
                out, _ = self.model(xb)
                pred = self.head(out[:, -1])
                loss = self.loss_fn(pred, yb)
                loss.backward()
                self.optim.step()
                losses.append(float(loss.detach().cpu()))
            rows.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses))})
        return pd.DataFrame(rows)

    def predict_next(self, X: np.ndarray, sequence_length: int = 30) -> np.ndarray:
        torch = self.torch
        xs, _ = self._make_sequences(X, sequence_length)
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(xs), 128):
                xb = torch.tensor(xs[i:i + 128]).to(self.device)
                out, _ = self.model(xb)
                pred = self.head(out[:, -1]).detach().cpu().numpy()
                preds.append(pred)
        return np.vstack(preds)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"gru": self.model.state_dict(), "head": self.head.state_dict()}, path)
        return path
