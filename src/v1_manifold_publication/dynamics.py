from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import regression_row


def make_horizon_dataset(Z: np.ndarray, horizon: int, history: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Build autoregressive samples for h-step latent prediction."""
    Z = np.asarray(Z, dtype=float)
    horizon = int(horizon)
    history = int(history)
    if Z.ndim != 2:
        raise ValueError("Z must be [time, latent_dim].")
    if horizon < 1 or history < 1:
        raise ValueError("horizon and history must be positive.")
    n = Z.shape[0] - horizon - history + 1
    if n <= 0:
        raise ValueError("Not enough samples for the requested horizon/history.")
    X = np.stack([Z[i:i + history].reshape(-1) for i in range(n)])
    Y = np.stack([Z[i + history + horizon - 1] for i in range(n)])
    return X, Y


def persistence_prediction(Z: np.ndarray, horizon: int, history: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Predict the future latent state as the latest observed state."""
    X, Y = make_horizon_dataset(Z, horizon=horizon, history=history)
    latent_dim = np.asarray(Z).shape[1]
    pred = X[:, -latent_dim:]
    return Y, pred


def mean_state_prediction(Z: np.ndarray, horizon: int, history: int = 1) -> tuple[np.ndarray, np.ndarray]:
    X, Y = make_horizon_dataset(Z, horizon=horizon, history=history)
    pred = np.repeat(np.asarray(Z, dtype=float).mean(axis=0, keepdims=True), len(Y), axis=0)
    return Y, pred


def linear_autoregressive_prediction(
    Z: np.ndarray,
    horizon: int,
    history: int = 10,
    train_fraction: float = 0.8,
    alphas=(0.1, 1, 10, 100),
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a linear autoregressive baseline on an early movie segment."""
    X, Y = make_horizon_dataset(Z, horizon=horizon, history=history)
    split = max(1, min(len(Y) - 1, int(len(Y) * float(train_fraction))))
    model = make_pipeline(StandardScaler(), MultiOutputRegressor(RidgeCV(alphas=list(alphas))))
    model.fit(X[:split], Y[:split])
    pred = np.full_like(Y, np.nan, dtype=float)
    pred[split:] = model.predict(X[split:])
    return Y[split:], pred[split:]


def evaluate_latent_dynamics(
    Z: np.ndarray,
    horizons=(1, 5, 10, 20, 50),
    history: int = 10,
    train_fraction: float = 0.8,
) -> pd.DataFrame:
    """Evaluate multi-horizon latent dynamics against simple baselines."""
    rows = []
    for horizon in horizons:
        for model_name, fn in [
            ("persistence_baseline", lambda z: persistence_prediction(z, horizon, history=1)),
            ("mean_state_baseline", lambda z: mean_state_prediction(z, horizon, history=1)),
            (
                "linear_autoregressive",
                lambda z: linear_autoregressive_prediction(
                    z,
                    horizon,
                    history=history,
                    train_fraction=train_fraction,
                ),
            ),
        ]:
            try:
                y_true, pred = fn(Z)
                rows.append({
                    "horizon": int(horizon),
                    "model": model_name,
                    "history": int(1 if "baseline" in model_name else history),
                    "n_samples": int(len(y_true)),
                    "latent_dim": int(np.asarray(Z).shape[1]),
                    **regression_row(y_true.ravel(), pred.ravel()),
                })
            except Exception as exc:
                rows.append({
                    "horizon": int(horizon),
                    "model": model_name,
                    "error": repr(exc),
                })
    return pd.DataFrame(rows)
