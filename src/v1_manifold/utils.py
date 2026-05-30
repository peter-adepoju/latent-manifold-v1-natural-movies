from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def save_table(df: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index)
    else:
        raise ValueError(f"Unsupported table format: {suffix}")
    return path


def finite_array(name: str, array: np.ndarray) -> None:
    if not np.isfinite(array).all():
        bad = np.size(array) - np.isfinite(array).sum()
        raise ValueError(f"{name} contains {bad} non-finite values")


def downsample_rows(X: np.ndarray, max_rows: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X)
    if X.shape[0] <= max_rows:
        idx = np.arange(X.shape[0])
        return X, idx
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(X.shape[0], size=max_rows, replace=False))
    return X[idx], idx


def require_file(path: str | Path, message: str | None = None) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(message or f"Required file was not found: {path}")
    return path
