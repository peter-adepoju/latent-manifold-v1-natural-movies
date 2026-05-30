from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, balanced_accuracy_score, accuracy_score, f1_score

def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def safe_spearman(y_true, y_pred) -> float:
    r, _ = spearmanr(y_true, y_pred)
    return float(r) if np.isfinite(r) else np.nan

def safe_pearson(y_true, y_pred) -> float:
    try:
        r, _ = pearsonr(np.asarray(y_true), np.asarray(y_pred))
    except Exception:
        r = np.nan
    return float(r) if np.isfinite(r) else np.nan

def regression_row(y_true, y_pred, prefix: str = "") -> dict:
    return {
        f"{prefix}r2": float(r2_score(y_true, y_pred)),
        f"{prefix}mae": float(mean_absolute_error(y_true, y_pred)),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}spearman_r": safe_spearman(y_true, y_pred),
        f"{prefix}pearson_r": safe_pearson(y_true, y_pred),
    }

def classification_row(y_true, y_pred, prefix: str = "") -> dict:
    return {
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def summarize_folds(rows: list[dict], group_cols: list[str] | None = None) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if group_cols is None:
        group_cols = [c for c in df.columns if c not in numeric_cols and c != "fold"]
    agg = df.groupby(group_cols, dropna=False)[numeric_cols].agg(["mean", "std", "median"]).reset_index()
    agg.columns = ["_".join([x for x in col if x]) for col in agg.columns.to_flat_index()]
    return agg
