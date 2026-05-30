from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.cross_decomposition import CCA
from .metrics import rmse


def zscore(X):
    X = np.asarray(X, dtype=float)
    return (X - np.nanmean(X, axis=0, keepdims=True)) / (np.nanstd(X, axis=0, keepdims=True) + 1e-12)

def procrustes_alignment_score(Z_source, Z_target):
    A = zscore(Z_source)
    B = zscore(Z_target)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    R, scale = orthogonal_procrustes(A, B)
    A_hat = A @ R
    return {"n_samples": n, "alignment_rmse": rmse(B, A_hat), "procrustes_scale": float(scale)}

def cca_alignment_score(Z_source, Z_target, n_components=3):
    A = zscore(Z_source)
    B = zscore(Z_target)
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    k = min(n_components, A.shape[1], B.shape[1], n - 1)
    cca = CCA(n_components=k, max_iter=1000)
    Xa, Yb = cca.fit_transform(A, B)
    cors = [np.corrcoef(Xa[:, i], Yb[:, i])[0, 1] for i in range(k)]
    return {"n_samples": n, "n_components": k, "mean_cca_r": float(np.nanmean(cors)), "median_cca_r": float(np.nanmedian(cors))}

def pairwise_session_alignment(embedding_by_session: dict, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    sessions = list(embedding_by_session)
    rows = []
    for i, a in enumerate(sessions):
        for b in sessions[i+1:]:
            row = {"session_a": a, "session_b": b}
            row.update(procrustes_alignment_score(embedding_by_session[a], embedding_by_session[b]))
            row.update(cca_alignment_score(embedding_by_session[a], embedding_by_session[b]))
            rows.append(row)
    out = pd.DataFrame(rows)
    if metadata is not None and not out.empty:
        meta = metadata.copy()
        meta["session_id"] = meta["session_id"].astype(str)
        out = out.merge(meta.add_suffix("_a"), left_on="session_a", right_on="session_id_a", how="left")
        out = out.merge(meta.add_suffix("_b"), left_on="session_b", right_on="session_id_b", how="left")
        if "putative_layer_a" in out.columns and "putative_layer_b" in out.columns:
            out["same_layer"] = out["putative_layer_a"] == out["putative_layer_b"]
        if "cre_line_a" in out.columns and "cre_line_b" in out.columns:
            out["same_cre_line"] = out["cre_line_a"] == out["cre_line_b"]
    return out
