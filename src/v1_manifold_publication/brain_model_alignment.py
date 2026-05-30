from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .splits import contiguous_blocks, group_kfold_indices
from .metrics import regression_row

EPS = 1e-12

def center_kernel(K):
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H

def linear_cka(X, Y):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    K = X @ X.T
    L = Y @ Y.T
    hsic = np.sum(center_kernel(K) * center_kernel(L))
    norm = np.sqrt(np.sum(center_kernel(K)**2) * np.sum(center_kernel(L)**2)) + EPS
    return float(hsic / norm)

def rdm(X, metric="correlation"):
    return squareform(pdist(np.asarray(X, dtype=float), metric=metric))

def rsa_score(X, Y, metric="correlation"):
    rx = rdm(X, metric=metric)
    ry = rdm(Y, metric=metric)
    iu = np.triu_indices_from(rx, k=1)
    r, p = spearmanr(rx[iu], ry[iu])
    return float(r), float(p)

def compare_representations(neural_reps: dict, model_reps: dict, metric="correlation") -> pd.DataFrame:
    rows = []
    for n_name, X in neural_reps.items():
        for m_name, Y in model_reps.items():
            n = min(len(X), len(Y))
            r, p = rsa_score(X[:n], Y[:n], metric=metric)
            cka = linear_cka(X[:n], Y[:n])
            rows.append({"neural_representation": n_name, "model_representation": m_name, "n_samples": n, "rsa_spearman": r, "rsa_p": p, "linear_cka": cka})
    return pd.DataFrame(rows)

def analytic_gabor_like_features(feature_table: pd.DataFrame) -> np.ndarray:
    cols = [c for c in feature_table.columns if c.startswith("orientation_energy_") or c in ["spatial_frequency_centroid", "rms_contrast", "temporal_absdiff_rms", "fourier_power_slope"]]
    if not cols:
        raise KeyError("No analytic visual feature columns found for gabor-like model.")
    return feature_table[cols].to_numpy(dtype=float)

def ridge_model_to_neural_encoding(model_features, neural_response, n_blocks=5, alphas=(0.1, 1, 10, 100)) -> pd.DataFrame:
    X = np.asarray(model_features, dtype=float)
    Y = np.asarray(neural_response, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    groups = contiguous_blocks(n, n_blocks=n_blocks)
    rows = []
    for fold, (tr, te) in enumerate(group_kfold_indices(groups, n_splits=n_blocks)):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=list(alphas)))
        fold_preds = []
        for cell in range(Y.shape[1]):
            model.fit(X[tr], Y[tr, cell])
            fold_preds.append(model.predict(X[te]))
        pred = np.vstack(fold_preds).T
        rows.append({"fold": fold, "n_train": len(tr), "n_test": len(te), "n_cells": Y.shape[1], **regression_row(Y[te].ravel(), pred.ravel())})
    return pd.DataFrame(rows)

def extract_torchvision_features_placeholder():
    raise ImportError(
        "Deep-model feature extraction is intentionally optional. Install torch/torchvision/timm and use scripts/publication_extract_dnn_features.py."
    )
