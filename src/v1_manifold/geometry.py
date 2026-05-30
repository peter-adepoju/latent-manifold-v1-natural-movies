from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def two_nn_intrinsic_dimension(X: np.ndarray) -> float:
    """Estimate intrinsic dimensionality with scikit-dimension TwoNN, with a fallback implementation."""
    X = np.asarray(X)
    try:
        import skdim
        est = skdim.id.TwoNN().fit(X)
        return float(est.dimension_)
    except Exception:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=3).fit(X)
        distances, _ = nn.kneighbors(X)
        r1 = distances[:, 1]
        r2 = distances[:, 2]
        mu = r2 / (r1 + 1e-12)
        mu = np.sort(mu[np.isfinite(mu) & (mu > 1)])
        if len(mu) < 5:
            return float("nan")
        y = -np.log(1 - (np.arange(1, len(mu) + 1) / (len(mu) + 1)))
        x = np.log(mu)
        slope = np.sum(x * y) / (np.sum(x * x) + 1e-12)
        return float(slope)


def fisher_separability_proxy(X: np.ndarray) -> float:
    """Compute a simple Fisher-separability proxy after centering and normalizing samples."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    cos = Xn @ Xn.T
    np.fill_diagonal(cos, -np.inf)
    nearest_cos = np.max(cos, axis=1)
    return float(np.mean(1.0 - nearest_cos))


def trajectory_speed(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    return np.linalg.norm(np.gradient(Z, axis=0), axis=1)


def trajectory_curvature(Z: np.ndarray) -> np.ndarray:
    """Estimate discrete curvature for an embedded trajectory in R^d."""
    Z = np.asarray(Z, dtype=float)
    v = np.gradient(Z, axis=0)
    a = np.gradient(v, axis=0)
    v_norm2 = np.sum(v * v, axis=1)
    a_norm2 = np.sum(a * a, axis=1)
    va = np.sum(v * a, axis=1)
    numerator = np.sqrt(np.maximum(v_norm2 * a_norm2 - va ** 2, 0.0))
    denominator = np.power(v_norm2 + 1e-12, 1.5)
    return numerator / denominator


def trajectory_tangling(Z: np.ndarray, eps: float = 1e-6, max_points: int | None = 900) -> np.ndarray:
    """Compute trajectory tangling Q(t) from a latent trajectory."""
    Z = np.asarray(Z, dtype=float)
    if max_points is not None and Z.shape[0] > max_points:
        idx = np.linspace(0, Z.shape[0] - 1, max_points).astype(int)
        Z_work = Z[idx]
    else:
        idx = np.arange(Z.shape[0])
        Z_work = Z
    dZ = np.gradient(Z_work, axis=0)
    q = np.zeros(Z_work.shape[0], dtype=float)
    for i in range(Z_work.shape[0]):
        num = np.sum((dZ[i] - dZ) ** 2, axis=1)
        den = np.sum((Z_work[i] - Z_work) ** 2, axis=1) + eps
        ratio = num / den
        ratio[i] = 0.0
        q[i] = np.max(ratio)
    if len(idx) == Z.shape[0]:
        return q
    full_q = np.interp(np.arange(Z.shape[0]), idx, q)
    return full_q


def persistent_homology(X: np.ndarray, maxdim: int = 2, subsample: int = 1500, seed: int = 42) -> dict:
    """Run Ripser persistent homology on a manageable subsample."""
    try:
        from ripser import ripser
    except ImportError as exc:
        raise ImportError("Install ripser to compute persistent homology.") from exc
    X = np.asarray(X, dtype=float)
    if X.shape[0] > subsample:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(X.shape[0], size=subsample, replace=False))
        X = X[idx]
    return ripser(X, maxdim=maxdim)


def rdm(X: np.ndarray, metric: str = "correlation") -> np.ndarray:
    return squareform(pdist(np.asarray(X), metric=metric))


def rsa_spearman(X: np.ndarray, Y: np.ndarray, metric: str = "correlation") -> float:
    rx = rdm(X, metric=metric)
    ry = rdm(Y, metric=metric)
    iu = np.triu_indices_from(rx, k=1)
    rho = spearmanr(rx[iu], ry[iu]).correlation
    return float(rho)


def summarize_geometry(name: str, Z: np.ndarray) -> dict[str, float | str]:
    speed = trajectory_speed(Z)
    curvature = trajectory_curvature(Z)
    tangling = trajectory_tangling(Z)
    return {
        "embedding": name,
        "mean_speed": float(np.mean(speed)),
        "median_speed": float(np.median(speed)),
        "mean_curvature": float(np.mean(curvature)),
        "median_curvature": float(np.median(curvature)),
        "mean_tangling": float(np.mean(tangling)),
        "median_tangling": float(np.median(tangling)),
        "twonn_id": two_nn_intrinsic_dimension(Z),
        "fisher_separability_proxy": fisher_separability_proxy(Z),
    }
