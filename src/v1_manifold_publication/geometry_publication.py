from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from .nulls import circular_shift, empirical_pvalue

EPS = 1e-12

def speed(Z):
    Z = np.asarray(Z, dtype=float)
    d = np.diff(Z, axis=0, prepend=Z[[0]])
    return np.linalg.norm(d, axis=1)

def curvature(Z):
    Z = np.asarray(Z, dtype=float)
    v = np.gradient(Z, axis=0)
    a = np.gradient(v, axis=0)
    num = np.linalg.norm(a, axis=1)
    den = np.linalg.norm(v, axis=1) ** 2 + EPS
    return num / den

def tangling(Z, eps=1e-6):
    Z = np.asarray(Z, dtype=float)
    dZ = np.gradient(Z, axis=0)
    Dz = squareform(pdist(Z)) ** 2
    Dv = squareform(pdist(dZ)) ** 2
    return np.nanmax(Dv / (Dz + eps), axis=1)

def robust_summary(x, name):
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    return {
        "metric": name,
        "n_points": len(x),
        "n_finite": len(finite),
        "mean": float(np.mean(finite)) if len(finite) else np.nan,
        "median": float(np.median(finite)) if len(finite) else np.nan,
        "std": float(np.std(finite)) if len(finite) else np.nan,
        "p90": float(np.percentile(finite, 90)) if len(finite) else np.nan,
        "p95": float(np.percentile(finite, 95)) if len(finite) else np.nan,
        "p99": float(np.percentile(finite, 99)) if len(finite) else np.nan,
        "max": float(np.max(finite)) if len(finite) else np.nan,
    }

def geometry_timeseries(Z):
    s = speed(Z)
    c = curvature(Z)
    t = tangling(Z)
    return pd.DataFrame({
        "movie_frame": np.arange(len(s)),
        "speed": s,
        "curvature": c,
        "tangling": t,
        "log1p_speed": np.log1p(np.clip(s, 0, None)),
        "log1p_curvature": np.log1p(np.clip(c, 0, None)),
        "log1p_tangling": np.log1p(np.clip(t, 0, None)),
    })

def summarize_geometry_ts(geom_ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in ["speed", "curvature", "tangling"]:
        if col in geom_ts.columns:
            rows.append(robust_summary(geom_ts[col], col))
    return pd.DataFrame(rows)

def geometry_stimulus_association(geom_ts: pd.DataFrame, features: pd.DataFrame, stimulus_cols: list[str]) -> pd.DataFrame:
    df = geom_ts.merge(features[["movie_frame"] + stimulus_cols], on="movie_frame", how="inner")
    rows = []
    for g in ["log1p_speed", "log1p_curvature", "log1p_tangling"]:
        for s in stimulus_cols:
            r, p = spearmanr(df[g], df[s], nan_policy="omit")
            rows.append({"geometry_metric": g, "stimulus_feature": s, "spearman_r": float(r), "p_uncorrected": float(p), "n": int(df[[g, s]].dropna().shape[0])})
    return pd.DataFrame(rows)

def peak_feature_enrichment(geom_ts: pd.DataFrame, features: pd.DataFrame, stimulus_cols: list[str], quantile=0.90, n_permutations=1000, rng=42, min_shift=60):
    rng = np.random.default_rng(rng)
    merged = geom_ts.merge(features[["movie_frame"] + stimulus_cols], on="movie_frame", how="inner")
    rows = []
    for metric in ["speed", "curvature", "tangling"]:
        x = merged[metric].to_numpy(dtype=float)
        threshold = np.nanquantile(x, quantile)
        peak_mask = x >= threshold
        for feat in stimulus_cols:
            y = merged[feat].to_numpy(dtype=float)
            obs = float(np.nanmean(y[peak_mask]) - np.nanmean(y[~peak_mask]))
            nulls = []
            for _ in range(n_permutations):
                y_null, shift = circular_shift(y, min_shift=min_shift, rng=rng)
                nulls.append(float(np.nanmean(y_null[peak_mask]) - np.nanmean(y_null[~peak_mask])))
            rows.append({
                "geometry_metric": metric,
                "stimulus_feature": feat,
                "peak_quantile": quantile,
                "n_peak_frames": int(peak_mask.sum()),
                "n_nonpeak_frames": int((~peak_mask).sum()),
                "observed_delta_peak_minus_nonpeak": obs,
                "null_delta_mean": float(np.mean(nulls)),
                "null_delta_std": float(np.std(nulls)),
                "empirical_p_two_sided": empirical_pvalue(obs, nulls, alternative="two-sided"),
            })
    return pd.DataFrame(rows)
