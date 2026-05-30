from __future__ import annotations

import numpy as np
import pandas as pd


def cebra_variant_specs() -> list[dict[str, object]]:
    """Return the CEBRA variants needed for controlled manuscript claims."""
    return [
        {
            "variant": "cebra_time",
            "description": "Time/frame-conditioned CEBRA baseline.",
            "label_columns": ["movie_frame"],
            "claim_role": "temporal smoothness baseline",
        },
        {
            "variant": "cebra_stimulus_low_level",
            "description": "Stimulus-conditioned CEBRA using luminance/contrast/spatial features.",
            "label_columns": [
                "rms_contrast",
                "luminance_std",
                "spatial_frequency_centroid",
                "orientation_selectivity",
            ],
            "claim_role": "low-level visual-feature alignment",
        },
        {
            "variant": "cebra_stimulus_motion",
            "description": "Stimulus-conditioned CEBRA using temporal and motion-energy features.",
            "label_columns": [
                "temporal_absdiff_rms",
                "temporal_contrast",
                "motion_energy",
                "motion_proxy_magnitude",
            ],
            "claim_role": "natural-movie temporal-feature alignment",
        },
        {
            "variant": "cebra_hybrid_time_stimulus",
            "description": "Hybrid CEBRA using frame index plus real stimulus features.",
            "label_columns": [
                "movie_frame",
                "rms_contrast",
                "spatial_frequency_centroid",
                "orientation_selectivity",
                "motion_energy",
            ],
            "claim_role": "does stimulus supervision improve beyond time",
        },
    ]


def build_cebra_label_matrix(
    feature_table: pd.DataFrame,
    label_columns: list[str],
    zscore: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Build a finite label matrix for a CEBRA variant."""
    available = [c for c in label_columns if c in feature_table.columns]
    if not available:
        raise KeyError(f"None of the requested CEBRA label columns are available: {label_columns}")
    X = feature_table[available].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(X)
    if not finite.all():
        col_means = np.nanmean(np.where(finite, X, np.nan), axis=0)
        inds = np.where(~finite)
        X[inds] = np.take(col_means, inds[1])
    if zscore:
        X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)
    return X.astype(np.float32), available


def cebra_variant_plan(feature_table: pd.DataFrame) -> pd.DataFrame:
    """Describe which CEBRA variants are runnable for a feature table."""
    rows = []
    for spec in cebra_variant_specs():
        requested = list(spec["label_columns"])
        available = [c for c in requested if c in feature_table.columns]
        rows.append({
            "variant": spec["variant"],
            "description": spec["description"],
            "claim_role": spec["claim_role"],
            "requested_label_columns": ",".join(requested),
            "available_label_columns": ",".join(available),
            "n_requested": len(requested),
            "n_available": len(available),
            "is_runnable": len(available) > 0,
        })
    return pd.DataFrame(rows)
