from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import safe_pearson


def coerce_trial_tensor_layout(
    trial_tensor: np.ndarray,
    layout: str = "auto",
    expected_frames: int | None = None,
) -> np.ndarray:
    """Return a trial tensor as [repeats, frames, cells].

    The original project stores tensors as [repeat, cell, frame], while the
    publication-upgrade reliability code works on [repeat, frame, cell].
    Accepting both layouts prevents a silent swap of "mean cell reliability"
    into "mean frame reliability" when notebook 11 reads existing tensors.
    """
    X = np.asarray(trial_tensor, dtype=float)
    if X.ndim != 3:
        raise ValueError("trial_tensor must be three-dimensional.")

    normalized = layout.lower().replace("-", "_")
    if normalized in {"repeats_frames_cells", "repeat_frame_cell", "rfc"}:
        return X
    if normalized in {"repeats_cells_frames", "repeat_cell_frame", "rcf"}:
        return np.transpose(X, (0, 2, 1))
    if normalized != "auto":
        raise ValueError(
            "layout must be 'auto', 'repeats_frames_cells', or 'repeats_cells_frames'."
        )

    if expected_frames is not None:
        expected_frames = int(expected_frames)
        if X.shape[1] == expected_frames and X.shape[2] != expected_frames:
            return X
        if X.shape[2] == expected_frames and X.shape[1] != expected_frames:
            return np.transpose(X, (0, 2, 1))

    # Natural-movie analyses usually have many more frames than cells. If the
    # final axis is larger, this is very likely the core project's [R, C, F].
    if X.shape[2] > X.shape[1]:
        return np.transpose(X, (0, 2, 1))
    return X


def split_half_reliability(
    trial_tensor: np.ndarray,
    n_splits: int = 200,
    rng=None,
    layout: str = "auto",
    expected_frames: int | None = None,
) -> pd.DataFrame:
    """Estimate repeat reliability for a tensor coerced to [repeats, frames, cells]."""
    rng = np.random.default_rng(rng)
    X = coerce_trial_tensor_layout(
        trial_tensor,
        layout=layout,
        expected_frames=expected_frames,
    )
    repeats = X.shape[0]
    rows = []
    if repeats < 2:
        return pd.DataFrame(columns=["split", "population_r", "mean_cell_r"])
    idx = np.arange(repeats)
    for s in range(n_splits):
        perm = rng.permutation(idx)
        half = repeats // 2
        a = np.nanmean(X[perm[:half]], axis=0)
        b = np.nanmean(X[perm[half:]], axis=0)
        pop_r = safe_pearson(a.ravel(), b.ravel())
        cell_rs = [safe_pearson(a[:, c], b[:, c]) for c in range(X.shape[2])]
        rows.append({"split": s, "population_r": pop_r, "mean_cell_r": float(np.nanmean(cell_rs)), "median_cell_r": float(np.nanmedian(cell_rs))})
    return pd.DataFrame(rows)

def spearman_brown(r):
    r = np.asarray(r, dtype=float)
    return (2 * r) / (1 + r + 1e-12)

def noise_ceiling_from_repeats(
    trial_tensor: np.ndarray,
    layout: str = "auto",
    expected_frames: int | None = None,
) -> dict:
    rel = split_half_reliability(
        trial_tensor,
        n_splits=100,
        rng=42,
        layout=layout,
        expected_frames=expected_frames,
    )
    if rel.empty:
        return {"population_noise_ceiling": np.nan, "cell_noise_ceiling": np.nan}
    return {
        "population_noise_ceiling": float(np.nanmean(spearman_brown(rel["population_r"]))),
        "cell_noise_ceiling": float(np.nanmean(spearman_brown(rel["mean_cell_r"]))),
    }
