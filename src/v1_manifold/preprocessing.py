from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import h5py

from .schema import validate_dff_traces, validate_stimulus_table, validate_trial_tensor


def rolling_quantile_baseline(dff: np.ndarray, window: int = 101, quantile: float = 0.08) -> np.ndarray:
    """Subtract a rolling low-percentile baseline from each cell's ΔF/F trace."""
    validate_dff_traces(dff)
    if window % 2 == 0:
        window += 1
    corrected = np.empty_like(dff, dtype=np.float32)
    for i, trace in enumerate(np.asarray(dff, dtype=np.float32)):
        series = pd.Series(trace)
        baseline = series.rolling(window=window, center=True, min_periods=max(3, window // 10)).quantile(quantile)
        baseline = baseline.bfill().ffill().to_numpy(dtype=np.float32)
        corrected[i] = trace - baseline
    return corrected


def zscore_cells(dff: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score traces per cell after baseline correction."""
    validate_dff_traces(dff)
    mu = np.nanmean(dff, axis=1, keepdims=True)
    sigma = np.nanstd(dff, axis=1, keepdims=True)
    z = (dff - mu) / (sigma + eps)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def estimate_snr(trace: np.ndarray) -> float:
    """Estimate a simple calcium-trace SNR using robust signal and derivative noise scales."""
    trace = np.asarray(trace, dtype=float)
    finite = np.isfinite(trace)
    if finite.sum() < 5:
        return 0.0
    x = trace[finite]
    signal = np.percentile(x, 95) - np.percentile(x, 5)
    noise = np.median(np.abs(np.diff(x) - np.median(np.diff(x)))) / 0.6745
    return float(signal / (noise + 1e-8))


def quality_filter_cells(
    dff: np.ndarray,
    max_nan_fraction: float = 0.20,
    min_snr: float = 1.50,
    min_mean_dff: float = 0.10,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Apply explicit cell-level quality filters before manifold learning."""
    validate_dff_traces(dff)
    nan_fraction = np.mean(~np.isfinite(dff), axis=1)
    mean_dff = np.nanmean(dff, axis=1)
    snr = np.array([estimate_snr(row) for row in dff], dtype=float)
    keep = (nan_fraction <= max_nan_fraction) & (snr >= min_snr) & (mean_dff >= min_mean_dff)
    qc = pd.DataFrame({
        "cell_index": np.arange(dff.shape[0]),
        "nan_fraction": nan_fraction,
        "mean_dff": mean_dff,
        "snr": snr,
        "keep_cell": keep,
    })
    return keep, qc


def _infer_frame_column(stimulus_table: pd.DataFrame) -> str | None:
    for col in ["frame", "movie_frame", "stimulus_frame"]:
        if col in stimulus_table.columns:
            return col
    return None


def add_repeat_index(stimulus_table: pd.DataFrame, expected_movie_frames: int = 900) -> pd.DataFrame:
    """Attach repeat and movie_frame columns to an Allen natural movie stimulus table."""
    stim = stimulus_table.copy().reset_index(drop=True)
    frame_col = _infer_frame_column(stim)
    if frame_col is None:
        stim["movie_frame"] = np.arange(len(stim)) % expected_movie_frames
    else:
        stim["movie_frame"] = stim[frame_col].astype(int)
    if "repeat" not in stim.columns:
        # A repeat begins when the movie frame index returns to 0 or decreases.
        frames = stim["movie_frame"].to_numpy()
        starts = np.r_[True, np.diff(frames) < 0]
        stim["repeat"] = np.cumsum(starts) - 1
    return stim


def build_trial_tensor(
    dff: np.ndarray,
    stimulus_table: pd.DataFrame,
    expected_movie_frames: int = 900,
    max_repeats: int | None = 10,
    aggregation: str = "mean",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Construct a [repeat, cell, movie_frame] response tensor from Allen ΔF/F traces."""
    validate_dff_traces(dff)
    validate_stimulus_table(stimulus_table)
    stim = add_repeat_index(stimulus_table, expected_movie_frames=expected_movie_frames)
    if max_repeats is not None:
        stim = stim[stim["repeat"] < int(max_repeats)].copy()

    repeats = sorted(stim["repeat"].unique().tolist())
    n_repeats = len(repeats)
    n_cells = dff.shape[0]
    n_frames = int(expected_movie_frames)
    tensor = np.full((n_repeats, n_cells, n_frames), np.nan, dtype=np.float32)

    repeat_to_idx = {rep: i for i, rep in enumerate(repeats)}
    for _, row in stim.iterrows():
        rep_idx = repeat_to_idx[row["repeat"]]
        movie_frame = int(row["movie_frame"])
        if movie_frame < 0 or movie_frame >= n_frames:
            continue
        start, end = int(row["start"]), int(row["end"])
        start = max(0, min(start, dff.shape[1] - 1))
        end = max(start + 1, min(end, dff.shape[1]))
        window = dff[:, start:end]
        if aggregation == "mean":
            tensor[rep_idx, :, movie_frame] = np.nanmean(window, axis=1)
        elif aggregation == "max":
            tensor[rep_idx, :, movie_frame] = np.nanmax(window, axis=1)
        else:
            raise ValueError(f"Unsupported aggregation: {aggregation}")

    # Fill missing movie frames using per-cell frame means and then zeros as a last resort.
    frame_mean = np.nanmean(tensor, axis=0, keepdims=True)
    inds = np.where(~np.isfinite(tensor))
    tensor[inds] = np.take_along_axis(np.broadcast_to(frame_mean, tensor.shape), inds[2][None, None, :], axis=2).ravel()[: len(inds[0])] if False else tensor[inds]
    if not np.isfinite(tensor).all():
        cell_frame_mean = np.nanmean(tensor, axis=0)
        for r in range(tensor.shape[0]):
            mask = ~np.isfinite(tensor[r])
            tensor[r][mask] = cell_frame_mean[mask]
    tensor = np.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
    validate_trial_tensor(tensor)
    return tensor, stim


def repeat_average_population_matrix(trial_tensor: np.ndarray) -> np.ndarray:
    """Average repeats to get [movie_frame, cell] population matrix for manifold learning."""
    validate_trial_tensor(trial_tensor)
    R_cells_frames = np.nanmean(trial_tensor, axis=0)
    return R_cells_frames.T.astype(np.float32)


def save_trial_tensor_h5(
    path: str | Path,
    trial_tensor: np.ndarray,
    stimulus_table: pd.DataFrame,
    cell_metadata: pd.DataFrame | None = None,
    attrs: dict | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_trial_tensor(trial_tensor)
    with h5py.File(path, "w") as h5:
        h5.create_dataset("trial_tensor", data=trial_tensor, compression="gzip")
        h5.create_dataset("repeat_average", data=repeat_average_population_matrix(trial_tensor), compression="gzip")
        stim_group = h5.create_group("stimulus_table")
        for col in stimulus_table.columns:
            values = stimulus_table[col].to_numpy()
            try:
                stim_group.create_dataset(col, data=values)
            except TypeError:
                stim_group.create_dataset(col, data=values.astype(str).astype("S"))
        if cell_metadata is not None:
            cell_group = h5.create_group("cell_metadata")
            for col in cell_metadata.columns:
                values = cell_metadata[col].to_numpy()
                try:
                    cell_group.create_dataset(col, data=values)
                except TypeError:
                    cell_group.create_dataset(col, data=values.astype(str).astype("S"))
        for key, value in (attrs or {}).items():
            h5.attrs[key] = value
    return path


def load_trial_tensor_h5(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        tensor = h5["trial_tensor"][:]
        repeat_average = h5["repeat_average"][:]
    validate_trial_tensor(tensor)
    return tensor, repeat_average
