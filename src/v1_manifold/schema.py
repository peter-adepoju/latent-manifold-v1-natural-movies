from __future__ import annotations

import numpy as np
import pandas as pd


def validate_dff_traces(dff: np.ndarray) -> dict[str, int]:
    dff = np.asarray(dff)
    if dff.ndim != 2:
        raise ValueError(f"ΔF/F traces must be 2D [cells, time], got shape {dff.shape}")
    n_cells, n_time = dff.shape
    if n_cells < 1 or n_time < 2:
        raise ValueError("ΔF/F matrix must contain at least one cell and two time points.")
    return {"n_cells": int(n_cells), "n_timepoints": int(n_time)}


def validate_stimulus_table(stimulus_table: pd.DataFrame) -> dict[str, int]:
    required = {"start", "end"}
    missing = required.difference(stimulus_table.columns)
    if missing:
        raise ValueError(f"Stimulus table is missing columns: {sorted(missing)}")
    if not (stimulus_table["end"] >= stimulus_table["start"]).all():
        raise ValueError("Stimulus table contains rows where end < start.")
    return {"n_presentations": int(len(stimulus_table))}


def validate_trial_tensor(tensor: np.ndarray) -> dict[str, int]:
    tensor = np.asarray(tensor)
    if tensor.ndim != 3:
        raise ValueError(f"Trial tensor must be 3D [repeats, cells, movie_frames], got {tensor.shape}")
    repeats, cells, frames = tensor.shape
    if repeats < 1 or cells < 1 or frames < 2:
        raise ValueError("Trial tensor has invalid dimensions.")
    if not np.isfinite(tensor).all():
        raise ValueError("Trial tensor contains non-finite values after preprocessing.")
    return {"n_repeats": int(repeats), "n_cells": int(cells), "n_movie_frames": int(frames)}


def assert_group_disjoint(train_groups, test_groups) -> None:
    overlap = set(train_groups).intersection(set(test_groups))
    if overlap:
        raise ValueError(f"Group leakage detected. Overlapping groups: {sorted(overlap)[:10]}")
