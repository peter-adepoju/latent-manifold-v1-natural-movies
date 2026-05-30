from __future__ import annotations

from collections.abc import Callable
import numpy as np
import pandas as pd


def choose_matched_cell_count(
    cell_counts,
    configured: int | None = None,
    min_cells: int = 20,
) -> int:
    """Choose a defensible cell count for matched population comparisons."""
    counts = np.asarray(cell_counts, dtype=float)
    counts = counts[np.isfinite(counts)]
    if len(counts) == 0:
        raise ValueError("No finite cell counts were provided.")

    available = int(np.nanmin(counts))
    if configured is None:
        matched = available
    else:
        matched = min(int(configured), available)

    if matched < int(min_cells):
        raise ValueError(
            f"Matched cell count {matched} is below the requested minimum {int(min_cells)}."
        )
    return int(matched)


def subsample_cells(
    X: np.ndarray,
    n_cells: int,
    rng=None,
    cell_axis: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Subsample cells from a [frames, cells] or [cells, frames] matrix."""
    rng = np.random.default_rng(rng)
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError("X must be a two-dimensional population matrix.")
    if cell_axis not in (0, 1):
        raise ValueError("cell_axis must be 0 or 1.")

    available = arr.shape[cell_axis]
    n_cells = int(n_cells)
    if n_cells <= 0:
        raise ValueError("n_cells must be positive.")
    if n_cells > available:
        raise ValueError(f"Requested {n_cells} cells, but only {available} are available.")

    idx = np.sort(rng.choice(available, size=n_cells, replace=False))
    if cell_axis == 0:
        return arr[idx, :], idx
    return arr[:, idx], idx


def cell_count_matched_metric_table(
    population_by_session: dict[str, np.ndarray],
    metric_fn: Callable[[np.ndarray], dict | float],
    matched_cell_count: int | None = None,
    n_subsamples: int = 100,
    random_state: int = 42,
    min_cells: int = 20,
) -> pd.DataFrame:
    """Evaluate a metric after repeated cell-count-matched subsampling.

    ``population_by_session`` values are expected to be [frames, cells] matrices.
    The metric function receives each subsampled [frames, matched_cells] matrix
    and may return either a scalar or a dictionary of scalar metrics.
    """
    if not population_by_session:
        return pd.DataFrame()

    counts = {str(k): np.asarray(v).shape[1] for k, v in population_by_session.items()}
    matched = choose_matched_cell_count(
        list(counts.values()),
        configured=matched_cell_count,
        min_cells=min_cells,
    )
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, object]] = []

    for session_id, X in population_by_session.items():
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Session {session_id} is not a [frames, cells] matrix.")
        for subsample in range(int(n_subsamples)):
            sub_rng = np.random.default_rng(int(rng.integers(0, 2**32 - 1)))
            X_sub, cell_idx = subsample_cells(X, matched, rng=sub_rng, cell_axis=1)
            metrics = metric_fn(X_sub)
            if isinstance(metrics, dict):
                metric_values = metrics
            else:
                metric_values = {"metric_value": float(metrics)}
            rows.append({
                "session_id": str(session_id),
                "subsample": int(subsample),
                "matched_cell_count": int(matched),
                "available_cell_count": int(counts[str(session_id)]),
                "cell_indices": " ".join(map(str, cell_idx.tolist())),
                **metric_values,
            })

    return pd.DataFrame(rows)
