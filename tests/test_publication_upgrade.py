import numpy as np
import pandas as pd

from v1_manifold_publication.splits import contiguous_blocks
from v1_manifold_publication.metrics import rmse, regression_row
from v1_manifold_publication.geometry_publication import geometry_timeseries, summarize_geometry_ts
from v1_manifold_publication.brain_model_alignment import linear_cka, rsa_score
from v1_manifold_publication.nulls import empirical_pvalue
from v1_manifold_publication.reliability import coerce_trial_tensor_layout, split_half_reliability
from v1_manifold_publication.cell_matching import (
    choose_matched_cell_count,
    subsample_cells,
    cell_count_matched_metric_table,
)


def test_contiguous_blocks():
    g = contiguous_blocks(10, 3)
    assert len(g) == 10
    assert len(np.unique(g)) == 3


def test_rmse_no_squared_kwarg_dependency():
    assert abs(rmse([0, 1], [0, 3]) - np.sqrt(2)) < 1e-8


def test_geometry_timeseries_shape():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(50, 3))
    geom = geometry_timeseries(Z)
    assert set(["speed", "curvature", "tangling"]).issubset(geom.columns)
    assert len(geom) == 50
    summary = summarize_geometry_ts(geom)
    assert len(summary) == 3


def test_alignment_scores_are_finite_for_identical_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 5))
    assert linear_cka(X, X) > 0.99
    r, p = rsa_score(X, X)
    assert r > 0.99


def test_empirical_pvalue_bounds():
    p = empirical_pvalue(1.0, [0.1, 0.2, 2.0], alternative="greater")
    assert 0 < p <= 1


def test_reliability_accepts_core_repeat_cell_frame_layout():
    rng = np.random.default_rng(2)
    repeat_frame_cell = rng.normal(size=(6, 20, 4))
    repeat_cell_frame = np.transpose(repeat_frame_cell, (0, 2, 1))

    coerced = coerce_trial_tensor_layout(repeat_cell_frame, layout="auto")
    assert coerced.shape == repeat_frame_cell.shape
    assert np.allclose(coerced, repeat_frame_cell)

    rel_a = split_half_reliability(repeat_frame_cell, n_splits=5, rng=0, layout="repeats_frames_cells")
    rel_b = split_half_reliability(repeat_cell_frame, n_splits=5, rng=0, layout="repeats_cells_frames")
    assert np.allclose(rel_a["mean_cell_r"], rel_b["mean_cell_r"], equal_nan=True)


def test_cell_count_matching_subsamples_feature_axis():
    X = np.arange(50).reshape(10, 5)
    X_sub, idx = subsample_cells(X, n_cells=3, rng=0, cell_axis=1)
    assert X_sub.shape == (10, 3)
    assert len(idx) == 3
    assert choose_matched_cell_count([5, 7, 9], configured=6, min_cells=3) == 5


def test_cell_count_matched_metric_table_runs():
    rng = np.random.default_rng(3)
    populations = {
        "a": rng.normal(size=(30, 6)),
        "b": rng.normal(size=(30, 8)),
    }

    def metric_fn(X):
        return {"mean_abs": float(np.mean(np.abs(X)))}

    table = cell_count_matched_metric_table(
        populations,
        metric_fn,
        matched_cell_count=5,
        n_subsamples=3,
        random_state=0,
        min_cells=3,
    )
    assert len(table) == 6
    assert set(table["matched_cell_count"]) == {5}
    assert "mean_abs" in table.columns
