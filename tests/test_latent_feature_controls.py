import numpy as np
import pandas as pd

from v1_manifold.evaluation import (
    make_contiguous_frame_blocks,
    frame_time_features,
    evaluate_feature_regression,
    latent_gain_over_temporal_baseline,
)
from v1_manifold.visualization import plot_embedding_2d


def test_make_contiguous_frame_blocks_creates_ordered_groups():
    groups = make_contiguous_frame_blocks(10, n_blocks=3)
    assert groups.shape == (10,)
    assert set(groups) == {0, 1, 2}
    assert np.all(np.diff(groups) >= 0)


def test_frame_time_features_shape():
    frames = np.arange(12)
    X = frame_time_features(frames, n_harmonics=2)
    assert X.shape == (12, 5)


def test_evaluate_feature_regression_runs_with_block_groups():
    rng = np.random.default_rng(0)
    n = 40
    X = rng.normal(size=(n, 3))
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    features = pd.DataFrame({
        "movie_frame": np.arange(n),
        "rms_contrast": y,
    })
    groups = make_contiguous_frame_blocks(n, n_blocks=4)

    scores = evaluate_feature_regression(
        {"test_representation": X},
        features,
        ["rms_contrast"],
        groups=groups,
    )

    assert len(scores) == 1
    assert scores.loc[0, "representation"] == "test_representation"
    assert scores.loc[0, "cv_strategy"] == "contiguous_movie_block_groupkfold"


def test_latent_gain_over_temporal_baseline_computes_delta():
    table = pd.DataFrame({
        "representation": [
            "pca_3d",
            "frame_index_fourier_baseline",
            "population_l2_norm_baseline",
        ],
        "target": ["rms_contrast", "rms_contrast", "rms_contrast"],
        "mean_r2": [0.20, 0.10, 0.05],
        "mean_mae": [0.1, 0.2, 0.3],
        "mean_spearman_r": [0.5, 0.2, 0.1],
    })
    out = latent_gain_over_temporal_baseline(table)
    assert out.loc[0, "delta_r2_latent_minus_temporal"] == 0.10


def test_plot_embedding_accepts_custom_colorbar_label():
    Z = np.random.default_rng(0).normal(size=(20, 2))
    fig = plot_embedding_2d(Z, color=np.arange(20), colorbar_label="RMS contrast")
    assert fig is not None
