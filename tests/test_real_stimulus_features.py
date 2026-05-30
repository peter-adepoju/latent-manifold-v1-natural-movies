import numpy as np
import pandas as pd
import pytest

from v1_manifold.features import (
    build_real_movie_feature_table,
    assert_real_stimulus_features,
    fallback_frame_labels,
    make_single_trial_design_matrix,
)


def test_real_movie_feature_table_has_nonfallback_visual_columns():
    rng = np.random.default_rng(0)
    movie = rng.normal(size=(12, 32, 40))
    features = build_real_movie_feature_table(
        movie,
        n_orientation_bins=8,
        n_spatial_frequency_bins=5,
        downsample_factor=2,
    )
    assert len(features) == 12
    assert "dominant_orientation_bin" in features.columns
    assert "spatial_frequency_centroid" in features.columns
    assert "rms_contrast" in features.columns
    assert not features["fallback_labels"].any()
    assert_real_stimulus_features(features)


def test_assert_real_stimulus_features_rejects_fallback_labels():
    fallback = fallback_frame_labels(10, n_orientation_bins=8, n_sf_bins=5)
    with pytest.raises(ValueError, match="fallback labels"):
        assert_real_stimulus_features(fallback)


def test_single_trial_design_matrix_shape_and_groups():
    tensor = np.ones((3, 4, 5), dtype=float)
    X, repeat, frame = make_single_trial_design_matrix(tensor)
    assert X.shape == (15, 4)
    assert repeat.tolist() == [0] * 5 + [1] * 5 + [2] * 5
    assert frame.tolist() == [0, 1, 2, 3, 4] * 3
