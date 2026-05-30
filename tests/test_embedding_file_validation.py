import numpy as np
import pandas as pd
import pytest

from v1_manifold.models import (
    as_2d_feature_matrix,
    valid_embedding_names,
    describe_embedding_file,
    evaluate_decoders,
)


class DummyNpz:
    def __init__(self, arrays):
        self._arrays = arrays
        self.files = list(arrays.keys())

    def __getitem__(self, key):
        return self._arrays[key]


def test_valid_embedding_names_skips_metadata_arrays():
    emb = DummyNpz({
        "frame": np.arange(10),
        "pca": np.zeros((10, 3)),
        "umap": np.zeros((10, 2)),
        "pca_explained_variance": np.ones(5),
        "participation_ratio": np.array(2.5),
        "wrong_rows": np.zeros((8, 3)),
    })

    names = valid_embedding_names(emb, n_samples=10)

    assert names == ["pca", "umap"]


def test_describe_embedding_file_marks_valid_frame_embeddings():
    emb = DummyNpz({
        "frame": np.arange(6),
        "isomap": np.zeros((6, 2)),
        "variance": np.ones(4),
    })

    summary = describe_embedding_file(emb, n_samples=6)
    isomap_row = summary.loc[summary["array_name"] == "isomap"].iloc[0]
    variance_row = summary.loc[summary["array_name"] == "variance"].iloc[0]

    assert bool(isomap_row["is_valid_frame_embedding"])
    assert not bool(variance_row["is_valid_frame_embedding"])


def test_as_2d_feature_matrix_reshapes_valid_single_feature_vector():
    X = np.arange(5)
    y = np.array([0, 1, 0, 1, 0])
    out = as_2d_feature_matrix(X, y)
    assert out.shape == (5, 1)


def test_as_2d_feature_matrix_rejects_metadata_vector():
    X = np.arange(3)
    y = np.array([0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="metadata array"):
        as_2d_feature_matrix(X, y)


def test_evaluate_decoders_accepts_single_feature_vector_when_aligned():
    X = np.linspace(0, 1, 12)
    y = np.array([0, 1] * 6)
    df, preds, fitted = evaluate_decoders(
        X,
        y,
        n_splits=3,
        random_state=0,
        include_null_baselines=True,
    )
    assert not df.empty
    assert set(df["n_features"]) == {1}
