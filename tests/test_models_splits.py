import numpy as np

from v1_manifold.evaluation import grouped_stratified_splits, classification_metrics
from v1_manifold.schema import assert_group_disjoint
from v1_manifold.models import evaluate_decoders, evaluate_regressors


def test_grouped_splits_are_disjoint():
    X = np.random.randn(30, 4)
    y = np.array([0, 1, 2] * 10)
    groups = np.repeat(np.arange(10), 3)
    for train_idx, test_idx in grouped_stratified_splits(X, y, groups=groups, n_splits=5):
        assert_group_disjoint(groups[train_idx], groups[test_idx])


def test_classification_metrics_keys():
    m = classification_metrics([0, 1, 1], [0, 1, 0])
    assert "accuracy" in m
    assert "balanced_accuracy" in m
    assert "macro_f1" in m


def test_evaluate_decoders_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.array([0, 1, 2] * 20)
    df, preds, fitted = evaluate_decoders(X, y, n_splits=3)
    assert not df.empty
    assert "balanced_accuracy" in df.columns
    assert set(preds).issubset(set(fitted))


def test_evaluate_regressors_smoke():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = X[:, 0] * 0.5 + rng.normal(scale=0.1, size=50)
    groups = np.repeat(np.arange(10), 5)
    df, preds, fitted = evaluate_regressors(X, y, groups=groups, n_splits=5)
    assert not df.empty
    assert "r2" in df.columns
    assert set(preds).issubset(set(fitted))
