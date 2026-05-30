import numpy as np

from v1_manifold.geometry import trajectory_speed, trajectory_curvature, trajectory_tangling, two_nn_intrinsic_dimension, rsa_spearman


def test_trajectory_metrics_have_expected_lengths():
    t = np.linspace(0, 2 * np.pi, 100)
    Z = np.c_[np.cos(t), np.sin(t), t / t.max()]
    assert trajectory_speed(Z).shape == (100,)
    assert trajectory_curvature(Z).shape == (100,)
    assert trajectory_tangling(Z).shape == (100,)


def test_two_nn_returns_positive_for_plane():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 2))
    dim = two_nn_intrinsic_dimension(X)
    assert np.isfinite(dim)
    assert dim > 0


def test_rsa_spearman_identical_is_high():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 5))
    rho = rsa_spearman(X, X)
    assert rho > 0.99
