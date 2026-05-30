import numpy as np
import pandas as pd

from v1_manifold.features import add_coarse_orientation_target, summarize_class_balance, safe_classification_target
from v1_manifold.evaluation import circular_shift_null_spearman, stimulus_neural_circular_shift_table
from v1_manifold.models import evaluate_decoders


def test_coarse_orientation_target_groups_rare_bins():
    df = pd.DataFrame({"dominant_orientation_bin": [3] * 10 + [4] * 6 + [0, 1, 7]})
    out = add_coarse_orientation_target(df, top_k=2)
    assert "dominant_orientation_coarse" in out.columns
    assert set(out["dominant_orientation_coarse"]) == {"bin_3", "bin_4", "other"}
    summary = summarize_class_balance(out["dominant_orientation_coarse"], "dominant_orientation_coarse")
    assert np.isclose(summary["fraction"].sum(), 1.0)


def test_safe_classification_target_rejects_singletons():
    assert not safe_classification_target([0, 0, 1], min_count_per_class=2)
    assert safe_classification_target([0, 0, 1, 1], min_count_per_class=2)


def test_circular_shift_null_returns_empirical_p_value():
    x = np.sin(np.linspace(0, 4 * np.pi, 120))
    y = np.roll(x, 3)
    rho, p, null = circular_shift_null_spearman(x, y, n_perm=25, min_shift=10, seed=0)
    assert -1 <= rho <= 1
    assert 0 < p <= 1
    assert len(null) == 25


def test_stimulus_neural_circular_shift_table_columns():
    frame_features = pd.DataFrame({
        "population_l2_norm": np.arange(80, dtype=float),
        "rms_contrast": np.arange(80, dtype=float)[::-1],
    })
    table = stimulus_neural_circular_shift_table(
        frame_features,
        neural_col="population_l2_norm",
        stimulus_cols=["rms_contrast"],
        n_perm=20,
        min_shift=5,
        seed=1,
    )
    assert {"spearman_r", "circular_shift_p", "null_mean", "null_std"}.issubset(table.columns)


def test_evaluate_decoders_includes_null_baselines():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 5))
    y = np.array(["bin_3", "bin_4", "other"] * 20)
    df, preds, fitted = evaluate_decoders(X, y, n_splits=3, include_null_baselines=True)
    assert "is_null_baseline" in df.columns
    assert "majority_class_baseline" in set(df["model"])
    assert "ridge_classifier_trained_on_shuffled_labels" in set(df["model"])
    assert set(preds).issubset(set(fitted))
