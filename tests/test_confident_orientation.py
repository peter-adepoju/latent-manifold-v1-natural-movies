import numpy as np
import pandas as pd

from v1_manifold.features import (
    add_coarse_orientation_target,
    add_confident_orientation_target,
    balanced_confident_orientation_subset,
    orientation_label_interpretation_table,
    summarize_class_balance,
)


def _feature_table():
    return pd.DataFrame({
        "movie_frame": np.arange(12),
        "dominant_orientation_bin": [3, 3, 3, 3, 4, 4, 4, 0, 1, 2, 5, 6],
        "orientation_selectivity": [0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.75, 0.1, 0.2, 0.3, 0.4, 0.45],
    })


def test_confident_orientation_target_marks_ambiguous_frames():
    df = add_coarse_orientation_target(_feature_table(), top_k=2)
    out = add_confident_orientation_target(df, confidence_quantile=0.5)
    assert "dominant_orientation_confident" in out.columns
    assert "orientation_confidence" in out.columns
    assert "ambiguous" in set(out["dominant_orientation_confident"])
    summary = summarize_class_balance(out["dominant_orientation_confident"], "dominant_orientation_confident")
    assert np.isclose(summary["fraction"].sum(), 1.0)


def test_balanced_confident_orientation_subset_balances_nonambiguous_classes():
    df = add_coarse_orientation_target(_feature_table(), top_k=2)
    out = add_confident_orientation_target(df, confidence_quantile=0.4)
    balanced = balanced_confident_orientation_subset(out, random_state=0)
    counts = balanced["dominant_orientation_confident"].value_counts()
    assert len(counts) >= 2
    assert counts.nunique() == 1
    assert "ambiguous" not in set(balanced["dominant_orientation_confident"])


def test_orientation_label_interpretation_table_has_recommendations():
    df = add_coarse_orientation_target(_feature_table(), top_k=2)
    out = add_confident_orientation_target(df, confidence_quantile=0.5)
    table = orientation_label_interpretation_table(out)
    assert "recommended_primary_decoding_target" in set(table["finding"])
    assert "fraction_ambiguous_orientation_frames" in set(table["finding"])
