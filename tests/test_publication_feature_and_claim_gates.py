import numpy as np
import pandas as pd

from v1_manifold_publication.cebra_variants import build_cebra_label_matrix, cebra_variant_plan
from v1_manifold_publication.decoding import summarize_real_vs_null
from v1_manifold_publication.dynamics import evaluate_latent_dynamics, make_horizon_dataset
from v1_manifold_publication.multisession import build_session_run_plan, claim_gate_table
from v1_manifold_publication.stimulus_features import (
    extract_frame_feature_table,
    feature_hierarchy_columns,
)


def test_enhanced_stimulus_features_include_motion_and_hierarchy():
    rng = np.random.default_rng(0)
    frames = rng.normal(size=(4, 24, 32))
    features = extract_frame_feature_table(frames, resize_shape=(12, 16), include_gabor=False)
    assert "motion_energy" in features.columns
    assert "temporal_contrast" in features.columns
    assert "edge_density_p90" in features.columns
    hierarchy = feature_hierarchy_columns(features)
    assert "motion_energy" in hierarchy["level3_temporal_motion"]
    assert "rms_contrast" in hierarchy["level1_luminance_contrast"]


def test_cebra_variant_plan_detects_available_labels():
    features = pd.DataFrame({
        "movie_frame": [0, 1, 2],
        "rms_contrast": [0.1, 0.2, 0.3],
        "motion_energy": [1.0, 2.0, 3.0],
    })
    labels, cols = build_cebra_label_matrix(features, ["movie_frame", "rms_contrast"])
    assert labels.shape == (3, 2)
    assert cols == ["movie_frame", "rms_contrast"]
    plan = cebra_variant_plan(features)
    assert plan["is_runnable"].any()


def test_summarize_real_vs_null_reports_empirical_pvalue():
    real = pd.DataFrame({"r2": [0.5, 0.4]})
    nulls = pd.DataFrame({"null": ["shuffle", "shuffle", "shuffle"], "r2": [0.1, 0.2, 0.3]})
    summary = summarize_real_vs_null(real, nulls, metric="r2")
    assert summary.loc[0, "delta_observed_minus_null"] > 0
    assert 0 < summary.loc[0, "empirical_p_greater"] <= 1


def test_latent_dynamics_horizon_dataset_and_baselines():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(80, 3)).cumsum(axis=0)
    X, Y = make_horizon_dataset(Z, horizon=5, history=4)
    assert X.shape[0] == Y.shape[0]
    assert X.shape[1] == 12
    table = evaluate_latent_dynamics(Z, horizons=(1, 5), history=4)
    assert {"persistence_baseline", "linear_autoregressive"}.issubset(set(table["model"]))


def test_multisession_run_plan_and_claim_gates():
    cohort = pd.DataFrame({
        "session_id": ["1", "2"],
        "cre_line": ["Cux2", "Rorb"],
        "putative_layer": ["L2/3", "L4"],
    })
    existing = pd.DataFrame({
        "session_id": ["1"],
        "tensor_file": ["tensor.h5"],
        "embedding_file": ["emb.npz"],
    })
    plan = build_session_run_plan(cohort, existing)
    assert plan.loc[plan["session_id"] == "1", "ready_for_publication_stats"].iloc[0]
    assert plan.loc[plan["session_id"] == "2", "needs_preprocessing"].iloc[0]

    report = {
        "cohort": {"n_processed_sessions": 1, "layers": ["L2/3"]},
        "blockers": ["Deep model alignment table is empty", "Cross-session manifold alignment table is empty"],
    }
    gates = claim_gate_table(report)
    assert "not_ready" in set(gates["status"])
