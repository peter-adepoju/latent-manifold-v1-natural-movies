from __future__ import annotations

import pandas as pd


def build_session_run_plan(
    cohort: pd.DataFrame,
    existing_session_index: pd.DataFrame,
) -> pd.DataFrame:
    """Mark which cohort sessions still need preprocessing/manifold/evaluation."""
    if cohort.empty:
        return pd.DataFrame()
    out = cohort.copy()
    out["session_id"] = out["session_id"].astype(str)
    index = existing_session_index.copy()
    if index.empty:
        index = pd.DataFrame({"session_id": []})
    index["session_id"] = index["session_id"].astype(str)
    keep_cols = [c for c in ["session_id", "embedding_file", "tensor_file"] if c in index.columns]
    out = out.merge(index[keep_cols], on="session_id", how="left")
    if "tensor_file" not in out.columns:
        out["tensor_file"] = pd.NA
    if "embedding_file" not in out.columns:
        out["embedding_file"] = pd.NA
    out["has_tensor"] = out["tensor_file"].notna()
    out["has_embedding"] = out["embedding_file"].notna()
    out["needs_preprocessing"] = ~out["has_tensor"]
    out["needs_manifold"] = out["has_tensor"] & ~out["has_embedding"]
    out["ready_for_publication_stats"] = out["has_tensor"] & out["has_embedding"]
    return out


def claim_gate_table(readiness_report: dict) -> pd.DataFrame:
    """Translate readiness state into manuscript-claim gates."""
    cohort = readiness_report.get("cohort", {})
    processed = int(cohort.get("n_processed_sessions", 0))
    layers = len(cohort.get("layers", []))
    blockers = readiness_report.get("blockers", [])
    return pd.DataFrame([
        {
            "claim": "single-session V1 natural-movie decoding",
            "minimum_evidence": "one processed session with real features and held-out repeat decoding",
            "status": "supported" if processed >= 1 else "not_ready",
        },
        {
            "claim": "multi-session replication",
            "minimum_evidence": "at least 12 processed sessions across mice",
            "status": "supported" if processed >= 12 else "not_ready",
        },
        {
            "claim": "layer-specific manifold geometry",
            "minimum_evidence": "at least 3 layers with cell-count-matched session-level statistics",
            "status": "supported" if processed >= 12 and layers >= 3 else "not_ready",
        },
        {
            "claim": "deep-model brain alignment",
            "minimum_evidence": "nonempty DNN/ViT alignment table with random and low-level controls",
            "status": "not_ready" if any("Deep model alignment" in b for b in blockers) else "supported",
        },
        {
            "claim": "cross-session shared manifold",
            "minimum_evidence": "nonempty cross-session alignment table",
            "status": "not_ready" if any("Cross-session" in b for b in blockers) else "supported",
        },
    ])
