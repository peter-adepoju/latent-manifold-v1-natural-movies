from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def infer_layer_from_depth(depth: float, layer_depth_bins: dict) -> str:
    if pd.isna(depth):
        return "unknown"
    for layer, bounds in layer_depth_bins.items():
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo <= float(depth) < hi:
            return layer
    return "other"

def standardize_experiment_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = df.copy()
    colmap = {
        "ophys_experiment_id": "session_id",
        "experiment_id": "session_id",
        "id": "session_id",
        "targeted_structure": "targeted_structure",
        "cre_line": "cre_line",
        "imaging_depth": "imaging_depth",
        "specimen_id": "mouse_id",
        "donor_id": "mouse_id",
    }
    for old, new in colmap.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]
    bins = cfg.get("cohort", {}).get("layer_depth_bins", {})
    if "imaging_depth" in out.columns and bins:
        out["putative_layer"] = [infer_layer_from_depth(x, bins) for x in out["imaging_depth"]]
    else:
        out["putative_layer"] = "unknown"
    return out

def select_publication_cohort(experiments: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = standardize_experiment_table(experiments, cfg)
    cohort_cfg = cfg.get("cohort", {})
    if "targeted_structure" in out.columns:
        out = out[out["targeted_structure"].astype(str).eq(cohort_cfg.get("target_structure", "VISp"))].copy()
    preferred = cohort_cfg.get("preferred_cre_lines", [])
    if preferred and "cre_line" in out.columns:
        out["is_preferred_cre_line"] = out["cre_line"].isin(preferred)
        out = out[out["is_preferred_cre_line"]].copy()
    if "n_cells" in out.columns:
        out = out[out["n_cells"].fillna(0) >= cohort_cfg.get("min_cells", 0)].copy()
    group_cols = [c for c in ["cre_line", "putative_layer"] if c in out.columns]
    if group_cols:
        max_per_group = cohort_cfg.get("max_sessions_per_group", None)
        if max_per_group:
            out = out.sort_values([*group_cols, "session_id"]).groupby(group_cols, as_index=False, group_keys=False).head(int(max_per_group))
    return out.reset_index(drop=True)

def summarize_cohort(cohort: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ["cre_line", "putative_layer"] if c in cohort.columns]
    if not group_cols:
        return pd.DataFrame({"n_sessions": [len(cohort)]})
    return cohort.groupby(group_cols, dropna=False).agg(
        n_sessions=("session_id", "nunique"),
        n_mice=("mouse_id", "nunique") if "mouse_id" in cohort.columns else ("session_id", "nunique"),
        median_cells=("n_cells", "median") if "n_cells" in cohort.columns else ("session_id", "size"),
    ).reset_index()

def discover_existing_session_files(processed_dir: str | Path, interim_dir: str | Path) -> pd.DataFrame:
    processed_dir = Path(processed_dir)
    interim_dir = Path(interim_dir)
    rows = []
    for emb in processed_dir.glob("session_*_embeddings.npz"):
        parts = emb.name.split("_")
        if len(parts) >= 2:
            sid = parts[1]
            rows.append({"session_id": sid, "embedding_file": str(emb)})
    for h5 in interim_dir.glob("session_*_tensor.h5"):
        sid = h5.name.split("_")[1]
        rows.append({"session_id": sid, "tensor_file": str(h5)})
    if not rows:
        return pd.DataFrame(columns=["session_id", "embedding_file", "tensor_file"])
    return pd.DataFrame(rows).groupby("session_id", as_index=False).first()
