from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import math
import os
import re
import warnings
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def extract_session_id(path_or_name) -> str | None:
    """Extract the first numeric session-like token from a path or filename."""
    name = Path(path_or_name).name
    # Prefer 9-digit Allen experiment/session ids in filenames.
    m = re.search(r"(?:session_)?(\d{8,12})", name)
    if m:
        return m.group(1)
    parts = re.split(r"[_\-.]", name)
    for part in parts:
        if part.isdigit():
            return part
    return None


def canonical_session_id(x) -> str | None:
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        s = str(x).strip()
        if s.endswith(".0") and s[:-2].isdigit():
            return s[:-2]
        return s


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV robustly. Empty CSVs return an empty DataFrame."""
    path = Path(path)
    try:
        return pd.read_csv(path, **kwargs)
    except EmptyDataError:
        return pd.DataFrame()
    except FileNotFoundError:
        return pd.DataFrame()


def ensure_table(path: Path, columns: list[str]) -> Path:
    """Ensure an empty but parseable CSV exists."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not path.exists()) or path.stat().st_size == 0:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
    return path


def choose_best_feature_table(feature_candidates, require_movie_frame: bool = False) -> Path | None:
    """Choose the strongest available feature table for a session.

    Priority:
    1. ignore backup files;
    2. readable CSV;
    3. has movie_frame column;
    4. more columns;
    5. most recent modification time.
    """
    candidates = [Path(f) for f in feature_candidates if "backup" not in Path(f).name.lower()]
    if not candidates:
        return None

    scored = []
    for f in candidates:
        try:
            head = pd.read_csv(f, nrows=5)
            has_movie_frame = "movie_frame" in head.columns
            if require_movie_frame and not has_movie_frame:
                continue
            scored.append({
                "path": f,
                "n_columns": len(head.columns),
                "has_movie_frame": has_movie_frame,
                "modified_time": f.stat().st_mtime,
                "is_root_level": f.parent.name == "processed",
            })
        except Exception as exc:
            warnings.warn(f"Could not inspect feature table {f}: {exc}")
    if not scored:
        return None
    scored_df = pd.DataFrame(scored)
    scored_df = scored_df.sort_values(
        ["has_movie_frame", "n_columns", "is_root_level", "modified_time"],
        ascending=[False, False, True, False],
    )
    return Path(scored_df.iloc[0]["path"])


def candidate_feature_files(paths, session_id: str) -> list[Path]:
    session_id = str(session_id)
    candidates = []
    for base in [getattr(paths, "versioned_processed_dir", None), getattr(paths, "processed_dir", None)]:
        if base is None:
            continue
        base = Path(base)
        candidates += sorted(base.glob(f"session_{session_id}*publication_enhanced_frame_features.csv"))
        candidates += sorted(base.glob(f"session_{session_id}*real_frame_features.csv"))
        candidates += sorted(base.glob(f"session_{session_id}*frame_features.csv"))
    # de-duplicate while preserving order
    out = []
    seen = set()
    for f in candidates:
        key = str(f.resolve())
        if key not in seen:
            out.append(f)
            seen.add(key)
    return out


def find_tensor_file(paths, session_id: str) -> Path | None:
    session_id = str(session_id)
    candidates = sorted(Path(paths.interim_dir).glob(f"session_{session_id}*_tensor.h5"))
    return candidates[0] if candidates else None


def find_embedding_file(paths, session_id: str) -> Path | None:
    session_id = str(session_id)
    candidates = sorted(Path(paths.processed_dir).glob(f"session_{session_id}_embeddings.npz"))
    return candidates[0] if candidates else None


def discover_session_ids(paths, metadata_path: Path | None = None) -> list[str]:
    ids = set()
    if metadata_path and Path(metadata_path).exists():
        md = safe_read_csv(metadata_path)
        for col in ["session_id", "id", "ophys_experiment_id", "experiment_id", "ophys_session_id"]:
            if col in md.columns:
                ids |= {canonical_session_id(v) for v in md[col].dropna().tolist()}
                break
    for pattern_base in [Path(paths.processed_dir), Path(paths.interim_dir)]:
        for p in pattern_base.rglob("session_*"):
            sid = extract_session_id(p)
            if sid:
                ids.add(sid)
    return sorted(x for x in ids if x and x != "nan")


def build_multisession_asset_index(paths, metadata_path: Path | None = None) -> pd.DataFrame:
    """Return one row per discovered session with feature/tensor/embedding readiness."""
    session_ids = discover_session_ids(paths, metadata_path)
    rows = []
    for sid in session_ids:
        feature_path = choose_best_feature_table(candidate_feature_files(paths, sid))
        tensor_path = find_tensor_file(paths, sid)
        embedding_path = find_embedding_file(paths, sid)

        feature_shape = None
        if feature_path is not None:
            try:
                tmp = pd.read_csv(feature_path, nrows=5)
                # row count is optional/expensive but useful for small feature tables
                n_rows = sum(1 for _ in open(feature_path, "r", encoding="utf-8", errors="ignore")) - 1
                feature_shape = f"({n_rows}, {len(tmp.columns)})"
            except Exception:
                feature_shape = None

        embedding_names = []
        embedding_shapes = []
        if embedding_path is not None:
            try:
                with np.load(embedding_path, allow_pickle=False) as emb:
                    for name in emb.files:
                        arr = emb[name]
                        if arr.ndim == 2:
                            embedding_names.append(name)
                            embedding_shapes.append(f"{name}:{arr.shape}")
            except Exception as exc:
                embedding_names.append(f"ERROR:{exc}")

        rows.append({
            "session_id": sid,
            "feature_file": str(feature_path) if feature_path else "",
            "feature_shape": feature_shape,
            "tensor_file": str(tensor_path) if tensor_path else "",
            "embedding_file": str(embedding_path) if embedding_path else "",
            "embedding_names": ",".join(embedding_names),
            "embedding_shapes": ";".join(embedding_shapes),
            "has_features": feature_path is not None,
            "has_tensor": tensor_path is not None,
            "has_embeddings": embedding_path is not None,
            "ready_for_latent_decoding": feature_path is not None and embedding_path is not None,
            "ready_for_population_encoding": feature_path is not None and tensor_path is not None,
            "ready_for_full_neural_analysis": feature_path is not None and tensor_path is not None and embedding_path is not None,
        })
    return pd.DataFrame(rows)


def ready_sessions(asset_index: pd.DataFrame, column: str) -> list[str]:
    if asset_index.empty or column not in asset_index.columns:
        return []
    return asset_index.loc[asset_index[column].astype(bool), "session_id"].astype(str).tolist()


def movie_feature_targets(features: pd.DataFrame, requested: list[str]) -> list[str]:
    return [c for c in requested if c in features.columns and pd.api.types.is_numeric_dtype(features[c])]


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranked = np.empty_like(pvals)
    n = np.sum(np.isfinite(pvals))
    if n == 0:
        return np.full_like(pvals, np.nan)
    valid = np.isfinite(pvals)
    vals = pvals[valid]
    order = np.argsort(vals)
    adj = np.empty_like(vals)
    prev = 1.0
    for i in range(len(vals)-1, -1, -1):
        rank = i + 1
        val = vals[order[i]] * len(vals) / rank
        prev = min(prev, val)
        adj[order[i]] = prev
    out = np.full_like(pvals, np.nan)
    out[valid] = np.minimum(adj, 1.0)
    return out


def empirical_null_summary(df: pd.DataFrame, metric: str = "spearman_r") -> pd.DataFrame:
    """Summarize observed-vs-null rows using standardized metadata keys."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "row_type" not in df.columns:
        df["row_type"] = np.nan
    df["row_type"] = df["row_type"].fillna("null")
    if "permutation" in df.columns:
        df.loc[df["permutation"].notna(), "row_type"] = "null"

    for col in ["session_id", "representation", "target"]:
        if col not in df.columns:
            df[col] = np.nan

    df["_sid"] = df["session_id"].map(canonical_session_id)
    df["_rep"] = df["representation"].astype(str).str.strip()
    df["_target"] = df["target"].astype(str).str.strip()

    observed = df[df["row_type"] == "observed"].copy()
    nulls = df[df["row_type"] == "null"].copy()
    if observed.empty:
        return pd.DataFrame()

    obs_summary = observed.groupby(["_sid", "_rep", "_target"], as_index=False).agg(
        session_id=("session_id", "first"),
        representation=("representation", "first"),
        target=("target", "first"),
        observed_mean=(metric, "mean"),
        observed_median=(metric, "median"),
        observed_mean_r2=("r2", "mean") if "r2" in observed.columns else (metric, "mean"),
        n_observed_folds=(metric, "size"),
    )

    rows = []
    for _, row in obs_summary.iterrows():
        mask = (nulls["_sid"] == row["_sid"]) & (nulls["_rep"] == row["_rep"]) & (nulls["_target"] == row["_target"])
        null_values = nulls.loc[mask, metric].dropna().to_numpy()
        obs = float(row["observed_mean"]) if pd.notna(row["observed_mean"]) else np.nan
        if len(null_values) and np.isfinite(obs):
            p = (1 + np.sum(null_values >= obs)) / (len(null_values) + 1)
            z = (obs - np.mean(null_values)) / (np.std(null_values) + 1e-12)
            null_mean = float(np.mean(null_values))
            null_std = float(np.std(null_values))
            null_median = float(np.median(null_values))
        else:
            p = z = null_mean = null_std = null_median = np.nan
        rows.append({
            "session_id": row["session_id"],
            "representation": row["representation"],
            "target": row["target"],
            "metric": metric,
            "observed_mean": obs,
            "observed_median": row["observed_median"],
            "observed_mean_r2": row.get("observed_mean_r2", np.nan),
            "n_observed_folds": int(row["n_observed_folds"]),
            "n_null_rows": int(len(null_values)),
            "null_mean": null_mean,
            "null_median": null_median,
            "null_std": null_std,
            "empirical_p_one_sided": p,
            "null_z_score": z,
        })
    out = pd.DataFrame(rows)
    if not out.empty and "empirical_p_one_sided" in out:
        out["empirical_q_bh"] = bh_fdr(out["empirical_p_one_sided"].to_numpy())
    return out


def safe_table_index(folder: Path) -> pd.DataFrame:
    rows = []
    for table in sorted(Path(folder).glob("*.csv")):
        try:
            df = pd.read_csv(table)
            rows.append({"table": table.name, "rows": len(df), "columns": len(df.columns), "empty": df.empty, "error": ""})
        except EmptyDataError:
            rows.append({"table": table.name, "rows": 0, "columns": 0, "empty": True, "error": "EmptyDataError"})
        except Exception as exc:
            rows.append({"table": table.name, "rows": np.nan, "columns": np.nan, "empty": np.nan, "error": repr(exc)})
    return pd.DataFrame(rows)


def claim_gate_from_assets(asset_index: pd.DataFrame, table_folder: Path) -> pd.DataFrame:
    """Conservative claim gate for publication framing."""
    n_full = int(asset_index["ready_for_full_neural_analysis"].sum()) if not asset_index.empty and "ready_for_full_neural_analysis" in asset_index else 0
    n_embedding = int(asset_index["has_embeddings"].sum()) if not asset_index.empty and "has_embeddings" in asset_index else 0

    def exists_nonempty(name):
        p = Path(table_folder) / name
        if not p.exists() or p.stat().st_size == 0:
            return False
        try:
            return not pd.read_csv(p).empty
        except Exception:
            return False

    rows = [
        {
            "claim": "single-session V1 natural-movie decoding",
            "minimum_evidence": "at least one processed session with real features, neural tensors, embeddings, block-CV decoding, and null controls",
            "status": "supported" if n_full >= 1 and exists_nonempty("13_latent_feature_decoding_blockcv_summary.csv") else "not_ready",
        },
        {
            "claim": "multi-session replication",
            "minimum_evidence": "at least three fully processed sessions with consistent direction of effects and session-level uncertainty",
            "status": "supported" if n_full >= 3 else "not_ready",
        },
        {
            "claim": "layer-specific manifold geometry",
            "minimum_evidence": "at least three represented layers with cell-count-matched sessions and reliability/noise-ceiling controls",
            "status": "supported" if n_full >= 6 and exists_nonempty("14_layer_or_cre_geometry_statistics.csv") else "not_ready",
        },
        {
            "claim": "deep-model brain alignment",
            "minimum_evidence": "nonempty DNN/ViT alignment and encoding tables with pretrained-vs-untrained controls",
            "status": "supported" if exists_nonempty("15_brain_model_alignment_deep_features.csv") and exists_nonempty("15_dnn_to_population_encoding_summary.csv") else "not_ready",
        },
        {
            "claim": "cross-session shared manifold",
            "minimum_evidence": "nonempty cross-session manifold alignment table with at least three sessions and within/between-group tests",
            "status": "supported" if n_embedding >= 3 and exists_nonempty("16_pairwise_cross_session_manifold_alignment.csv") else "not_ready",
        },
    ]
    return pd.DataFrame(rows)
