from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .paths import PublicationPaths, ensure_publication_dirs, load_yaml, save_table
from .cohort import discover_existing_session_files, select_publication_cohort, summarize_cohort
from .geometry_publication import geometry_timeseries, summarize_geometry_ts


def validate_environment(project_root=".") -> pd.DataFrame:
    root = Path(project_root)
    checks = []
    for rel in ["notebooks", "src", "data", "reports", "reports/tables", "reports/figures"]:
        p = root / rel
        checks.append({"path": rel, "exists": p.exists(), "type": "dir" if p.is_dir() else "file_or_missing"})
    try:
        import numpy, pandas, sklearn, scipy
        checks.append({"path": "core_python_dependencies", "exists": True, "type": "import"})
    except Exception:
        checks.append({"path": "core_python_dependencies", "exists": False, "type": "import"})
    return pd.DataFrame(checks)


def build_existing_session_index(config_path="configs/publication_upgrade.yaml") -> pd.DataFrame:
    cfg = load_yaml(config_path)
    paths = PublicationPaths.from_config(config_path)
    ensure_publication_dirs(paths)
    index = discover_existing_session_files(paths.processed_dir, paths.interim_dir)
    save_table(index, paths.publication_tables_dir / "existing_session_file_index.csv")
    return index


def reconstruct_geometry_from_embeddings(config_path="configs/publication_upgrade.yaml") -> pd.DataFrame:
    paths = PublicationPaths.from_config(config_path)
    rows = []
    for emb_path in paths.processed_dir.glob("session_*_embeddings.npz"):
        session_id = emb_path.name.split("_")[1]
        with np.load(emb_path, allow_pickle=False) as emb:
            for name in emb.files:
                Z = emb[name]
                if Z.ndim == 2 and Z.shape[0] > 10 and Z.shape[1] >= 2:
                    geom = geometry_timeseries(Z)
                    summary = summarize_geometry_ts(geom)
                    pivot = summary.set_index("metric")[["median", "p99", "max"]].stack().to_dict()
                    rows.append({"session_id": session_id, "embedding": name, "n_frames": Z.shape[0], "embedding_dim": Z.shape[1], **{f"{k[0]}_{k[1]}": v for k, v in pivot.items()}})
    df = pd.DataFrame(rows)
    save_table(df, paths.publication_tables_dir / "publication_reconstructed_embedding_geometry.csv")
    return df
