from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import random
import os

import numpy as np
import yaml


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_dir: Path
    interim_dir: Path
    processed_dir: Path
    external_metadata: Path
    allen_manifest: Path
    figures_dir: Path
    tables_dir: Path
    models_dir: Path


def load_config(config_path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load my YAML configuration and resolve it as a plain dictionary."""
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(config_path)
    return cfg


def get_project_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg.get("paths", {}).get("root", ".")).expanduser().resolve()
    return root


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    root = get_project_root(cfg)
    value = cfg["paths"][key]
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path


def get_paths(cfg: dict[str, Any]) -> ProjectPaths:
    root = get_project_root(cfg)
    paths = ProjectPaths(
        root=root,
        raw_dir=resolve_path(cfg, "raw_dir"),
        interim_dir=resolve_path(cfg, "interim_dir"),
        processed_dir=resolve_path(cfg, "processed_dir"),
        external_metadata=resolve_path(cfg, "external_metadata"),
        allen_manifest=resolve_path(cfg, "allen_manifest"),
        figures_dir=resolve_path(cfg, "figures_dir"),
        tables_dir=resolve_path(cfg, "tables_dir"),
        models_dir=resolve_path(cfg, "models_dir"),
    )
    for path in [paths.raw_dir, paths.interim_dir, paths.processed_dir, paths.external_metadata.parent, paths.allen_manifest.parent, paths.figures_dir, paths.tables_dir, paths.models_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def set_global_seed(seed: int = 42) -> None:
    """Set the random seeds I use across NumPy, Python, and torch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def select_device(requested: str = "auto") -> str:
    if requested == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return requested
