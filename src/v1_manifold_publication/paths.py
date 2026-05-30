from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class PublicationPaths:
    root: Path
    data_dir: Path
    interim_dir: Path
    processed_dir: Path
    tables_dir: Path
    figures_dir: Path
    models_dir: Path
    publication_tables_dir: Path
    publication_figures_dir: Path

    @classmethod
    def from_config(cls, config_path: str | Path = "configs/publication_upgrade.yaml") -> "PublicationPaths":
        config_path = Path(config_path)
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        root = Path(cfg["paths"].get("existing_project_root", ".")).resolve()
        p = cfg["paths"]
        return cls(
            root=root,
            data_dir=root / p["data_dir"],
            interim_dir=root / p["interim_dir"],
            processed_dir=root / p["processed_dir"],
            tables_dir=root / p["tables_dir"],
            figures_dir=root / p["figures_dir"],
            models_dir=root / p["models_dir"],
            publication_tables_dir=root / p["publication_tables_dir"],
            publication_figures_dir=root / p["publication_figures_dir"],
        )

def ensure_publication_dirs(paths: PublicationPaths) -> None:
    for d in [paths.publication_tables_dir, paths.publication_figures_dir, paths.models_dir]:
        d.mkdir(parents=True, exist_ok=True)

def load_yaml(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_table(df, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
