from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v1_manifold_publication.paths import PublicationPaths, load_yaml  # noqa: E402
from v1_manifold_publication.cohort import (  # noqa: E402
    discover_existing_session_files,
    select_publication_cohort,
)
from v1_manifold_publication.multisession import build_session_run_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a read-only multi-session run plan.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "publication_upgrade.yaml"))
    parser.add_argument("--metadata", default=str(PROJECT_ROOT / "data" / "external" / "allen_v1_natural_movie_experiments.csv"))
    args = parser.parse_args()

    import pandas as pd

    cfg = load_yaml(args.config)
    paths = PublicationPaths.from_config(args.config)
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata table not found: {metadata_path}")

    experiments = pd.read_csv(metadata_path)
    cohort = select_publication_cohort(experiments, cfg)
    existing = discover_existing_session_files(paths.processed_dir, paths.interim_dir)
    plan = build_session_run_plan(cohort, existing)
    if plan.empty:
        print("No sessions found for the publication run plan.")
    else:
        cols = [
            c for c in [
                "session_id",
                "cre_line",
                "putative_layer",
                "imaging_depth",
                "has_tensor",
                "has_embedding",
                "needs_preprocessing",
                "needs_manifold",
                "ready_for_publication_stats",
            ]
            if c in plan.columns
        ]
        print(plan[cols].to_string(index=False))


if __name__ == "__main__":
    main()
