from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v1_manifold_publication.cebra_variants import cebra_variant_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only CEBRA variant planning.")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    args = parser.parse_args()

    import pandas as pd

    processed_dir = Path(args.processed_dir)
    feature_files = sorted(processed_dir.glob("session_*frame_features.csv"))
    if not feature_files:
        raise FileNotFoundError(f"No frame-feature tables found in {processed_dir}")
    for feature_path in feature_files:
        session_id = feature_path.name.split("_")[1]
        features = pd.read_csv(feature_path)
        plan = cebra_variant_plan(features)
        plan.insert(0, "session_id", session_id)
        plan.insert(1, "feature_file", str(feature_path))
        print(f"\nSession {session_id}: {feature_path.name}")
        print(plan.to_string(index=False))


if __name__ == "__main__":
    main()
