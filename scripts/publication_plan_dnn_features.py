from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v1_manifold_publication.dnn_features import planned_dnn_outputs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only DNN feature extraction plan.")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--run-label", default="publication_upgrade_v2")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = processed_dir / args.run_label
    sessions = sorted({p.name.split("_")[1] for p in processed_dir.glob("session_*frame_features.csv")})
    if not sessions:
        raise FileNotFoundError(f"No session frame-feature tables found in {processed_dir}")
    print("Planned DNN feature outputs. This script is read-only and does not extract features.")
    print("Output directory for an explicit extraction run would be:", output_dir)
    for session_id in sessions:
        print(f"\nSession {session_id}")
        for out in planned_dnn_outputs(session_id, output_dir):
            print(" ", out)


if __name__ == "__main__":
    main()
