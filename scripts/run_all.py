from __future__ import annotations

import argparse
import subprocess
import sys

STEPS = [
    ["scripts/download_allen_data.py"],
    ["scripts/preprocess_sessions.py"],
    ["scripts/run_manifold_analysis.py"],
    ["scripts/evaluate_models.py"],
    ["scripts/build_report.py"],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args()
    steps = STEPS[1:] if args.skip_data else STEPS
    for step in steps:
        cmd = [sys.executable, *step, "--config", args.config]
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
