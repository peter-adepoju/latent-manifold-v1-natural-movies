from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


NOTEBOOKS = [
    "12c_multisession_asset_preparation_and_driver.ipynb",
    "13_multisession_encoding_decoding_and_nulls.ipynb",
    "14_layer_geometry_statistics.ipynb",
    "15_brain_model_alignment_rsa_cka.ipynb",
    "16_cross_session_manifold_alignment.ipynb",
    "17_publication_figures_and_manuscript_assets.ipynb",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run publication multi-session upgrade notebooks.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-label", default="publication_upgrade_v3_multisession")
    parser.add_argument("--skip-12c", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually execute notebooks with nbconvert. Without this flag, only prints commands.")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    notebooks_dir = project_root / "notebooks"

    env = os.environ.copy()
    env["LATENT_MANIFOLD_PROJECT_ROOT"] = str(project_root)
    env["PUBLICATION_RUN_LABEL"] = args.run_label

    notebooks = NOTEBOOKS[1:] if args.skip_12c else NOTEBOOKS

    for notebook in notebooks:
        nb_path = notebooks_dir / notebook
        if not nb_path.exists():
            print(f"Missing notebook: {nb_path}")
            continue

        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(nb_path),
        ]

        print(" ".join(cmd))
        if args.execute:
            result = subprocess.run(cmd, cwd=project_root, env=env)
            if result.returncode != 0:
                return result.returncode

    if not args.execute:
        print("\nDry run only. Add --execute to run notebooks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
