"""Script-level runner for publication upgrade checks.

By default this script is read-only: it audits what is already on disk without
rewriting generated data, models, figures, or result tables. Pass
``--write-derived`` only when you intentionally want to refresh lightweight
derived publication tables from saved embeddings.
"""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v1_manifold_publication.readiness import build_readiness_report, format_readiness_markdown

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-derived",
        action="store_true",
        help="Refresh lightweight derived publication tables from existing saved files.",
    )
    args = parser.parse_args()

    if not args.write_derived:
        print(format_readiness_markdown(build_readiness_report(Path.cwd())))
        raise SystemExit(0)

    from v1_manifold_publication.pipeline import (
        validate_environment,
        build_existing_session_index,
        reconstruct_geometry_from_embeddings,
    )

    print("Validating environment...")
    print(validate_environment(Path.cwd()).to_string(index=False))
    print("\nIndexing existing session files...")
    print(build_existing_session_index().to_string(index=False))
    print("\nReconstructing geometry summaries from saved embeddings...")
    print(reconstruct_geometry_from_embeddings().head().to_string(index=False))
