from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v1_manifold_publication.readiness import (  # noqa: E402
    build_readiness_report,
    format_readiness_markdown,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read-only publication-readiness audit for the V1 natural-movie project."
    )
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--min-sessions", type=int, default=12)
    parser.add_argument("--min-layers", type=int, default=3)
    parser.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Exit with status 1 when manuscript-blocking issues are detected.",
    )
    args = parser.parse_args()

    report = build_readiness_report(
        args.project_root,
        min_sessions=args.min_sessions,
        min_layers=args.min_layers,
    )
    print(format_readiness_markdown(report))
    if args.fail_on_blockers and report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
