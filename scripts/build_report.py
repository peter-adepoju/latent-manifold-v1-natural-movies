from __future__ import annotations

import argparse
from v1_manifold.config import load_config, get_paths
from v1_manifold.reporting import collect_report_assets, write_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    assets = collect_report_assets(paths.figures_dir, paths.tables_dir)
    report_path = paths.root / "reports" / "html" / "project_summary.md"
    write_markdown_report(
        report_path,
        title="Latent manifold geometry of V1 population codes during natural image viewing",
        summary="This generated report indexes the figures and tables produced by my notebook workflow.",
        assets=assets,
    )
    print(f"Saved report index: {report_path}")


if __name__ == "__main__":
    main()
