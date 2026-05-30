from __future__ import annotations

from pathlib import Path
import pandas as pd


def collect_report_assets(figures_dir: str | Path, tables_dir: str | Path) -> dict[str, list[Path]]:
    figures_dir = Path(figures_dir)
    tables_dir = Path(tables_dir)
    return {
        "figures": sorted(figures_dir.glob("*.png")),
        "tables": sorted(list(tables_dir.glob("*.csv")) + list(tables_dir.glob("*.parquet"))),
    }


def write_markdown_report(path: str | Path, title: str, summary: str, assets: dict[str, list[Path]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", summary, "", "## Generated figures", ""]
    for fig in assets.get("figures", []):
        lines.append(f"- `{fig}`")
    lines.extend(["", "## Generated tables", ""])
    for table in assets.get("tables", []):
        lines.append(f"- `{table}`")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
