from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


GENERATED_ARTIFACT_DIRS = {"data", "models", "reports"}

EXPECTED_PUBLICATION_TABLES = (
    "11_publication_candidate_cohort.csv",
    "11_noise_ceiling_summary.csv",
    "13_latent_feature_decoding_blockcv_summary.csv",
    "13_stimulus_to_population_encoding_summary.csv",
    "14_geometry_with_session_metadata.csv",
    "15_brain_model_alignment_analytic_features.csv",
    "15_brain_model_alignment_deep_features.csv",
    "16_pairwise_cross_session_manifold_alignment.csv",
)


@dataclass(frozen=True)
class CsvShape:
    rows: int
    columns: int
    error: str | None = None


def is_generated_artifact_path(path: str | Path, project_root: str | Path = ".") -> bool:
    """Return True if a path lives under generated, expensive-artifact folders."""
    root = Path(project_root).resolve()
    target = Path(path).resolve()
    try:
        rel = target.relative_to(root)
    except ValueError:
        return False
    return bool(rel.parts and rel.parts[0] in GENERATED_ARTIFACT_DIRS)


def require_non_generated_output_path(path: str | Path, project_root: str | Path = ".") -> None:
    """Guard scripts from accidentally writing into generated artifact folders."""
    if is_generated_artifact_path(path, project_root=project_root):
        raise ValueError(
            f"Refusing to write into generated-artifact path: {Path(path)}. "
            "Use an explicit regeneration script when you intend to update data, models, or reports."
        )


def csv_shape(path: str | Path) -> CsvShape:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            rows = sum(1 for _ in reader)
        return CsvShape(rows=rows, columns=len(header))
    except Exception as exc:
        return CsvShape(rows=0, columns=0, error=repr(exc))


def read_csv_dicts(path: str | Path) -> list[dict[str, str]]:
    try:
        with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _unique_nonempty(rows: list[dict[str, str]], column: str) -> list[str]:
    values = {str(row.get(column, "")).strip() for row in rows}
    return sorted(v for v in values if v and v.lower() != "nan")


def _sessions_from_patterns(root: Path) -> list[str]:
    sessions: set[str] = set()
    for folder, pattern in [
        (root / "data" / "interim", "session_*_tensor.h5"),
        (root / "data" / "processed", "session_*_embeddings.npz"),
        (root / "data" / "processed", "session_*_frame_features.csv"),
    ]:
        for path in folder.glob(pattern):
            parts = path.name.split("_")
            if len(parts) > 1:
                sessions.add(parts[1])
    return sorted(sessions)


def build_readiness_report(
    project_root: str | Path = ".",
    min_sessions: int = 12,
    min_layers: int = 3,
) -> dict:
    """Build a read-only publication-readiness report from files on disk."""
    root = Path(project_root).resolve()
    pub_tables = root / "reports" / "tables" / "publication_upgrade"
    cohort_path = pub_tables / "11_publication_candidate_cohort.csv"
    cohort_rows = read_csv_dicts(cohort_path)

    cohort_sessions = _unique_nonempty(cohort_rows, "session_id")
    layers = _unique_nonempty(cohort_rows, "putative_layer")
    cre_lines = _unique_nonempty(cohort_rows, "cre_line")
    processed_sessions = _sessions_from_patterns(root)

    table_status = []
    for name in EXPECTED_PUBLICATION_TABLES:
        path = pub_tables / name
        shape = csv_shape(path) if path.exists() else CsvShape(0, 0, "missing")
        table_status.append({
            "table": name,
            "exists": path.exists(),
            "rows": shape.rows,
            "columns": shape.columns,
            "error": shape.error,
            "nonempty": path.exists() and shape.rows > 0 and shape.columns > 0,
        })

    inventory = {
        "raw_nwb_files": len(list((root / "data" / "raw").glob("**/*.nwb"))),
        "interim_tensors": len(list((root / "data" / "interim").glob("session_*_tensor.h5"))),
        "processed_embeddings": len(list((root / "data" / "processed").glob("session_*_embeddings.npz"))),
        "processed_feature_tables": len(list((root / "data" / "processed").glob("session_*frame_features.csv"))),
        "model_files": len(list((root / "models").glob("*"))),
        "report_tables": len(list((root / "reports" / "tables").glob("*.csv"))),
        "publication_tables": len(list(pub_tables.glob("*.csv"))),
        "figures": len(list((root / "reports" / "figures").glob("*.png"))),
    }

    blockers: list[str] = []
    if len(processed_sessions) < min_sessions:
        blockers.append(
            f"Only {len(processed_sessions)} processed session(s); target at least {min_sessions}."
        )
    if len(layers) < min_layers:
        blockers.append(f"Only {len(layers)} represented layer(s); target at least {min_layers}.")
    if not any(t["table"] == "16_pairwise_cross_session_manifold_alignment.csv" and t["nonempty"] for t in table_status):
        blockers.append("Cross-session manifold alignment table is empty or missing.")
    if not any(t["table"] == "15_brain_model_alignment_deep_features.csv" and t["nonempty"] for t in table_status):
        blockers.append("Deep model alignment table is empty or missing.")
    if not any(t["table"] == "13_stimulus_to_population_encoding_summary.csv" and t["nonempty"] for t in table_status):
        blockers.append("Stimulus-to-population encoding summary is empty or missing.")

    if blockers:
        tier = "exploratory_or_scaffold"
    elif len(processed_sessions) >= min_sessions and len(layers) >= min_layers:
        tier = "candidate_top_journal_analysis_package"
    else:
        tier = "multi_session_in_progress"

    return {
        "project_root": str(root),
        "inventory": inventory,
        "cohort": {
            "cohort_sessions": cohort_sessions,
            "processed_sessions": processed_sessions,
            "n_cohort_sessions": len(cohort_sessions),
            "n_processed_sessions": len(processed_sessions),
            "layers": layers,
            "cre_lines": cre_lines,
        },
        "publication_tables": table_status,
        "blockers": blockers,
        "tier": tier,
    }


def format_readiness_markdown(report: dict) -> str:
    lines = [
        "# Publication Readiness Audit",
        "",
        f"Project root: `{report['project_root']}`",
        f"Current tier: `{report['tier']}`",
        "",
        "## Inventory",
    ]
    for key, value in report["inventory"].items():
        lines.append(f"- `{key}`: {value}")

    cohort = report["cohort"]
    lines.extend([
        "",
        "## Cohort",
        f"- Processed sessions: {cohort['n_processed_sessions']} ({', '.join(cohort['processed_sessions']) or 'none'})",
        f"- Cohort sessions: {cohort['n_cohort_sessions']} ({', '.join(cohort['cohort_sessions']) or 'none'})",
        f"- Layers: {', '.join(cohort['layers']) or 'none'}",
        f"- Cre lines: {', '.join(cohort['cre_lines']) or 'none'}",
        "",
        "## Publication Tables",
    ])
    for row in report["publication_tables"]:
        status = "ok" if row["nonempty"] else "missing_or_empty"
        lines.append(f"- `{row['table']}`: {status}, rows={row['rows']}, columns={row['columns']}")

    lines.append("")
    lines.append("## Blocking Issues")
    if report["blockers"]:
        lines.extend(f"- {item}" for item in report["blockers"])
    else:
        lines.append("- None detected by the read-only audit.")
    lines.append("")
    return "\n".join(lines)
