from pathlib import Path

from v1_manifold_publication.readiness import (
    build_readiness_report,
    csv_shape,
    format_readiness_markdown,
    require_non_generated_output_path,
)


def test_csv_shape_counts_rows_and_columns(tmp_path):
    p = tmp_path / "table.csv"
    p.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    shape = csv_shape(p)
    assert shape.rows == 2
    assert shape.columns == 2


def test_generated_artifact_guard_rejects_data_reports_models(tmp_path):
    root = tmp_path
    for rel in ["data/x.csv", "reports/tables/x.csv", "models/x.joblib"]:
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            require_non_generated_output_path(target, project_root=root)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected guard to reject {rel}")

    safe = root / "docs" / "audit.md"
    require_non_generated_output_path(safe, project_root=root)


def test_readiness_report_detects_scaffold_blockers(tmp_path):
    root = Path(tmp_path)
    pub = root / "reports" / "tables" / "publication_upgrade"
    processed = root / "data" / "processed"
    interim = root / "data" / "interim"
    pub.mkdir(parents=True)
    processed.mkdir(parents=True)
    interim.mkdir(parents=True)

    (processed / "session_1_embeddings.npz").write_bytes(b"placeholder")
    (interim / "session_1_natural_movie_one_tensor.h5").write_bytes(b"placeholder")
    (pub / "11_publication_candidate_cohort.csv").write_text(
        "session_id,putative_layer,cre_line\n1,L2/3,Cux2\n",
        encoding="utf-8",
    )
    (pub / "13_latent_feature_decoding_blockcv_summary.csv").write_text(
        "session_id,representation,target,r2_mean\n1,pca,rms_contrast,-1\n",
        encoding="utf-8",
    )

    report = build_readiness_report(root, min_sessions=2, min_layers=2)
    assert report["tier"] == "exploratory_or_scaffold"
    assert report["blockers"]
    md = format_readiness_markdown(report)
    assert "Publication Readiness Audit" in md
    assert "Cross-session manifold alignment" in md
