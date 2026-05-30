# Publication upgrade pack for `latent-manifold-v1-natural-movies`

This add-on upgrades the existing single-session V1 natural-movie manifold project into a multi-session, hypothesis-driven, publication-oriented analysis package.

It is intentionally designed as an **extension layer**. It does not replace notebooks `00`--`10`; instead it adds new notebooks, scripts, tests, configuration, and manuscript-planning assets that can be copied into the existing project directory.

## What this upgrade adds

1. **Multi-session cohort construction** across VISp/V1 Allen Brain Observatory sessions.
2. **Layer/Cre-line analysis plan** for L2/3, L4, L5, and L6 comparisons.
3. **Enhanced real-movie features**, including temporal contrast, frame-difference energy, optical-flow-compatible placeholders, Fourier slope, orientation/spatial-frequency summaries, and optional Gabor-bank features.
4. **Encoding and decoding analyses** with held-out repeats, contiguous movie blocks, session-level aggregation, and null controls.
5. **Noise ceiling and reliability analyses** so decoding/encoding results are interpreted relative to repeat reliability.
6. **Cell-count-matched analyses** to prevent dimensionality and decoding claims from being confounded by population size.
7. **Brain-model alignment** with RSA, CKA, and optional CNN/ViT feature extraction.
8. **Cross-session manifold alignment** using Procrustes and CCA-style alignment diagnostics.
9. **Hierarchical/statistical summaries** with bootstrap confidence intervals, empirical permutation p-values, FDR correction, and optional mixed-effects modeling.
10. **Top-journal manuscript scaffold**, figure plan, reviewer-risk register, and reproducibility checklist.

## Where to copy this pack

Copy the contents of this directory into your existing project root:

```text
C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies
```

It will add:

```text
configs/publication_upgrade.yaml
notebooks/11_publication_cohort_and_qc.ipynb
notebooks/12_enhanced_stimulus_features.ipynb
notebooks/13_multisession_encoding_decoding_and_nulls.ipynb
notebooks/14_layer_geometry_statistics.ipynb
notebooks/15_brain_model_alignment_rsa_cka.ipynb
notebooks/16_cross_session_manifold_alignment.ipynb
notebooks/17_publication_figures_and_manuscript_assets.ipynb
scripts/publication_*.py
src/v1_manifold_publication/*.py
docs/publication_upgrade/*.md
manuscript/top_journal_scaffold/*.md
```

Your existing notebooks remain intact.

## First commands after copying

```bash
pip install -r requirements-publication.txt
python scripts/publication_validate_environment.py
pytest tests/test_publication_upgrade.py
```

Then run the new notebooks in order from `11` to `17`.

## Safe audit mode

The publication-upgrade tooling now includes a read-only audit command:

```bash
python scripts/publication_audit_readiness.py
```

This inspects the current cohort, tables, and generated-artifact inventory
without overwriting `data/`, `models/`, or `reports/`. The script
`publication_run_all.py` is also read-only by default; pass `--write-derived`
only when you intentionally want to refresh lightweight derived publication
tables from already saved files.

## Publication planning commands

The upgrade now includes read-only planning commands for the major manuscript
gates:

```bash
make publication-plan       # session-level preprocessing/manifold run plan
make publication-cebra-plan # CEBRA-Time/Stimulus/Hybrid variant availability
make publication-dnn-plan   # planned pretrained/untrained DNN feature files
```

Longer-horizon latent dynamics can be evaluated as a dry run with:

```bash
python scripts/publication_run_latent_dynamics.py
```

Add `--write-output` only when you want to save a new versioned table under
`reports/tables/publication_upgrade_v2/`.

Heavy optional jobs are also guarded:

```bash
python scripts/publication_extract_dnn_features.py --frames-npy path/to/frames.npy --session-id 500855614
python scripts/publication_run_cebra_variants.py --session-id 500855614 --variant all
```

Both commands dry-run by default. Add `--write-output` only when you want to
create new versioned files under `data/processed/publication_upgrade_v2/` and,
for CEBRA, `models/publication_upgrade_v2/`.

## Important scientific note

This package is designed to make strong results possible and defensible, not to manufacture them. A scientifically relevant outcome can be positive or negative: for example, layer-specific manifold geometry would be strong, but a robust finding that low-dimensional visualizations fail under strict block-wise generalization would also be scientifically meaningful.
