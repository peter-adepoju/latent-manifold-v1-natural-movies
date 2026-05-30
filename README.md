# Latent Manifold Geometry of V1 Population Codes During Natural Movie Viewing

This project analyzes Allen Brain Observatory 2-photon calcium imaging recordings from mouse primary visual cortex (V1 / VISp) during natural movie viewing. It extracts real stimulus features from the presented movie frames, builds low-dimensional neural representations, quantifies manifold geometry, and compares V1 population codes with artificial vision representations.

This repository is also being prepared as part of a broader **NeuroAI** project collection. The goal is to keep related neuroscience and representation-learning projects under one public umbrella over time.

## Project at a glance

- **Scientific area:** computational neuroscience, systems neuroscience, and NeuroAI
- **Dataset:** Allen Brain Observatory Visual Coding 2-photon calcium imaging
- **Primary question:** does V1 activity during natural movie viewing occupy a low-dimensional manifold, and how does that geometry relate to stimulus statistics and neural representation choices?
- **Current status:** strong exploratory results, with publication-grade multi-session claims still in progress

## What the project does

The workflow includes:

- AllenSDK/NWB ingestion and session filtering
- DeltaF/F preprocessing and trial tensor construction
- Real movie-frame feature extraction from the actual Allen stimulus
- Low-dimensional embeddings such as PCA, UMAP, ISOMAP, and CEBRA
- Geometry metrics such as intrinsic dimensionality, curvature, speed, tangling, and persistent homology
- Stimulus decoding and encoding with held-out-repeat and block-wise controls
- Brain-model alignment with RSA and CKA
- Read-only publication auditing and manuscript planning

## What is scientifically interesting so far

The current analyses show that:

- V1 population activity carries information about real natural-movie features.
- Some latent representations preserve useful structure, but block-wise generalization is often modest.
- Geometry events in learned manifolds can align with visually interpretable movie segments, but the evidence is descriptive rather than causal.
- The project has careful null controls and leakage checks, which makes the exploratory findings more trustworthy.

## What is not yet claimed

The repository does **not** yet support strong broad claims about:

- multi-session replication across a large cohort
- layer-specific biological conclusions
- cross-session shared manifolds
- deep-model alignment at publication strength

Those are active next-step goals rather than finished conclusions.

## How to explore

- Open the project notes in `docs/index.md`
- Review generated figures in `reports/figures/`
- Inspect metric tables in `reports/tables/`
- Launch the Streamlit app after generating embeddings:

```bash
streamlit run app/streamlit_app.py
```

## Main workflow

```bash
make data
make preprocess
make manifold
make eval
make report
make app
```

Or run the full pipeline directly:

```bash
python scripts/run_all.py --config configs/default.yaml
```

## Publication audit

The repository includes a read-only audit that summarizes current evidence and remaining blockers:

```bash
python scripts/publication_audit_readiness.py
```

This is the safest way to inspect the project before rerunning expensive analyses.

## Key project outputs

- `data/interim/session_*_natural_movie_one_tensor.h5`
- `data/processed/session_*_natural_movie_one_real_frame_features.csv`
- `data/processed/session_*_embeddings.npz`
- `models/session_*_pca.joblib`
- `models/session_*_decoder_*.joblib`
- `reports/tables/06_baseline_decoding_benchmark.csv`
- `reports/tables/08_integrated_benchmark_table.csv`
- `reports/figures/07_cebra_embedding_3d.png`
- `reports/figures/08_model_comparison.png`
- `reports/figures/09_brain_model_similarity.png`

## Project structure

```text
app/            Streamlit explorer
configs/        YAML configuration files
data/           raw, interim, processed, and external data
docs/           GitHub Pages scaffold
models/         fitted embeddings and decoders
notebooks/      notebook-first workflow
reports/        figures, tables, and HTML outputs
scripts/        command-line pipeline entry points
src/            reusable analysis package
tests/          validation and regression tests
```

## Reproducibility

The codebase includes tests for preprocessing, geometry metrics, latent-feature validation, and leakage checks. The publication audit and versioned output folders are designed to keep expensive results from being overwritten accidentally.

## Contact with the work

If you are reading this as a future collaborator or reviewer, the most useful entry points are:

1. `docs/index.md`
2. `reports/tables/`
3. `reports/figures/`
4. `scripts/publication_audit_readiness.py`

