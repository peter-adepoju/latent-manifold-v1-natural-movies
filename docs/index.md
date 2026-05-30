# NeuroAI: V1 Natural Movie Manifold Geometry

This page introduces the project in a concise, public-facing way.

## Project summary

I analyze Allen Brain Observatory V1 calcium-imaging recordings from mice viewing natural movies. The goal is to understand whether population activity lives on a low-dimensional manifold and how that geometry changes with stimulus structure, representation choice, and neural organization.

## Current status

This is a serious exploratory NeuroAI project with:

- real AllenSDK/NWB data
- real stimulus features extracted from the movie frames
- leakage-aware decoding and block-wise evaluation
- manifold geometry summaries
- brain-model alignment analyses

The current repo is strongest as an honest research showcase. The broad multi-session biological claims are still being built up.

## What is in the repo

1. Data acquisition and validation
2. DeltaF/F preprocessing and trial tensor construction
3. PCA, UMAP, ISOMAP, and CEBRA embeddings
4. Geometry metrics such as speed, curvature, tangling, and intrinsic dimensionality
5. Stimulus decoding and encoding with null controls
6. RSA and CKA comparisons against analytic and deep visual features
7. A Streamlit app for exploring saved outputs

## Best starting points

- [Main README](../README.md)
- [Publication audit script](../scripts/publication_audit_readiness.py)
- [Results tables](../reports/tables/)
- [Figures](../reports/figures/)
- [Streamlit app](../app/streamlit_app.py)

## Why this project is worth looking at

The project is interesting because it combines public neuroscience data, careful controls, and modern representation-learning tools. It does not treat a pretty manifold plot as the conclusion. Instead, it asks whether latent geometry, stimulus statistics, and decoding performance remain meaningful under repeat-aware and block-aware evaluation.

## For later GitHub Pages setup

When this repository is published, this page can become the landing page for a simple project site. A good next step would be to add a few of the strongest generated figures to `docs/assets/` and link them here.

