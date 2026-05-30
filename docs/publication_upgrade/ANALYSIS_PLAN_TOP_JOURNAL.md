# Analysis plan for a strong publication-grade version

## Primary hypothesis

V1 population responses to natural movies form structured neural manifolds whose geometry varies across cortical layers/Cre lines and predicts which naturalistic visual statistics are preserved in the population code.

## Primary analyses

1. Multi-session VISp/V1 cohort selection.
2. Reliability and noise-ceiling estimation from repeated movie presentations.
3. Continuous feature decoding from population activity with held-out repeats and held-out contiguous movie blocks.
4. Encoding models from real movie features and analytic/DNN visual features to neural responses.
5. Geometry summaries for PCA, CEBRA, UMAP, and ISOMAP embeddings.
6. Layer/Cre-line comparisons with cell-count matching.
7. RSA/CKA brain-model alignment with analytic and optional deep vision model representations.
8. Cross-session manifold alignment.
9. Event-triggered analysis of high-speed/curvature/tangling frames.

## Required controls

- Shuffled-label nulls.
- Circular-shift nulls.
- Block-permutation nulls.
- Frame-index and Fourier frame-index baselines.
- Population-energy-only baselines.
- Random projection and matched-dimensionality PCA controls.
- Persistence baseline for dRNN next-state prediction.
- Cell-count matched subsampling across layers.
- Session/mouse-level bootstrap confidence intervals.
- Explicit tensor-layout validation before repeat reliability.
- Versioned output namespaces before rerunning long analyses.

## Acceptance-level claims

A high-tier manuscript needs at least one strong result:

1. robust layer-specific manifold geometry after cell-count matching;
2. robust layer-specific brain-model alignment;
3. robust geometry-event enrichment for real movie transitions/features;
4. or a rigorous negative result showing where low-dimensional manifold visualizations fail under strict generalization tests.
