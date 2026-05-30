# Top-journal upgrade roadmap

## Target manuscript claim

The most defensible top-journal claim is not that a single embedding looks
structured. The stronger claim is:

> V1 responses to natural movies contain repeat-reliable, layer-dependent
> population geometry whose relationship to visual statistics can be separated
> from temporal autocorrelation, population size, and low-level image baselines.

## Non-negotiable evidence gates

1. **Multi-session replication:** process enough VISp sessions to support
   session-level statistics across mice, layers, and Cre lines.
2. **Correct repeat reliability:** compute split-half and noise-ceiling metrics
   on tensors interpreted as `[repeat, frame, cell]`.
3. **Held-out repeat decoding:** keep raw-cell continuous-feature decoding as a
   primary result, normalized by reliability where possible.
4. **Held-out movie block generalization:** report block-CV performance for
   latent representations, time baselines, population-energy baselines, and
   shuffled/circular/block-permutation nulls.
5. **Cell-count matching:** compare layers and Cre lines only after repeated
   matched-cell subsampling.
6. **Brain-model alignment controls:** compare analytic low-level features,
   pretrained DNN/ViT/video features, and untrained/random controls.
7. **Cross-session alignment:** quantify whether manifold structure is stable
   across sessions before claiming a shared manifold.
8. **Cautious geometry interpretation:** treat speed, curvature, tangling, and
   topology as descriptive unless they predict held-out responses or survive
   null controls.

## Priority work order

1. Fix or confirm all reliability and tensor-layout assumptions.
2. Expand preprocessing from one processed session to a balanced VISp cohort.
3. Run publication notebooks `11` through `17` into a new versioned output
   namespace.
4. Add deep-model feature extraction and untrained-model controls.
5. Add layer/Cre-line mixed-effects or hierarchical bootstrap summaries.
6. Build final figures only from the audited publication tables.
7. Write the manuscript around the strongest positive or negative result that
   survives the gates above.

## Project components now mapped to this roadmap

- Rich temporal/motion/movie features:
  `src/v1_manifold_publication/stimulus_features.py`
- Encoding models and reduced-rank population prediction:
  `src/v1_manifold_publication/encoding.py`
- Empirical null summaries:
  `src/v1_manifold_publication/decoding.py`
- Session-level bootstrap, FDR, and mixed-effects summaries:
  `src/v1_manifold_publication/stats.py`
- Cell-count matched analysis helpers:
  `src/v1_manifold_publication/cell_matching.py`
- CEBRA-Time/Stimulus/Hybrid planning:
  `src/v1_manifold_publication/cebra_variants.py`
- Multi-horizon latent dynamics baselines:
  `src/v1_manifold_publication/dynamics.py`
- DNN feature extraction planning:
  `src/v1_manifold_publication/dnn_features.py`
- Multi-session run planning and claim gates:
  `src/v1_manifold_publication/multisession.py`

Read-only planning commands:

```bash
make publication-audit
make publication-plan
make publication-cebra-plan
make publication-dnn-plan
python scripts/publication_run_latent_dynamics.py
```

## Strong outcomes worth publishing

- Layer-specific geometry after cell-count matching and reliability correction.
- Layer-specific DNN/analytic feature alignment that replicates across mice.
- Geometry-event enrichment for visual transitions that survives circular-shift
  and block-permutation controls.
- A rigorous negative result showing that visually compelling manifolds fail
  under strict block-wise generalization while raw populations retain reliable
  visual-feature information.
