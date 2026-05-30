# Reviewer risk register

| Risk | Why it matters | Mitigation in this upgrade |
|---|---|---|
| Single-session overclaiming | Reviewers will not accept general claims from one session. | Multi-session cohort, session-level statistics, held-out mouse/session analyses. |
| Temporal autocorrelation | Natural movie frames are not independent. | Contiguous block CV, circular-shift nulls, frame-index baselines. |
| Label imbalance | Dominant orientation labels can be highly imbalanced. | Class-balance tables, balanced accuracy, null baselines, continuous features prioritized. |
| Cell-count confound | Dimensionality/decoding can scale with number of neurons. | Cell-count matching and subsampling. |
| Pretty manifolds but weak prediction | Visual embeddings can be misleading. | Predictive tests, encoding/decoding, nulls, brain-model alignment. |
| CEBRA/dRNN overinterpretation | Time-conditioned embeddings and one-step dynamics can reflect smoothness. | Persistence baselines, longer-horizon dRNN extensions, explicit caveats. |
| No topological conclusion | Persistent homology can be noisy and hard to interpret. | Treat topology as supplementary only. |
| Tensor-layout reliability bug | A `[repeat, cell, frame]` tensor can be misread as `[repeat, frame, cell]`, corrupting cell-level reliability. | Coerce and test tensor layout before split-half reliability and noise-ceiling estimates. |
| Accidental overwrite of expensive outputs | Rerunning notebooks or scripts can replace long-running derived artifacts. | Read-only audit defaults, explicit write flags, and versioned output namespaces. |
