# Publication readiness checklist

Use this checklist before drafting claims as results.

## Cohort

- At least 12 processed VISp sessions.
- At least 3 putative layers represented.
- At least 3 mice represented.
- Session, mouse, Cre line, depth, and cell-count metadata attached to every
  row of every manuscript table.

## Reliability and noise ceilings

- Trial tensor layout verified as `[repeat, frame, cell]` before reliability.
- Split-half reliability reported per session.
- Noise ceilings reported for decoding and encoding interpretation.
- Low-reliability sessions flagged rather than silently pooled.

## Decoding and encoding

- Continuous movie features prioritized over imbalanced orientation classes.
- Held-out repeats used for raw population decoding.
- Contiguous movie-block CV used for frame-level latent analyses.
- Frame-index, Fourier-time, population-energy, shuffled-label, circular-shift,
  and block-permutation nulls included where appropriate.

## Geometry

- PCA/UMAP/ISOMAP/CEBRA geometry summarized at session level.
- Cell-count-matched subsampling used for layer/Cre-line comparisons.
- Geometry metrics reported with bootstrap confidence intervals.
- Persistent homology treated as supplementary unless replicated robustly.

## Brain-model alignment

- Analytic visual features included.
- Pretrained and untrained DNN/ViT/video-model controls included.
- RSA and CKA reported with session-level aggregation.
- Encoding performance reported alongside representational similarity.

## Manuscript discipline

- Single-session results labeled exploratory.
- Causal language avoided unless perturbational evidence exists.
- Figures generated only from audited source tables.
- All expensive generated outputs protected from accidental overwrite.
