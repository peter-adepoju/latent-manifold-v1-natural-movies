# V3 changelog — confidence-aware orientation labels

This version updates the real-feature V2 repository after notebook 04 showed that many natural movie frames have ambiguous dominant orientation.

## Changed from V2

1. Added `dominant_orientation_confident` and `orientation_confidence` to the saved frame-feature tables. Low-orientation-selectivity frames are labeled `ambiguous`.
2. Added `add_confident_orientation_target`, `balanced_confident_orientation_subset`, and `orientation_label_interpretation_table` to `src/v1_manifold/features.py`.
3. Updated notebook 04 to save confidence-aware orientation labels, class-balance tables, an orientation-label interpretation table, and representative movie-frame audits.
4. Updated notebook 06 to treat continuous decoding as primary, coarse/confident orientation decoding as secondary/exploratory, and to add a balanced confident-only orientation decoding analysis.
5. Updated `scripts/preprocess_sessions.py` and `scripts/evaluate_models.py` so command-line workflows match the notebooks.
6. Updated config with orientation confidence parameters.
7. Added tests for confidence-aware orientation labels and balanced confident-only subsets.
8. Updated README, docs, and report-generation notes to prevent overclaiming hard orientation-bin decoding.

## Scientific reason

Natural movie frames often contain mixed edges, shadows, objects, and viewpoints. A single hard dominant-orientation class can be unstable when orientation selectivity is weak. This version makes that uncertainty explicit and keeps the primary scientific focus on continuous visual-feature decoding and latent manifold geometry.
