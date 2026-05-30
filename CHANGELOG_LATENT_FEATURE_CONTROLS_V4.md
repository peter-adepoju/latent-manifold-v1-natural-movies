# V4 changelog — latent-feature controls for notebook 05

This update integrates the improved notebook 05 into the full project repository.

## Main changes

1. Replaced `notebooks/05_feature_extraction_and_latent_representations.ipynb` with the updated version.
2. Fixed embedding colorbar labels so contrast and spatial-frequency plots are not mislabeled as movie-frame plots.
3. Added contiguous movie-block cross-validation for latent-feature regression.
4. Added temporal baselines:
   - linear frame index,
   - Fourier frame-index features,
   - scalar population L2 norm.
5. Added reusable source functions in `src/v1_manifold/evaluation.py`:
   - `make_contiguous_frame_blocks`,
   - `frame_time_features`,
   - `evaluate_feature_regression`,
   - `latent_gain_over_temporal_baseline`.
6. Updated `src/v1_manifold/visualization.py` so `plot_embedding_2d` accepts a custom `colorbar_label`.
7. Added `scripts/evaluate_latent_feature_predictivity.py` for command-line reproduction of the notebook 05 feature-predictivity tables.
8. Updated `Makefile` with a `latent-eval` target and made `make eval` run latent-feature controls before model evaluation.
9. Added tests for block-wise feature regression, time baselines, latent-vs-time comparison, and colorbar-label support.
10. Updated README and GitHub Pages notes with the new interpretation: a structured manifold must be compared against temporal baselines before claiming stimulus-feature organization.

## Scientific reason

UMAP and ISOMAP produce smooth trajectories for the Allen natural movie response, but smoothness alone can reflect temporal autocorrelation in the movie. The V4 update makes notebook 05 quantify whether latent coordinates predict real movie-frame features beyond time/frame-index baselines.
