# Changes in the real-stimulus-feature version

This version replaces placeholder/fallback decoding labels with deterministic visual features extracted from the actual Allen natural movie frames.

## Main scientific change

The previous ZIP allowed the decoding pipeline to run with fallback labels such as `orientation_bin = frame_index % n_bins`. That was useful for smoke testing, but it was not scientifically interpretable. This version refuses to use fallback labels for real decoding. Notebook 04 now loads the Allen natural movie template and computes real frame-level image features from the movie pixels.

## Files changed

### `src/v1_manifold/features.py`

- Added `build_real_movie_feature_table(...)`.
- Added Fourier-domain stimulus feature extraction for each natural movie frame.
- Added real columns:
  - `luminance_mean`
  - `luminance_std`
  - `rms_contrast`
  - `dominant_orientation_bin`
  - `dominant_orientation_angle_rad`
  - `orientation_selectivity`
  - `spatial_frequency_centroid`
  - `spatial_frequency_bin`
  - `total_spectral_power`
  - normalized orientation/spatial-frequency energy columns
- Added `assert_real_stimulus_features(...)` so final decoding fails if fallback labels are present.
- Added `make_single_trial_design_matrix(...)` for repeat-wise held-out decoding from `[repeat, cell, frame]` tensors.

### `configs/default.yaml`

- Changed `min_mean_dff` from `0.10` to `0.001` so valid Allen calcium traces are not discarded too aggressively.
- Changed `zscore_eps` to `1.0e-6`.
- Added:
  - `stimulus_downsample_factor: 4`
  - `require_real_stimulus_features: true`

### `notebooks/04_exploratory_neural_data_analysis.ipynb`

- Now reloads the Allen experiment object and calls `get_stimulus_template(...)`.
- Extracts real movie-frame features from the Allen natural movie template.
- Saves both:
  - `*_real_frame_features.csv`
  - `*_frame_features.csv`
- Adds an assertion preventing fallback labels from entering the workflow.
- Adds real stimulus-feature diagnostic plots.
- Adds an EDA summary table including whether fallback labels were used.

### `notebooks/05_feature_extraction_and_latent_representations.ipynb`

- Now requires the real feature table from notebook 04.
- Adds latent-trajectory plots colored by real visual statistics such as RMS contrast and spatial-frequency centroid.

### `notebooks/06_baseline_decoding_models.ipynb`

- Replaces `orientation_bin` with `dominant_orientation_bin`.
- Uses `spatial_frequency_bin` derived from actual movie-frame spectra.
- Adds repeat-wise single-trial decoding from raw neural activity.
- Adds continuous regression targets:
  - `rms_contrast`
  - `spatial_frequency_centroid`
  - `orientation_selectivity`
- Saves both classification and continuous-feature decoding benchmarks.

### `notebooks/08_model_evaluation_and_comparison.ipynb`

- Updates integrated benchmark logic to handle real visual-feature targets.
- Updates figures to refer to real feature decoding rather than placeholder orientation decoding.

### `notebooks/10_report_generation.ipynb`

- Updates the summary language so decoding claims are about image-derived visual statistics.

### `scripts/preprocess_sessions.py`

- Extracts real movie-frame visual features during script-based preprocessing.
- Fails if the Allen movie template is missing and `require_real_stimulus_features: true`.

### `scripts/run_manifold_analysis.py`

- Requires real frame features before running manifold analysis.
- Stops creating fallback feature labels during manifold execution.

### `scripts/evaluate_models.py`

- Evaluates real visual-feature classification and continuous-feature regression.
- Adds repeat-wise held-out single-trial decoding from raw cells.

### `src/v1_manifold/models.py` and `src/v1_manifold/evaluation.py`

- Added regression baselines and grouped regression splits.
- Added continuous-feature decoding metrics.

### `tests/`

- Added tests that real movie features are non-fallback and contain expected columns.
- Added tests that fallback labels are rejected by `assert_real_stimulus_features`.
- Added tests for single-trial design-matrix construction.
- Added a regression-decoder smoke test.

## What is different from the last ZIP

The last ZIP was structurally complete but still allowed a decoding smoke test with synthetic frame labels. This version makes the decoding scientifically meaningful by deriving targets from the actual Allen natural movie pixels and by refusing to run final decoding on placeholders.

The default workflow is now stricter. If AllenSDK cannot load the stimulus template, notebook 04 and script preprocessing raise an error instead of silently continuing with fake labels.
