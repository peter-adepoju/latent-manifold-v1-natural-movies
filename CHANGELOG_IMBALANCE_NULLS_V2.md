# Changelog — imbalance-aware real-stimulus decoding update

This update builds on `latent-manifold-v1-natural-movies-real-features.zip`. The previous version already extracted real frame-level visual features from the Allen natural movie pixels. This version makes the downstream interpretation more rigorous by addressing label imbalance and temporal autocorrelation.

## Changed files

### `src/v1_manifold/features.py`

Added:

- `summarize_class_balance(...)` for target count/fraction tables.
- `add_coarse_orientation_target(...)` to convert highly imbalanced fine orientation bins into a more interpretable coarse target such as `bin_3`, `bin_4`, and `other`.
- `safe_classification_target(...)` to skip classification targets that do not have enough samples per class.

### `src/v1_manifold/evaluation.py`

Added:

- adaptive stratified fold selection for imbalanced labels.
- `majority_class_baseline(...)`.
- `shuffled_label_baseline(...)`.
- `circular_shift_null_spearman(...)` for autocorrelated natural-movie time series.
- `stimulus_neural_circular_shift_table(...)` for table-based neural--stimulus correlation diagnostics.

### `src/v1_manifold/models.py`

Updated `evaluate_decoders(...)` so classification benchmarks now include:

- majority-class baseline,
- shuffled-label baseline,
- ridge classifier trained on shuffled labels,
- an `is_null_baseline` column in the benchmark table.

### `configs/default.yaml`

Added:

- `features.coarse_orientation_top_k`
- `evaluation.circular_shift_permutations`
- `evaluation.circular_shift_min_frames`
- `evaluation.include_null_baselines`

### `notebooks/04_exploratory_neural_data_analysis.ipynb`

Added default cells for:

- fine dominant-orientation class balance,
- coarse orientation class balance,
- real stimulus/neural Spearman checks,
- circular-shift null p-values,
- z-scored overlay of V1 population energy and real movie features,
- expanded EDA summary with class-balance statistics.

### `notebooks/06_baseline_decoding_models.ipynb`

Changed the decoding logic so that:

- continuous movie-feature decoding is treated as the primary analysis,
- coarse dominant-orientation decoding is used as the main classification target,
- fine 8-bin orientation is retained only as an exploratory target,
- unsupported/too-rare classification targets are skipped,
- classification results include null baselines and an `is_null_baseline` flag,
- the best confusion matrix excludes null baselines.

### `scripts/evaluate_models.py`

Updated the batch evaluation script to match notebook `06`.

### `tests/test_imbalance_and_nulls.py`

Added tests for:

- coarse orientation target construction,
- class-balance summaries,
- classification support checks,
- circular-shift null tests,
- null baselines in decoder benchmarks.

## Why this matters

The previous real-feature version prevented fake labels, but real natural-movie orientation labels can still be strongly imbalanced and temporally autocorrelated. This version prevents overclaiming by making imbalance, null baselines, and autocorrelation-aware correlation tests part of the default workflow.
