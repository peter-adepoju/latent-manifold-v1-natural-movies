# V5 update: embedding-file validation for notebook 06

This update fixes a failure in `06_baseline_decoding_models.ipynb` where the notebook looped over every array stored in `session_*_embeddings.npz`. The embeddings file can contain both true frame-level embeddings, such as `pca`, `umap`, `isomap`, and optional `cebra`, and metadata arrays, such as PCA explained-variance spectra or scalar dimensionality summaries. Passing metadata arrays to the decoder caused an `IndexError` because they are not two-dimensional sample-by-feature matrices.

## Changed files

- `src/v1_manifold/models.py`
  - Added `as_2d_feature_matrix(...)` to validate model input matrices.
  - Added `valid_embedding_names(...)` to keep only arrays shaped as `[n_movie_frames, n_latent_dimensions]`.
  - Added `describe_embedding_file(...)` to save/debug the contents of embedding `.npz` files.
  - Updated `evaluate_decoders(...)` and `evaluate_regressors(...)` to validate or reshape model inputs before fitting.

- `notebooks/06_baseline_decoding_models.ipynb`
  - Added an embedding-file summary table.
  - Saves `reports/tables/06_embedding_file_summary_session_<session_id>.csv`.
  - Replaced loops over `emb.files` with validated `embedding_names`.
  - Skips non-embedding arrays such as variance spectra before decoding.

- `scripts/evaluate_models.py`
  - Uses `valid_embedding_names(...)` and `describe_embedding_file(...)` before latent-frame decoding.
  - Saves the same embedding-file summary used by the notebook.
  - Skips sessions whose embedding files do not contain valid frame-level embeddings.

- `tests/test_embedding_file_validation.py`
  - Tests that metadata arrays are skipped.
  - Tests that valid frame-level embeddings are retained.
  - Tests that one-dimensional aligned feature vectors are reshaped safely.
  - Tests that mismatched metadata vectors raise a clear `ValueError`.
  - Tests that decoders work with valid single-feature inputs.

## Why this matters

The decoding notebook now uses only true frame-level latent representations for classification and regression. This prevents confusing shape errors and makes notebook 06 robust to richer embedding files produced by notebook 05 and future CEBRA/dRNN analyses.
