# Publication Multi-session Upgrade v3

This patch upgrades notebooks 13–17 so they run over **all sessions that have the required assets** instead of silently analyzing only one session.

It does **not** fabricate results. It adds the multi-session machinery needed to obtain top-journal-grade evidence after you preprocess and embed additional sessions.

## What this ZIP contains

```text
notebooks/
  12c_multisession_asset_preparation_and_driver.ipynb
  13_multisession_encoding_decoding_and_nulls.ipynb
  14_layer_geometry_statistics.ipynb
  15_brain_model_alignment_rsa_cka.ipynb
  16_cross_session_manifold_alignment.ipynb
  17_publication_figures_and_manuscript_assets.ipynb

src/v1_manifold_publication/
  multisession_core.py

scripts/
  run_publication_upgrade_multisession.py

configs/
  publication_multisession_upgrade.yaml
```

## Installation

Copy the contents of this ZIP into your existing project root:

```text
C:\Users\Peter\Documents\projects\NeuroAI\latent-manifold-v1-natural-movies
```

Keep your old notebooks if you want; these notebooks are versioned upgrade copies. The only required source addition is:

```text
src/v1_manifold_publication/multisession_core.py
```

## Recommended run order

1. Run `12c_multisession_asset_preparation_and_driver.ipynb`.
   - This reports which sessions have features, tensors, and embeddings.
   - It writes a missing-assets to-do table.

2. Preprocess and embed additional sessions until at least 3 sessions have:
   - `data/interim/session_<id>_natural_movie_one_tensor.h5`
   - `data/processed/session_<id>_embeddings.npz`
   - `data/processed/session_<id>_publication_enhanced_frame_features.csv`

3. Rerun:
   - `13_multisession_encoding_decoding_and_nulls.ipynb`
   - `14_layer_geometry_statistics.ipynb`
   - `15_brain_model_alignment_rsa_cka.ipynb`
   - `16_cross_session_manifold_alignment.ipynb`
   - `17_publication_figures_and_manuscript_assets.ipynb`

## Environment switches

Use these in a notebook cell or terminal before running the relevant notebook.

```python
import os
os.environ["PUBLICATION_RUN_LABEL"] = "publication_upgrade_v3_multisession"
```

For null controls:

```python
import os
os.environ["RUN_EXPENSIVE_PUBLICATION_NULLS"] = "1"
os.environ["PUBLICATION_NULL_PERMUTATIONS"] = "100"   # test run
# Use 1000+ for final manuscript tables.
```

## What will make this top-journal stronger

The final claim-gate notebook requires:

- At least 3 fully processed neural sessions for multi-session replication.
- More than one layer/Cre-line group before any layer-specific biological claim.
- Empirical null controls and noise/reliability summaries.
- Cross-session manifold alignment after 2+ embedding sessions, preferably 3+.
- DNN/ViT activation files and pretrained-vs-untrained controls if claiming deep-model alignment.

## Honest limitation

This ZIP upgrades the analysis code and publication gates. It cannot guarantee a top-journal result. The scientific strength will come from running the upgraded workflow on enough sessions and reporting the outcomes honestly.
