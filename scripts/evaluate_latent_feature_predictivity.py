from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from v1_manifold.config import load_config, get_paths, set_global_seed
from v1_manifold.evaluation import (
    make_contiguous_frame_blocks,
    frame_time_features,
    evaluate_feature_regression,
    latent_gain_over_temporal_baseline,
)
from v1_manifold.features import assert_real_stimulus_features
from v1_manifold.utils import save_table


CONTINUOUS_TARGETS = [
    "rms_contrast",
    "spatial_frequency_centroid",
    "orientation_selectivity",
    "luminance_std",
    "total_spectral_power",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["random_seed"])
    paths = get_paths(cfg)

    all_scores = []

    for emb_path in sorted(paths.processed_dir.glob("session_*_embeddings.npz")):
        session_id = emb_path.name.split("_")[1]
        feature_candidates = sorted(paths.processed_dir.glob(f"session_{session_id}*_real_frame_features.csv"))
        if not feature_candidates:
            feature_candidates = sorted(paths.processed_dir.glob(f"session_{session_id}*_frame_features.csv"))
        if not feature_candidates:
            continue

        features = pd.read_csv(feature_candidates[0])
        assert_real_stimulus_features(features)

        embeddings = np.load(emb_path)
        targets = [col for col in CONTINUOUS_TARGETS if col in features.columns]
        groups = make_contiguous_frame_blocks(
            len(features),
            n_blocks=int(cfg.get("evaluation", {}).get("n_movie_blocks", 5)),
        )

        representations = {}
        if "pca" in embeddings:
            pca = embeddings["pca"]
            representations["pca_3d"] = pca[:, : min(3, pca.shape[1])]
            representations["pca_20d"] = pca[:, : min(20, pca.shape[1])]
        if "umap" in embeddings:
            representations["umap_3d"] = embeddings["umap"]
        if "isomap" in embeddings:
            representations["isomap_3d"] = embeddings["isomap"]
        if "cebra" in embeddings:
            representations["cebra_3d"] = embeddings["cebra"]

        representations.update({
            "frame_index_linear_baseline": features["movie_frame"].to_numpy().reshape(-1, 1),
            "frame_index_fourier_baseline": frame_time_features(
                features["movie_frame"].to_numpy(),
                n_harmonics=int(cfg.get("evaluation", {}).get("temporal_fourier_harmonics", 5)),
            ),
            "population_l2_norm_baseline": features["population_l2_norm"].to_numpy().reshape(-1, 1),
        })

        scores = evaluate_feature_regression(
            representations,
            features,
            targets,
            groups=groups,
            alpha=float(cfg.get("evaluation", {}).get("latent_regression_alpha", 1.0)),
        )
        scores.insert(0, "session_id", session_id)
        all_scores.append(scores)

    if not all_scores:
        raise FileNotFoundError(
            "No embedding/feature pairs were found. Run preprocessing and manifold analysis first."
        )

    scores = pd.concat(all_scores, ignore_index=True)
    save_table(scores, paths.tables_dir / "05_latent_vs_time_feature_prediction.csv")

    gain = (
        scores
        .groupby(["representation", "target"], as_index=False)
        .agg(
            mean_r2=("mean_r2", "mean"),
            mean_mae=("mean_mae", "mean"),
            mean_spearman_r=("mean_spearman_r", "mean"),
        )
    )
    save_table(gain, paths.tables_dir / "05_latent_vs_time_feature_prediction_session_averaged.csv")

    comparison = latent_gain_over_temporal_baseline(gain)
    save_table(comparison, paths.tables_dir / "05_latent_gain_over_temporal_baseline.csv")


if __name__ == "__main__":
    main()
