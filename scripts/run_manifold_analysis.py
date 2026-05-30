from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from v1_manifold.config import load_config, get_paths, set_global_seed, select_device
from v1_manifold.preprocessing import load_trial_tensor_h5
from v1_manifold.features import standardize_matrix, assert_real_stimulus_features
from v1_manifold.manifold import fit_pca, fit_umap, fit_isomap, fit_cebra_embedding, save_embedding_npz, save_model
from v1_manifold.geometry import summarize_geometry
from v1_manifold.utils import save_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["random_seed"])
    paths = get_paths(cfg)
    tensors = sorted(paths.interim_dir.glob("session_*_tensor.h5"))
    if not tensors:
        raise FileNotFoundError("No preprocessed tensors found. Run `make preprocess` first.")

    rows = []
    for path in tensors:
        session_id = path.name.split("_")[1]
        feature_candidates = sorted(paths.processed_dir.glob(f"session_{session_id}*_frame_features.csv"))
        if not feature_candidates:
            raise FileNotFoundError(f"Real frame features missing for session {session_id}. Run `make preprocess` first.")
        frame_df = pd.read_csv(feature_candidates[0])
        assert_real_stimulus_features(frame_df)

        _, R = load_trial_tensor_h5(path)
        Xz, scaler = standardize_matrix(R)
        save_model(scaler, paths.models_dir / f"session_{session_id}_scaler.joblib")
        pca_scores, pca, pr = fit_pca(Xz, cfg["features"]["pca_components"], cfg["project"]["random_seed"])
        save_model(pca, paths.models_dir / f"session_{session_id}_pca.joblib")
        umap_z = fit_umap(
            pca_scores,
            n_components=3,
            n_neighbors=cfg["features"]["umap_neighbors"],
            min_dist=cfg["features"]["umap_min_dist"],
            random_state=cfg["project"]["random_seed"],
        )
        isomap_z = fit_isomap(pca_scores, n_components=3, n_neighbors=cfg["features"]["isomap_neighbors"])
        arrays = {"pca": pca_scores[:, :3], "umap": umap_z, "isomap": isomap_z, "frame": np.arange(R.shape[0])}
        if cfg.get("cebra", {}).get("enabled", True):
            device = select_device(cfg["cebra"].get("device", "auto"))
            cebra_z, cebra_model = fit_cebra_embedding(
                Xz,
                labels=np.arange(R.shape[0])[:, None],
                output_dimension=cfg["cebra"]["output_dimension"],
                model_architecture=cfg["cebra"]["model_architecture"],
                conditional=cfg["cebra"]["conditional"],
                max_iterations=cfg["cebra"]["max_iterations"],
                batch_size=cfg["cebra"]["batch_size"],
                learning_rate=cfg["cebra"]["learning_rate"],
                distance=cfg["cebra"]["distance"],
                device=device,
                random_state=cfg["project"]["random_seed"],
            )
            arrays["cebra"] = cebra_z
            save_model(cebra_model, paths.models_dir / f"session_{session_id}_cebra.joblib")
        save_embedding_npz(paths.processed_dir / f"session_{session_id}_embeddings.npz", **arrays)
        for name, Z in arrays.items():
            if name == "frame":
                continue
            metrics = summarize_geometry(name, Z)
            metrics.update({"session_id": session_id, "participation_ratio": pr, "n_cells": R.shape[1], "n_frames": R.shape[0]})
            rows.append(metrics)
    save_table(pd.DataFrame(rows), paths.tables_dir / "05_geometry_summary.csv")


if __name__ == "__main__":
    main()
