from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from v1_manifold.config import load_config, get_paths, set_global_seed
from v1_manifold.models import evaluate_decoders, evaluate_regressors, save_fitted_models, valid_embedding_names, describe_embedding_file
from v1_manifold.preprocessing import load_trial_tensor_h5
from v1_manifold.features import (
    assert_real_stimulus_features,
    make_single_trial_design_matrix,
    add_coarse_orientation_target,
    add_confident_orientation_target,
    summarize_class_balance,
    safe_classification_target,
    balanced_confident_orientation_subset,
)
from v1_manifold.utils import save_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["random_seed"])
    paths = get_paths(cfg)

    class_rows = []
    reg_rows = []
    balance_rows = []
    tensors_by_session = {p.name.split("_")[1]: p for p in sorted(paths.interim_dir.glob("session_*_tensor.h5"))}

    for emb_path in sorted(paths.processed_dir.glob("session_*_embeddings.npz")):
        session_id = emb_path.name.split("_")[1]
        label_candidates = sorted(paths.processed_dir.glob(f"session_{session_id}*_frame_features.csv"))
        if not label_candidates:
            continue
        labels = pd.read_csv(label_candidates[0])
        assert_real_stimulus_features(labels)
        if "dominant_orientation_coarse" not in labels.columns:
            labels = add_coarse_orientation_target(
                labels,
                top_k=cfg["features"].get("coarse_orientation_top_k", 2),
            )
        if "dominant_orientation_confident" not in labels.columns:
            labels = add_confident_orientation_target(
                labels,
                confidence_quantile=cfg["features"].get("orientation_confidence_quantile", 0.50),
                ambiguous_label=cfg["features"].get("orientation_ambiguous_label", "ambiguous"),
            )

        data = np.load(emb_path)
        embedding_names = valid_embedding_names(data, n_samples=len(labels))
        if not embedding_names:
            print(f"Skipping {emb_path}: no valid 2D frame-level embeddings were found.")
            continue
        embedding_summary = describe_embedding_file(data, n_samples=len(labels))
        save_table(
            embedding_summary,
            paths.tables_dir / f"06_embedding_file_summary_session_{session_id}.csv",
        )

        classification_targets = {
            "dominant_orientation_coarse_secondary": labels["dominant_orientation_coarse"].to_numpy(),
            "dominant_orientation_confident_exploratory": labels["dominant_orientation_confident"].to_numpy(),
            "spatial_frequency_bin_secondary": labels["spatial_frequency_bin"].to_numpy(),
            "dominant_orientation_bin_fine_exploratory": labels["dominant_orientation_bin"].to_numpy(),
        }
        classification_targets = {
            name: y for name, y in classification_targets.items()
            if safe_classification_target(y, min_classes=2, min_count_per_class=2)
        }

        balanced_confident = balanced_confident_orientation_subset(
            labels,
            random_state=cfg["project"]["random_seed"],
        )

        for target_name, y in classification_targets.items():
            balance = summarize_class_balance(y, target_name=target_name)
            balance.insert(0, "session_id", session_id)
            balance.insert(1, "target", target_name)
            balance_rows.append(balance)

        for target_name, y in classification_targets.items():
            for emb_name in embedding_names:
                X = data[emb_name]
                metrics, preds, fitted = evaluate_decoders(
                    X,
                    y,
                    groups=None,
                    n_splits=cfg["evaluation"]["n_splits"],
                    random_state=cfg["project"]["random_seed"],
                    include_null_baselines=cfg["evaluation"].get("include_null_baselines", True),
                )
                metrics.insert(0, "target", target_name)
                metrics.insert(0, "analysis", "frame_level_latent_exploratory")
                metrics.insert(0, "embedding", emb_name)
                metrics.insert(0, "session_id", session_id)
                class_rows.append(metrics)
                save_fitted_models(fitted, paths.models_dir, prefix=f"session_{session_id}_{emb_name}_{target_name}")

        if session_id in tensors_by_session:
            tensor, _ = load_trial_tensor_h5(tensors_by_session[session_id])
            X_trial, repeat_group, frame_idx = make_single_trial_design_matrix(tensor)
            for target_name, y_frame in classification_targets.items():
                y_trial = y_frame[frame_idx]
                metrics, preds, fitted = evaluate_decoders(
                    X_trial,
                    y_trial,
                    groups=repeat_group,
                    n_splits=min(cfg["evaluation"]["n_splits"], len(np.unique(repeat_group))),
                    random_state=cfg["project"]["random_seed"],
                    include_null_baselines=cfg["evaluation"].get("include_null_baselines", True),
                )
                metrics.insert(0, "target", target_name)
                metrics.insert(0, "analysis", "single_trial_repeat_heldout")
                metrics.insert(0, "embedding", "raw_cells")
                metrics.insert(0, "session_id", session_id)
                class_rows.append(metrics)
                save_fitted_models(fitted, paths.models_dir, prefix=f"session_{session_id}_raw_cells_{target_name}")

            if not balanced_confident.empty and safe_classification_target(
                balanced_confident["dominant_orientation_confident"],
                min_classes=2,
                min_count_per_class=2,
            ):
                target_name = "dominant_orientation_confident_balanced_binary_exploratory"
                frame_subset = balanced_confident["movie_frame"].astype(int).to_numpy()
                y_balanced = balanced_confident["dominant_orientation_confident"].to_numpy()
                balance = summarize_class_balance(y_balanced, target_name=target_name)
                balance.insert(0, "session_id", session_id)
                balance.insert(1, "target", target_name)
                balance_rows.append(balance)

                for emb_name in embedding_names:
                    metrics, preds, fitted = evaluate_decoders(
                        data[emb_name][frame_subset],
                        y_balanced,
                        groups=None,
                        n_splits=min(cfg["evaluation"]["n_splits"], pd.Series(y_balanced).value_counts().min()),
                        random_state=cfg["project"]["random_seed"],
                        include_null_baselines=cfg["evaluation"].get("include_null_baselines", True),
                    )
                    metrics.insert(0, "target", target_name)
                    metrics.insert(0, "analysis", "balanced_confident_frame_level_exploratory")
                    metrics.insert(0, "embedding", emb_name)
                    metrics.insert(0, "session_id", session_id)
                    class_rows.append(metrics)
                    save_fitted_models(fitted, paths.models_dir, prefix=f"session_{session_id}_{emb_name}_{target_name}")

                frame_to_label = dict(zip(frame_subset, y_balanced))
                trial_mask = np.isin(frame_idx, frame_subset)
                y_trial_balanced = np.array([frame_to_label[int(i)] for i in frame_idx[trial_mask]])
                metrics, preds, fitted = evaluate_decoders(
                    X_trial[trial_mask],
                    y_trial_balanced,
                    groups=repeat_group[trial_mask],
                    n_splits=min(cfg["evaluation"]["n_splits"], len(np.unique(repeat_group[trial_mask]))),
                    random_state=cfg["project"]["random_seed"],
                    include_null_baselines=cfg["evaluation"].get("include_null_baselines", True),
                )
                metrics.insert(0, "target", target_name)
                metrics.insert(0, "analysis", "balanced_confident_single_trial_repeat_heldout")
                metrics.insert(0, "embedding", "raw_cells")
                metrics.insert(0, "session_id", session_id)
                class_rows.append(metrics)
                save_fitted_models(fitted, paths.models_dir, prefix=f"session_{session_id}_raw_cells_{target_name}")

            regression_targets = {
                "rms_contrast": labels["rms_contrast"].to_numpy(),
                "spatial_frequency_centroid": labels["spatial_frequency_centroid"].to_numpy(),
                "orientation_selectivity": labels["orientation_selectivity"].to_numpy(),
                "luminance_std": labels["luminance_std"].to_numpy(),
                "total_spectral_power": labels["total_spectral_power"].to_numpy(),
            }
            for target_name, y_frame in regression_targets.items():
                y_trial = y_frame[frame_idx]
                metrics, preds, fitted = evaluate_regressors(
                    X_trial,
                    y_trial,
                    groups=repeat_group,
                    n_splits=min(cfg["evaluation"]["n_splits"], len(np.unique(repeat_group))),
                    random_state=cfg["project"]["random_seed"],
                )
                metrics.insert(0, "target", target_name)
                metrics.insert(0, "analysis", "single_trial_repeat_heldout")
                metrics.insert(0, "embedding", "raw_cells")
                metrics.insert(0, "session_id", session_id)
                reg_rows.append(metrics)
                save_fitted_models(fitted, paths.models_dir, prefix=f"session_{session_id}_raw_cells_{target_name}_regression")

    if balance_rows:
        balance = pd.concat(balance_rows, ignore_index=True)
        save_table(balance, paths.tables_dir / "06_classification_target_balance.csv")
    if class_rows:
        benchmark = pd.concat(class_rows, ignore_index=True)
        save_table(benchmark, paths.tables_dir / "06_baseline_decoding_benchmark.csv")
    else:
        print("No embeddings were available for classification evaluation.")
    if reg_rows:
        regression = pd.concat(reg_rows, ignore_index=True)
        save_table(regression, paths.tables_dir / "06_continuous_feature_decoding_benchmark.csv")
        primary = regression.sort_values("r2", ascending=False).groupby("target", as_index=False).first()
        save_table(primary, paths.tables_dir / "06_primary_continuous_decoding_summary.csv")


if __name__ == "__main__":
    main()
