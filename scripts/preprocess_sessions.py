from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from v1_manifold.config import load_config, get_paths, set_global_seed
from v1_manifold.data_access import (
    get_boc,
    get_experiment_data,
    extract_dff_and_metadata,
    get_stimulus_table,
    get_stimulus_template,
)
from v1_manifold.preprocessing import (
    rolling_quantile_baseline,
    zscore_cells,
    quality_filter_cells,
    build_trial_tensor,
    save_trial_tensor_h5,
    repeat_average_population_matrix,
)
from v1_manifold.features import (
    build_real_movie_feature_table,
    fallback_frame_labels,
    build_frame_feature_table,
    assert_real_stimulus_features,
    add_coarse_orientation_target,
    add_confident_orientation_target,
)
from v1_manifold.utils import save_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["random_seed"])
    paths = get_paths(cfg)
    if not paths.external_metadata.exists():
        raise FileNotFoundError(f"Experiment catalog not found: {paths.external_metadata}. Run `make data` first.")
    experiments = pd.read_csv(paths.external_metadata)
    if experiments.empty:
        raise RuntimeError("No experiments were found in the catalog.")

    boc = get_boc(paths.allen_manifest)
    prep = cfg["preprocessing"]
    feat_cfg = cfg["features"]

    for _, row in experiments.iterrows():
        experiment_id = int(row["id"])
        print(f"Preprocessing experiment {experiment_id}")
        data_set = get_experiment_data(boc, experiment_id)
        cell_meta, dff, _ = extract_dff_and_metadata(data_set)
        keep, qc = quality_filter_cells(
            dff,
            max_nan_fraction=prep["max_nan_fraction"],
            min_snr=prep["min_snr"],
            min_mean_dff=prep["min_mean_dff"],
        )
        save_table(qc, paths.tables_dir / f"03_qc_experiment_{experiment_id}.csv")
        print(f"Retained {int(keep.sum())} / {len(keep)} cells")

        dff_kept = dff[keep]
        cell_meta_kept = cell_meta.loc[keep].reset_index(drop=True) if len(cell_meta) == len(keep) else cell_meta
        dff_corrected = rolling_quantile_baseline(
            dff_kept,
            window=prep["rolling_window"],
            quantile=prep["baseline_quantile"],
        )
        dff_z = zscore_cells(dff_corrected, eps=prep.get("zscore_eps", 1e-6))
        stim = get_stimulus_table(data_set, prep["stimulus_name"])
        tensor, stim_aug = build_trial_tensor(
            dff_z,
            stim,
            expected_movie_frames=prep["expected_movie_frames"],
            max_repeats=prep["max_repeats"],
            aggregation=prep["response_aggregation"],
        )
        out = paths.interim_dir / f"session_{experiment_id}_{prep['stimulus_name']}_tensor.h5"
        save_trial_tensor_h5(out, tensor, stim_aug, cell_metadata=cell_meta_kept, attrs=row.to_dict())
        R = repeat_average_population_matrix(tensor)

        template = get_stimulus_template(data_set, prep["stimulus_name"])
        if template is None:
            if feat_cfg.get("require_real_stimulus_features", True):
                raise RuntimeError(
                    f"Could not load stimulus template for experiment {experiment_id}. "
                    "Real stimulus-feature decoding is required, so I am not using fallback labels."
                )
            labels = fallback_frame_labels(R.shape[0], feat_cfg["orientation_bins"], feat_cfg["spatial_frequency_bins"])
        else:
            labels = build_real_movie_feature_table(
                template,
                n_orientation_bins=feat_cfg["orientation_bins"],
                n_spatial_frequency_bins=feat_cfg["spatial_frequency_bins"],
                downsample_factor=feat_cfg.get("stimulus_downsample_factor", 4),
                max_frames=R.shape[0],
            )

        frame_features = build_frame_feature_table(R, labels)
        assert_real_stimulus_features(frame_features)
        frame_features = add_coarse_orientation_target(
            frame_features,
            top_k=feat_cfg.get("coarse_orientation_top_k", 2),
        )
        frame_features = add_confident_orientation_target(
            frame_features,
            confidence_quantile=feat_cfg.get("orientation_confidence_quantile", 0.50),
            ambiguous_label=feat_cfg.get("orientation_ambiguous_label", "ambiguous"),
        )
        save_table(frame_features, paths.processed_dir / f"session_{experiment_id}_{prep['stimulus_name']}_real_frame_features.csv")
        save_table(frame_features, paths.processed_dir / f"session_{experiment_id}_{prep['stimulus_name']}_frame_features.csv")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
