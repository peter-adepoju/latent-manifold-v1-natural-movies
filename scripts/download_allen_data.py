from __future__ import annotations

import argparse
from v1_manifold.config import load_config, get_paths, set_global_seed
from v1_manifold.data_access import get_boc, query_v1_natural_movie_experiments, save_experiment_catalog, download_selected_nwb_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--download-nwb", action="store_true", help="Actually download selected NWB files through AllenSDK.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_global_seed(cfg["project"]["random_seed"])
    paths = get_paths(cfg)
    boc = get_boc(paths.allen_manifest)
    experiments = query_v1_natural_movie_experiments(boc, cfg)
    save_experiment_catalog(experiments, paths.external_metadata)
    print(f"Saved experiment catalog: {paths.external_metadata} ({len(experiments)} rows)")

    download = args.download_nwb or bool(cfg.get("allen", {}).get("download_nwb", False))
    downloaded = download_selected_nwb_files(boc, experiments, download=download)
    if downloaded:
        print(f"Downloaded/cached {len(downloaded)} experiment NWB files through AllenSDK.")
    else:
        print("NWB download was skipped. Set --download-nwb or allen.download_nwb=true when ready.")


if __name__ == "__main__":
    main()
