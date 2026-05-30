from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v1_manifold.manifold import fit_cebra_embedding, save_embedding_npz, save_model  # noqa: E402
from v1_manifold.preprocessing import load_trial_tensor_h5, repeat_average_population_matrix  # noqa: E402
from v1_manifold.features import standardize_matrix  # noqa: E402
from v1_manifold_publication.cebra_variants import (  # noqa: E402
    build_cebra_label_matrix,
    cebra_variant_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled CEBRA variants.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--variant", default="all")
    parser.add_argument("--run-label", default="publication_upgrade_v2")
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--write-output", action="store_true")
    args = parser.parse_args()

    import pandas as pd
    import numpy as np

    tensor_path = PROJECT_ROOT / "data" / "interim" / f"session_{args.session_id}_natural_movie_one_tensor.h5"
    feature_candidates = sorted((PROJECT_ROOT / "data" / "processed").glob(f"session_{args.session_id}*frame_features.csv"))
    if not tensor_path.exists():
        raise FileNotFoundError(tensor_path)
    if not feature_candidates:
        raise FileNotFoundError(f"No feature table found for session {args.session_id}")

    tensor, _ = load_trial_tensor_h5(tensor_path)
    R = repeat_average_population_matrix(tensor)
    Xz, _ = standardize_matrix(R)
    features = pd.read_csv(feature_candidates[0])

    specs = cebra_variant_specs()
    if args.variant != "all":
        specs = [s for s in specs if s["variant"] == args.variant]
    if not specs:
        raise ValueError(f"No CEBRA variants matched {args.variant!r}")

    out_processed = PROJECT_ROOT / "data" / "processed" / args.run_label
    out_models = PROJECT_ROOT / "models" / args.run_label
    for spec in specs:
        labels, used_cols = build_cebra_label_matrix(features, list(spec["label_columns"]))
        print(f"Variant {spec['variant']} uses labels: {used_cols}")
        if not args.write_output:
            print("Dry run; would train and save into", out_processed, "and", out_models)
            continue
        embedding, model = fit_cebra_embedding(
            Xz,
            labels=labels,
            output_dimension=3,
            max_iterations=args.max_iterations,
            device=args.device,
        )
        out_processed.mkdir(parents=True, exist_ok=True)
        out_models.mkdir(parents=True, exist_ok=True)
        emb_path = out_processed / f"session_{args.session_id}_{spec['variant']}_embedding.npz"
        model_path = out_models / f"session_{args.session_id}_{spec['variant']}_cebra.joblib"
        save_embedding_npz(emb_path, **{str(spec["variant"]): embedding, "labels": labels, "frame": np.arange(len(embedding))})
        save_model(model, model_path)
        print("Saved", emb_path)
        print("Saved", model_path)


if __name__ == "__main__":
    main()
