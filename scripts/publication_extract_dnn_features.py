from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v1_manifold_publication.dnn_features import (  # noqa: E402
    dnn_feature_file_name,
    extract_layer_features,
    get_torchvision_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract optional DNN features from saved natural-movie frames."
    )
    parser.add_argument("--frames-npy", required=True, help="Path to [frames,H,W] or [frames,H,W,C] movie frames.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model-name", default="resnet50")
    parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3", "layer4"])
    parser.add_argument("--run-label", default="publication_upgrade_v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--untrained", action="store_true")
    parser.add_argument("--write-output", action="store_true")
    args = parser.parse_args()

    import numpy as np

    frames_path = Path(args.frames_npy)
    if not frames_path.exists():
        raise FileNotFoundError(frames_path)
    frames = np.load(frames_path)
    pretrained = not args.untrained
    print("Frames:", frames.shape)
    print("Model:", args.model_name, "pretrained:", pretrained)
    print("Layers:", args.layers)

    if not args.write_output:
        out_dir = PROJECT_ROOT / "data" / "processed" / args.run_label
        for layer in args.layers:
            print("Dry run; would save", out_dir / dnn_feature_file_name(args.session_id, args.model_name, layer, trained=pretrained))
        print("Pass --write-output to run extraction and save versioned feature arrays.")
        return

    model = get_torchvision_model(args.model_name, pretrained=pretrained)
    features = extract_layer_features(
        model,
        frames,
        args.layers,
        batch_size=args.batch_size,
        device=args.device,
    )
    out_dir = PROJECT_ROOT / "data" / "processed" / args.run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    for layer, arr in features.items():
        out = out_dir / dnn_feature_file_name(args.session_id, args.model_name, layer, trained=pretrained)
        np.save(out, arr)
        print("Saved", out, arr.shape)


if __name__ == "__main__":
    main()
