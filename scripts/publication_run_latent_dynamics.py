from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from v1_manifold_publication.dynamics import evaluate_latent_dynamics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-horizon latent dynamics.")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data" / "processed"))
    parser.add_argument("--run-label", default="publication_upgrade_v2")
    parser.add_argument("--embedding", default="cebra")
    parser.add_argument("--history", type=int, default=10)
    parser.add_argument("--write-output", action="store_true")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    processed_dir = Path(args.processed_dir)
    rows = []
    for emb_path in sorted(processed_dir.glob("session_*_embeddings.npz")):
        session_id = emb_path.name.split("_")[1]
        with np.load(emb_path, allow_pickle=False) as emb:
            if args.embedding not in emb.files:
                continue
            df = evaluate_latent_dynamics(emb[args.embedding], history=args.history)
            df.insert(0, "session_id", session_id)
            df.insert(1, "embedding", args.embedding)
            rows.append(df)
    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    print(result.to_string(index=False) if not result.empty else "No latent dynamics rows produced.")
    if args.write_output:
        out = PROJECT_ROOT / "reports" / "tables" / args.run_label / f"dynamics_{args.embedding}_multihorizon.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(out, index=False)
        print("Saved", out)
    else:
        print("Dry run. Pass --write-output to save into the versioned publication table directory.")


if __name__ == "__main__":
    main()
