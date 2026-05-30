from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v1_manifold_publication.pipeline import reconstruct_geometry_from_embeddings

if __name__ == "__main__":
    df = reconstruct_geometry_from_embeddings()
    print(df.head().to_string(index=False))
