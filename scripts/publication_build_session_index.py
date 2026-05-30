from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v1_manifold_publication.pipeline import build_existing_session_index

if __name__ == "__main__":
    df = build_existing_session_index()
    print(df.to_string(index=False))
