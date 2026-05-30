from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from v1_manifold_publication.pipeline import validate_environment

if __name__ == "__main__":
    df = validate_environment(Path.cwd())
    print(df.to_string(index=False))
    if not df["exists"].all():
        raise SystemExit("Some expected project paths or dependencies are missing.")
