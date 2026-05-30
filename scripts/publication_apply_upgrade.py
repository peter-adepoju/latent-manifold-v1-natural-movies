"""Copy this upgrade pack into an existing project root without deleting anything."""
from pathlib import Path
import shutil
import argparse

SKIP_DIRS = {".git", "__pycache__", ".pytest_cache"}


def copy_tree(src: Path, dst: Path):
    for item in src.iterdir():
        if item.name in SKIP_DIRS:
            continue
        target = dst / item.name
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            copy_tree(item, target)
        else:
            if target.exists():
                backup = target.with_suffix(target.suffix + ".bak_publication_upgrade")
                if not backup.exists():
                    shutil.copy2(target, backup)
            shutil.copy2(item, target)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_root", help="Existing latent-manifold-v1-natural-movies project root")
    args = parser.parse_args()
    src = Path(__file__).resolve().parents[1]
    dst = Path(args.project_root).resolve()
    if not dst.exists():
        raise SystemExit(f"Project root does not exist: {dst}")
    copy_tree(src, dst)
    print(f"Copied publication upgrade into: {dst}")
    print("Existing files were preserved with .bak_publication_upgrade backups when overwritten.")
