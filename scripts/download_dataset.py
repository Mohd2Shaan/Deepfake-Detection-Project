"""Download the 140k Real and Fake Faces dataset and arrange it under data/.

Kaggle ships it as  real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/
This script finds that folder and copies (or moves) the splits into data/.

Usage:
    # option 1: download with kagglehub (needs a Kaggle account / API token)
    python scripts/download_dataset.py

    # option 2: you already downloaded + extracted the zip yourself
    python scripts/download_dataset.py --source "path/to/real_vs_fake"
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src import config  # noqa: E402

KAGGLE_ID = "xhlulu/140k-real-and-fake-faces"


def find_split_root(base: Path) -> Path:
    """Return the first folder under `base` that contains train/, valid/ and test/."""
    for candidate in [base, *base.rglob("*")]:
        if candidate.is_dir() and all((candidate / s).is_dir() for s in config.SPLITS):
            return candidate
    raise FileNotFoundError(f"No folder with train/valid/test found under {base}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="already-extracted dataset folder")
    parser.add_argument("--move", action="store_true", help="move instead of copy (saves disk)")
    args = parser.parse_args()

    if args.source is None:
        import kagglehub  # pip install kagglehub
        args.source = Path(kagglehub.dataset_download(KAGGLE_ID))
        print("downloaded to", args.source)

    split_root = find_split_root(args.source)
    print("found splits in", split_root)

    for split in config.SPLITS:
        dst = config.DATA_DIR / split
        if dst.exists():
            print(f"  {dst} already exists - skipping")
            continue
        print(f"  {'moving' if args.move else 'copying'} {split} ...")
        if args.move:
            shutil.move(str(split_root / split), str(dst))
        else:
            shutil.copytree(split_root / split, dst)

    print("done. run `python -m src.dataset` to verify.")


if __name__ == "__main__":
    main()
