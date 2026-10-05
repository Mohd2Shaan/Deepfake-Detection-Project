"""Helper functions for Exploratory Data Analysis (used by notebooks/01_eda.ipynb).

Run directly to produce all EDA plots into outputs/plots/:

    python -m src.eda --max-per-class 2000
"""

import argparse
import hashlib
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from src import config
from src.dataset import IMG_EXTENSIONS


def list_images(data_dir=config.DATA_DIR, max_per_class=None, seed=config.SEED):
    """Return a DataFrame with one row per image: path, split, class, label."""
    rng = random.Random(seed)
    rows = []
    for split in config.SPLITS:
        for class_name, label in config.CLASS_TO_IDX.items():
            folder = Path(data_dir) / split / class_name
            files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTENSIONS)
            if max_per_class is not None and len(files) > max_per_class:
                files = rng.sample(files, max_per_class)
            rows += [{"path": str(p), "split": split, "class": class_name, "label": label} for p in files]
    return pd.DataFrame(rows)


def scan_images(df):
    """Open every image and record width, height, mode, file size, md5 hash and corruption."""
    widths, heights, modes, sizes, hashes, corrupted = [], [], [], [], [], []
    for path in tqdm(df["path"], desc="scanning"):
        raw = Path(path).read_bytes()
        sizes.append(len(raw))
        hashes.append(hashlib.md5(raw).hexdigest())
        try:
            with Image.open(path) as img:
                img.verify()                      # checks file integrity
            with Image.open(path) as img:         # verify() invalidates the object, so reopen
                widths.append(img.width)
                heights.append(img.height)
                modes.append(img.mode)
            corrupted.append(False)
        except Exception:
            widths.append(None)
            heights.append(None)
            modes.append(None)
            corrupted.append(True)

    df = df.copy()
    df["width"], df["height"], df["mode"] = widths, heights, modes
    df["file_kb"] = np.array(sizes) / 1024
    df["md5"], df["corrupted"] = hashes, corrupted
    return df


def find_duplicates(df):
    """Rows whose file content (md5) appears more than once - within or across splits."""
    dup = df[df.duplicated("md5", keep=False)].sort_values("md5")
    return dup[["path", "split", "class", "md5"]]


# ---------------------------------------------------------------- plots
def plot_class_balance(df, out_path=None):
    counts = df.groupby(["split", "class"]).size().unstack()[config.CLASS_NAMES]
    counts = counts.loc[list(config.SPLITS)]
    ax = counts.plot(kind="bar", figsize=(7, 4), rot=0)
    ax.set_ylabel("number of images")
    ax.set_title("Class balance per split")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=100)
    return counts


def plot_sample_grid(df, class_name, n=16, out_path=None, seed=config.SEED):
    subset = df[df["class"] == class_name].sample(n, random_state=seed)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    for ax, path in zip(axes.flat, subset["path"]):
        ax.imshow(Image.open(path).convert("RGB"))
        ax.axis("off")
    fig.suptitle(f"Random {class_name} samples")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=100)
    return fig


def plot_file_size_distribution(df, out_path=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    for class_name in config.CLASS_NAMES:
        ax.hist(df[df["class"] == class_name]["file_kb"], bins=50, alpha=0.6, label=class_name)
    ax.set_xlabel("file size (KB)")
    ax.set_ylabel("count")
    ax.set_title("JPEG file size by class")
    ax.legend()
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=100)
    return fig


def mean_image(paths, size=config.IMG_SIZE):
    """Pixel-wise average of a list of images (shows the 'average face' of a class)."""
    acc = np.zeros((size, size, 3), dtype=np.float64)
    for p in paths:
        acc += np.asarray(Image.open(p).convert("RGB").resize((size, size)), dtype=np.float64)
    return (acc / len(paths)).astype(np.uint8)


def mean_spectrum(paths, size=config.IMG_SIZE):
    """Average log-magnitude 2D Fourier spectrum of the grayscale images.

    GAN up-sampling layers can leave periodic high-frequency patterns that show up
    as bright dots / grids in the spectrum even when invisible in the image.
    """
    acc = np.zeros((size, size), dtype=np.float64)
    for p in paths:
        gray = np.asarray(Image.open(p).convert("L").resize((size, size)), dtype=np.float64)
        f = np.fft.fftshift(np.fft.fft2(gray))
        acc += np.log1p(np.abs(f))
    return acc / len(paths)


def plot_mean_faces_and_spectra(df, n=500, out_path=None, seed=config.SEED):
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for col, class_name in enumerate(config.CLASS_NAMES):
        subset = df[df["class"] == class_name]
        paths = subset.sample(min(n, len(subset)), random_state=seed)["path"].tolist()
        axes[0, col].imshow(mean_image(paths))
        axes[0, col].set_title(f"mean {class_name} face")
        axes[1, col].imshow(mean_spectrum(paths), cmap="inferno")
        axes[1, col].set_title(f"mean {class_name} spectrum")
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=100)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="limit images per class per split (faster)")
    args = parser.parse_args()

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = scan_images(list_images(max_per_class=args.max_per_class))

    print("\n== counts ==")
    print(plot_class_balance(df, config.PLOTS_DIR / "eda_class_balance.png"))
    print("\n== image sizes ==")
    print(df.groupby(["width", "height"]).size())
    print("\n== colour modes ==")
    print(df["mode"].value_counts())
    print("\ncorrupted files:", int(df["corrupted"].sum()))
    print("duplicate files:", len(find_duplicates(df)))

    for class_name in config.CLASS_NAMES:
        plot_sample_grid(df, class_name, out_path=config.PLOTS_DIR / f"eda_samples_{class_name}.png")
    plot_file_size_distribution(df, config.PLOTS_DIR / "eda_file_size.png")
    plot_mean_faces_and_spectra(df, out_path=config.PLOTS_DIR / "eda_mean_face_spectrum.png")
    print("plots saved to", config.PLOTS_DIR)


if __name__ == "__main__":
    main()
