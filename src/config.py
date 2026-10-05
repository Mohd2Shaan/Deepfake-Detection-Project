"""Central configuration: paths and hyper-parameters shared by every module."""

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------- kaggle helpers
KAGGLE_INPUT = Path("/kaggle/input")


def find_kaggle_dataset(base=KAGGLE_INPUT, max_depth=8):
    """Find the attached dataset folder (the one holding train/ valid/ test/) at any depth.

    Kaggle's mount path has changed over time (/kaggle/input/<slug>/... vs
    /kaggle/input/datasets/<owner>/<slug>/...), so we search instead of hard-coding it.
    """
    base = str(base)
    for root, dirs, _ in os.walk(base):
        if {"train", "valid", "test"} <= set(dirs):
            return Path(root)
        if root[len(base):].count(os.sep) >= max_depth:
            dirs[:] = []
    return None


def import_kaggle_checkpoints(base=KAGGLE_INPUT):
    """Copy *_best.pth from an attached notebook output (notebook 02) into checkpoints/."""
    import shutil

    found = []
    for root, dirs, files in os.walk(str(base)):
        if {"train", "valid"} <= set(dirs):        # don't walk into the image dataset
            dirs[:] = []
            continue
        for f in files:
            if f.endswith("_best.pth"):
                dst = CHECKPOINT_DIR / f
                if not dst.exists():
                    CHECKPOINT_DIR.mkdir(exist_ok=True)
                    shutil.copy(os.path.join(root, f), dst)
                found.append(dst)
    return found


# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

# priority: DEEPFAKE_DATA_DIR env var  ->  dataset attached on Kaggle  ->  ./data
if os.environ.get("DEEPFAKE_DATA_DIR"):
    DATA_DIR = Path(os.environ["DEEPFAKE_DATA_DIR"])
elif KAGGLE_INPUT.exists() and find_kaggle_dataset() is not None:
    DATA_DIR = find_kaggle_dataset()
else:
    DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
MISCLASSIFIED_DIR = OUTPUT_DIR / "misclassified"

SPLITS = ("train", "valid", "test")

# ---------------------------------------------------------------- labels
# label 0 = real, label 1 = fake  ->  sigmoid(output) = P(fake)
CLASS_NAMES = ["real", "fake"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------- preprocessing
IMG_SIZE = 224                       # EfficientNet-B0 / ResNet-18 input size
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- training
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42

FROZEN_EPOCHS = 3                    # phase 1: train only the new head
FINETUNE_EPOCHS = 7                  # phase 2: unfreeze last blocks
LR_FROZEN = 1e-4
LR_FINETUNE = 1e-5


def get_device() -> torch.device:
    """Use the GPU when available (Colab), otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
