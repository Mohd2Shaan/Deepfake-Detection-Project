"""Central configuration: paths and hyper-parameters shared by every module."""

import os
from pathlib import Path

import torch

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parent.parent
# can be overridden, e.g. on Colab: os.environ["DEEPFAKE_DATA_DIR"] = "/content/data"
DATA_DIR = Path(os.environ.get("DEEPFAKE_DATA_DIR", ROOT_DIR / "data"))
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
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
