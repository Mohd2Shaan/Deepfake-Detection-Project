"""Dataset, transforms and DataLoader helpers for the 140k Real vs Fake Faces dataset.

Expected layout (the Kaggle splits are used as-is, never reshuffled):

    data/
    ├── train/{real,fake}/*.jpg
    ├── valid/{real,fake}/*.jpg
    └── test/{real,fake}/*.jpg

Run directly for a quick sanity check:

    python -m src.dataset
"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src import config

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------- transforms
def get_transforms(train: bool) -> transforms.Compose:
    """Training transforms add augmentation; valid/test only resize + normalize."""
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization so a tensor can be displayed (values in [0, 1])."""
    mean = torch.tensor(config.IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(config.IMAGENET_STD).view(-1, 1, 1)
    return (tensor.cpu() * std + mean).clamp(0, 1)


# ---------------------------------------------------------------- dataset
class DeepfakeDataset(Dataset):
    """Reads images from <root>/<split>/{real,fake}/ and returns (image, label).

    label is a float (0.0 = real, 1.0 = fake) because BCEWithLogitsLoss expects floats.
    `max_per_class` takes a random subset per class - handy for quick CPU experiments.
    """

    def __init__(self, root, split, transform=None, max_per_class=None, seed=config.SEED):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.samples = []  # list of (path, label)

        split_dir = self.root / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"{split_dir} not found. Download the dataset first (see README / scripts/download_dataset.py)."
            )

        rng = random.Random(seed)
        for class_name, label in config.CLASS_TO_IDX.items():
            files = sorted(
                p for p in (split_dir / class_name).iterdir()
                if p.suffix.lower() in IMG_EXTENSIONS
            )
            if max_per_class is not None and len(files) > max_per_class:
                files = sorted(rng.sample(files, max_per_class))
            self.samples.extend((p, label) for p in files)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

    def class_counts(self):
        counts = {name: 0 for name in config.CLASS_NAMES}
        for _, label in self.samples:
            counts[config.CLASS_NAMES[label]] += 1
        return counts


# ---------------------------------------------------------------- loaders
def get_dataloader(split, data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE,
                   num_workers=config.NUM_WORKERS, max_per_class=None):
    """DataLoader for one split. Only the training split is shuffled and augmented."""
    is_train = split == "train"
    dataset = DeepfakeDataset(data_dir, split, get_transforms(train=is_train), max_per_class)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE,
                    num_workers=config.NUM_WORKERS, max_per_class=None):
    """Returns a dict {"train": ..., "valid": ..., "test": ...}."""
    return {
        split: get_dataloader(split, data_dir, batch_size, num_workers, max_per_class)
        for split in config.SPLITS
    }


# ---------------------------------------------------------------- sanity check
def save_batch_grid(loader, out_path, n=16):
    """Save a grid of one transformed batch with labels - confirms images and labels match."""
    import matplotlib.pyplot as plt

    images, labels = next(iter(loader))
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax in axes.flat:
        ax.axis("off")
    for i in range(n):
        ax = axes.flat[i]
        ax.imshow(denormalize(images[i]).permute(1, 2, 0).numpy())
        ax.set_title(config.CLASS_NAMES[int(labels[i])])
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    loaders = get_dataloaders(num_workers=0)
    for split, loader in loaders.items():
        ds = loader.dataset
        print(f"{split:>5}: {len(ds):>6} images  {ds.class_counts()}")

    images, labels = next(iter(loaders["train"]))
    print("batch images:", tuple(images.shape), "labels:", tuple(labels.shape))

    out = config.PLOTS_DIR / "sample_train_batch.png"
    save_batch_grid(loaders["train"], out)
    print("saved", out)
