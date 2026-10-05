"""Two-phase transfer-learning training loop.

Phase 1 (warm-up)  : backbone frozen, only the new head trains   (lr = 1e-4)
Phase 2 (fine-tune): last 2 backbone blocks + head train         (lr = 1e-5)

The best checkpoint (lowest validation loss) is saved as checkpoints/<short>_best.pth,
and the latest epoch as checkpoints/<short>_last.pth so a disconnected Colab run can resume.

Examples:
    python -m src.train --model efficientnet_b0
    python -m src.train --model resnet18 --checkpoint-dir /content/drive/MyDrive/deepfake/checkpoints
    python -m src.train --model resnet18 --resume
    python -m src.train --model efficientnet_b0 --max-per-class 500 --epochs-frozen 1 --epochs-finetune 1   # quick test
"""

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import (build_model, count_parameters, freeze_backbone, save_checkpoint,
                       set_frozen_bn_eval, unfreeze_last_blocks)

SHORT_NAMES = {"efficientnet_b0": "efficientnet", "resnet18": "resnet"}


def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    set_frozen_bn_eval(model)
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        correct += ((logits > 0).float() == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="valid", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images).squeeze(1)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += ((logits > 0).float() == labels).sum().item()
        seen += len(labels)
    return total_loss / seen, correct / seen


def setup_phase(model, model_name, phase, args):
    """Freeze/unfreeze layers and build a fresh optimizer + scheduler for the phase."""
    if phase == "frozen":
        freeze_backbone(model, model_name)
        lr = args.lr_frozen
    else:
        unfreeze_last_blocks(model, model_name, args.unfreeze_blocks)
        lr = args.lr_finetune
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    _, trainable = count_parameters(model)
    print(f"\n=== phase: {phase} | lr {lr} | trainable params {trainable:,} ===")
    return optimizer, scheduler


def plot_history(history, out_path, title):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], "o-", label="train")
    ax1.plot(epochs, history["val_loss"], "o-", label="valid")
    ax1.set_title("loss")
    ax2.plot(epochs, history["train_acc"], "o-", label="train")
    ax2.plot(epochs, history["val_acc"], "o-", label="valid")
    ax2.set_title("accuracy")
    for ax in (ax1, ax2):
        ax.axvline(history["frozen_epochs"] + 0.5, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def train(args):
    set_seed()
    device = config.get_device()
    print("device:", device)

    loaders = {
        split: get_dataloader(split, args.data_dir, args.batch_size, args.num_workers, args.max_per_class)
        for split in ("train", "valid")
    }
    print(f"train {len(loaders['train'].dataset)} | valid {len(loaders['valid'].dataset)} images")

    model = build_model(args.model, pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    short = SHORT_NAMES[args.model]
    best_path = args.checkpoint_dir / f"{short}_best.pth"
    last_path = args.checkpoint_dir / f"{short}_last.pth"
    total_epochs = args.epochs_frozen + args.epochs_finetune

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [],
               "frozen_epochs": args.epochs_frozen}
    best_val_loss = float("inf")
    start_epoch = 0
    resume_ckpt = None

    if args.resume and last_path.exists():
        resume_ckpt = torch.load(last_path, map_location=device)
        model.load_state_dict(resume_ckpt["state_dict"])
        history = resume_ckpt["history"]
        best_val_loss = resume_ckpt["best_val_loss"]
        start_epoch = resume_ckpt["epoch"] + 1
        print(f"resumed from {last_path} at epoch {start_epoch + 1}")

    phase, optimizer, scheduler = None, None, None
    for epoch in range(start_epoch, total_epochs):
        epoch_phase = "frozen" if epoch < args.epochs_frozen else "finetune"
        if epoch_phase != phase:
            phase = epoch_phase
            optimizer, scheduler = setup_phase(model, args.model, phase, args)
            if resume_ckpt is not None and resume_ckpt["phase"] == phase:
                optimizer.load_state_dict(resume_ckpt["optimizer"])
                scheduler.load_state_dict(resume_ckpt["scheduler"])
            resume_ckpt = None

        start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, loaders["valid"], criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        for key, value in zip(("train_loss", "train_acc", "val_loss", "val_acc", "lr"),
                              (train_loss, train_acc, val_loss, val_acc, lr)):
            history[key].append(value)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            save_checkpoint(model, args.model, best_path, epoch=epoch, val_loss=val_loss, val_acc=val_acc)

        save_checkpoint(model, args.model, last_path, epoch=epoch, phase=phase, history=history,
                        best_val_loss=best_val_loss, optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict())

        print(f"epoch {epoch + 1:2d}/{total_epochs} [{phase:8s}] "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:.1e} | "
              f"{time.time() - start:.0f}s {'*best*' if improved else ''}")

    config.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.OUTPUT_DIR / f"history_{short}.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, config.PLOTS_DIR / f"training_curves_{short}.png", args.model)
    print(f"\nbest val loss {best_val_loss:.4f} -> {best_path}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train a deepfake detector")
    p.add_argument("--model", choices=list(SHORT_NAMES), default="efficientnet_b0")
    p.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    p.add_argument("--checkpoint-dir", type=Path, default=config.CHECKPOINT_DIR)
    p.add_argument("--epochs-frozen", type=int, default=config.FROZEN_EPOCHS)
    p.add_argument("--epochs-finetune", type=int, default=config.FINETUNE_EPOCHS)
    p.add_argument("--lr-frozen", type=float, default=config.LR_FROZEN)
    p.add_argument("--lr-finetune", type=float, default=config.LR_FINETUNE)
    p.add_argument("--unfreeze-blocks", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--max-per-class", type=int, default=None, help="subset size per class (quick runs)")
    p.add_argument("--resume", action="store_true", help="continue from <short>_last.pth")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
