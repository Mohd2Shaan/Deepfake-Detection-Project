"""Test-set evaluation, model comparison and error analysis.

    python -m src.evaluate                                   # evaluate every checkpoint found
    python -m src.evaluate --checkpoints checkpoints/efficientnet_best.pth

Outputs (in outputs/):
    plots/confusion_matrix_<name>.png, plots/roc_curve.png
    misclassified/<name>_most_confident_errors.png  (+ .csv)
    model_comparison.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from tqdm import tqdm

from src import config
from src.dataset import get_dataloader
from src.model import load_checkpoint


@torch.no_grad()
def predict(model, loader, device):
    """Return P(fake) and true labels for every image in the loader (in dataset order)."""
    model.eval()
    probs, labels = [], []
    for images, y in tqdm(loader, desc="predict", leave=False):
        probs.append(torch.sigmoid(model(images.to(device)).squeeze(1)).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy().astype(int)


def compute_metrics(y_true, probs, threshold=0.5):
    y_pred = (probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
    }


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=config.CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()
    return cm


def plot_roc_curves(results, out_path):
    """results: {model_name: (y_true, probs)} - all models on one ROC plot."""
    plt.figure(figsize=(6, 6))
    for name, (y_true, probs) in results.items():
        fpr, tpr, _ = roc_curve(y_true, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_true, probs):.4f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (test set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close()


def most_confident_errors(paths, y_true, probs, k=25):
    """Misclassified images sorted by how confident the (wrong) prediction was."""
    y_pred = (probs >= 0.5).astype(int)
    confidence = np.where(y_pred == 1, probs, 1 - probs)
    df = pd.DataFrame({"path": [str(p) for p in paths], "true": y_true, "pred": y_pred,
                       "p_fake": probs, "confidence": confidence})
    errors = df[df["true"] != df["pred"]].sort_values("confidence", ascending=False)
    return errors.head(k).reset_index(drop=True)


def plot_error_grid(errors, title, out_path, cols=5):
    if errors.empty:
        print("no misclassifications - nothing to plot")
        return
    rows = (len(errors) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, (_, row) in zip(np.atleast_1d(axes).flat, errors.iterrows()):
        ax.imshow(Image.open(row["path"]).convert("RGB"))
        ax.set_title(f"true {config.CLASS_NAMES[row['true']]}\n"
                     f"pred {config.CLASS_NAMES[row['pred']]} ({row['confidence']:.1%})", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def evaluate(checkpoint_paths, data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE,
             num_workers=config.NUM_WORKERS, max_per_class=None, top_k_errors=25):
    device = config.get_device()
    loader = get_dataloader("test", data_dir, batch_size, num_workers, max_per_class)
    paths = [p for p, _ in loader.dataset.samples]
    for d in (config.PLOTS_DIR, config.MISCLASSIFIED_DIR):
        d.mkdir(parents=True, exist_ok=True)

    rows, roc_inputs = [], {}
    for ckpt_path in checkpoint_paths:
        model, ckpt = load_checkpoint(ckpt_path, device)
        name = ckpt["model_name"]
        print(f"\n===== {name}  ({ckpt_path}) =====")

        probs, y_true = predict(model, loader, device)
        y_pred = (probs >= 0.5).astype(int)
        print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4,
                                    zero_division=0))

        metrics = compute_metrics(y_true, probs)
        rows.append({"model": name, **metrics})
        roc_inputs[name] = (y_true, probs)

        plot_confusion_matrix(y_true, y_pred, f"Confusion matrix - {name}",
                              config.PLOTS_DIR / f"confusion_matrix_{name}.png")

        errors = most_confident_errors(paths, y_true, probs, top_k_errors)
        errors.to_csv(config.MISCLASSIFIED_DIR / f"{name}_most_confident_errors.csv", index=False)
        plot_error_grid(errors, f"{name}: most confident mistakes",
                        config.MISCLASSIFIED_DIR / f"{name}_most_confident_errors.png")
        print(f"misclassified: {int((y_true != y_pred).sum())} / {len(y_true)}")

    plot_roc_curves(roc_inputs, config.PLOTS_DIR / "roc_curve.png")
    table = pd.DataFrame(rows).set_index("model").round(4)
    table.to_csv(config.OUTPUT_DIR / "model_comparison.csv")
    print("\n===== comparison =====")
    print(table.to_string())
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="*", type=Path,
                        help="checkpoint files (default: every *_best.pth in checkpoints/)")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--top-k-errors", type=int, default=25)
    args = parser.parse_args()

    ckpts = args.checkpoints or sorted(config.CHECKPOINT_DIR.glob("*_best.pth"))
    if not ckpts:
        raise SystemExit("no checkpoints found - train a model first (python -m src.train)")
    evaluate(ckpts, args.data_dir, args.batch_size, args.num_workers, args.max_per_class, args.top_k_errors)


if __name__ == "__main__":
    main()
