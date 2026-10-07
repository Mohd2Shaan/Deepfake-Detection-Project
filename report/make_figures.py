"""Collect result figures and draw the report-only diagrams/charts into report/figures/.

    python report/make_figures.py

Inputs: outputs/ (plots, heatmaps, histories) and docs/images/ - all produced by the notebooks.
"""

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# fixed colour per model (validated categorical palette, slots 1-3)
C_EFF, C_RES, C_EFF2 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1f2328", "#59636e", "#e3e6ea"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
})

# ---------------------------------------------------------------- copy existing figures
COPY = {
    "docs/images/confusion_matrix_efficientnet_b0.png": "confusion_efficientnet.png",
    "docs/images/confusion_matrix_resnet18.png": "confusion_resnet.png",
    "docs/images/roc_curve.png": "roc_curve.png",
    "docs/images/training_curves_efficientnet.png": "curves_efficientnet_4blocks.png",
    "docs/images/training_curves_efficientnet_2blocks.png": "curves_efficientnet_2blocks.png",
    "docs/images/training_curves_resnet.png": "curves_resnet.png",
    "docs/images/gradcam_correct_fake.jpg": "gradcam_fake.jpg",
    "docs/images/gradcam_correct_real.jpg": "gradcam_real.jpg",
    "docs/images/gradcam_misclassified.jpg": "gradcam_errors.jpg",
    "docs/images/gradcam_library_vs_manual.png": "gradcam_manual_vs_library.png",
    "docs/images/most_confident_errors_efficientnet.jpg": "errors_efficientnet.jpg",
}
for src, dst in COPY.items():
    shutil.copy(ROOT / src, FIG / dst)

# EDA plots live inside the executed EDA notebook - extract them by cell
import base64  # noqa: E402

import nbformat  # noqa: E402

EDA_CELLS = {7: ["eda_class_balance"], 11: ["eda_samples_real", "eda_samples_fake"],
             13: ["eda_file_size"], 15: ["eda_mean_face_spectrum"], 17: ["eda_batch"]}
eda_nb = nbformat.read(ROOT / "notebooks" / "01_eda.ipynb", 4)
for idx, names in EDA_CELLS.items():
    pngs = [o["data"]["image/png"] for o in eda_nb.cells[idx].get("outputs", []) if "image/png" in o.get("data", {})]
    for name, png in zip(names, pngs):
        (FIG / f"{name}.png").write_bytes(base64.b64decode(png))

# photo grids are large as PNG - store them as JPEG to keep the PDF/PPTX small
from PIL import Image  # noqa: E402

for name in ("eda_samples_real", "eda_samples_fake", "eda_batch"):
    Image.open(FIG / f"{name}.png").convert("RGB").save(FIG / f"{name}.jpg", quality=85)
    (FIG / f"{name}.png").unlink()


# ---------------------------------------------------------------- pipeline diagram
def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="white", edgecolor=color, linewidth=2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=11.5, weight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8.8, color=MUTED, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.6))


fig, ax = plt.subplots(figsize=(13, 4.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 4.2); ax.axis("off")
steps = [("Dataset", "140k faces\n70k real / 70k StyleGAN"),
         ("Preprocess", "resize 224×224\nImageNet normalise\naugment (train only)"),
         ("CNN backbone", "EfficientNet-B0 /\nResNet-18\n(ImageNet-pretrained)"),
         ("Head + sigmoid", "Linear(→1) logit\nP(fake) = σ(z)"),
         ("Decision", "P ≥ 0.5 → FAKE\nelse REAL")]
w, h, gap, y = 2.2, 1.6, 0.42, 2.3
for i, (t, s) in enumerate(steps):
    x = 0.15 + i * (w + gap)
    box(ax, x, y, w, h, t, s, C_EFF)
    if i:
        arrow(ax, x - gap + 0.02, y + h / 2, x - 0.02, y + h / 2)
gx = 0.15 + 2 * (w + gap)
box(ax, gx, 0.2, w, 1.4, "Grad-CAM", "last conv layer\n→ heatmap overlay", C_RES)
arrow(ax, gx + w / 2, y - 0.02, gx + w / 2, 1.62)
ex = 0.15 + 4 * (w + gap)
box(ax, ex, 0.2, w, 1.4, "Streamlit app", "upload → label +\nconfidence + heatmap", C_RES)
arrow(ax, ex + w / 2, y - 0.02, ex + w / 2, 1.62)
fig.savefig(FIG / "pipeline.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- two-phase schedule
fig, ax = plt.subplots(figsize=(11, 2.0))
ax.axis("off"); ax.set_xlim(0, 10.4); ax.set_ylim(0, 2)
for e in range(10):
    frozen = e < 3
    ax.add_patch(FancyBboxPatch((0.2 + e, 0.75), 0.9, 0.7, boxstyle="round,pad=0,rounding_size=0.06",
                                facecolor=MUTED if frozen else C_EFF, edgecolor="white", linewidth=2))
    ax.text(0.65 + e, 1.1, f"E{e + 1}", ha="center", va="center", color="white", weight="bold", fontsize=10)
ax.text(1.7, 0.45, "Phase 1 · warm-up\nbackbone frozen, head only · lr 1e-4", ha="center", va="top", fontsize=9.5, color=INK)
ax.text(6.7, 0.45, "Phase 2 · fine-tune\nhead + last blocks unfrozen · lr 1e-5 (run 02) / 1e-4 (run 02b)",
        ha="center", va="top", fontsize=9.5, color=INK)
ax.text(0.2, 1.75, "Epoch schedule (10 epochs, best checkpoint by validation loss)", fontsize=10.5, color=INK, weight="bold")
fig.savefig(FIG / "two_phase.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- validation accuracy of the three runs
hist = {name: json.load(open(ROOT / "outputs" / f"history_{name}.json"))
        for name in ("efficientnet", "resnet", "efficientnet_2blocks")}
runs = [("efficientnet", "EfficientNet-B0 · 4 blocks (final)", C_EFF),
        ("resnet", "ResNet-18", C_RES),
        ("efficientnet_2blocks", "EfficientNet-B0 · 2 blocks", C_EFF2)]
fig, ax = plt.subplots(figsize=(9, 4.4))
for key, label, color in runs:
    acc = [a * 100 for a in hist[key]["val_acc"]]
    ax.plot(range(1, 11), acc, color=color, lw=2, marker="o", ms=5, label=label)
    best = max(range(10), key=lambda i: acc[i])
    dy = {"efficientnet": 9, "resnet": -12, "efficientnet_2blocks": 9}[key]
    ax.annotate(f"best {acc[best] + 1e-9:.2f}%", (best + 1, acc[best]), xytext=(0, dy), textcoords="offset points",
                ha="center", va="center", fontsize=9.5, color=INK)
ax.axvspan(0.5, 3.5, color=GRID, alpha=0.6, lw=0)
ax.text(2, 101.2, "frozen", ha="center", fontsize=9, color=MUTED)
ax.text(7, 101.2, "fine-tune", ha="center", fontsize=9, color=MUTED)
ax.set_xlim(0.5, 10.6); ax.set_ylim(68, 103)
ax.set_xticks(range(1, 11)); ax.set_xlabel("epoch"); ax.set_ylabel("validation accuracy (%)")
ax.legend(loc="lower right", frameon=False, fontsize=9.5)
fig.savefig(FIG / "val_acc_runs.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- test errors per model
fig, ax = plt.subplots(figsize=(8, 2.6))
models = [("ResNet-18", 205, "98.98%", C_RES), ("EfficientNet-B0", 33, "99.84%", C_EFF)]
for i, (name, err, acc, color) in enumerate(models):
    ax.barh(i, err, height=0.5, color=color)
    ax.text(err + 4, i, f"{err} errors  ({acc} accuracy)", va="center", fontsize=10, color=INK)
ax.set_yticks(range(len(models))); ax.set_yticklabels([m[0] for m in models])
ax.set_xlim(0, 300); ax.set_xlabel("misclassified test images (out of 20,000)")
ax.grid(axis="y", visible=False)
fig.savefig(FIG / "test_errors.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

print("figures written to", FIG)
