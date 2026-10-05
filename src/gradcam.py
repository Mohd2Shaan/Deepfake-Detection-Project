"""Grad-CAM utilities: heatmaps showing which image regions drove a prediction.

    python -m src.gradcam --checkpoint checkpoints/efficientnet_best.pth --n 12

Outputs (outputs/heatmaps/):
    <name>_correct_real.png, <name>_correct_fake.png   - correctly classified examples
    <name>_misclassified.png                           - errors found by src.evaluate
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from src import config
from src.dataset import DeepfakeDataset, get_transforms
from src.model import get_target_layer, load_checkpoint


def to_display_rgb(pil_image):
    """Resize a PIL image to the model input size and return float RGB in [0, 1]."""
    img = pil_image.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    return np.asarray(img, dtype=np.float32) / 255.0


class GradCAMExplainer:
    """Wraps pytorch-grad-cam for our single-logit binary models."""

    def __init__(self, model, model_name, device="cpu"):
        self.model = model.eval()
        self.device = device
        self.cam = GradCAM(model=model, target_layers=[get_target_layer(model, model_name)])
        self.transform = get_transforms(train=False)

    def explain(self, pil_image, target_class=None):
        """Returns dict(prob_fake, pred, cam, overlay).

        target_class=None explains the predicted class:
          1 (fake) -> regions that push the logit up   (evidence for "fake")
          0 (real) -> regions that push the logit down (evidence for "real")
        """
        x = self.transform(pil_image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob_fake = torch.sigmoid(self.model(x)).item()
        pred = int(prob_fake >= 0.5)
        target = pred if target_class is None else target_class

        grayscale_cam = self.cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(target)])[0]
        overlay = show_cam_on_image(to_display_rgb(pil_image), grayscale_cam, use_rgb=True)
        return {"prob_fake": prob_fake, "pred": pred, "cam": grayscale_cam, "overlay": overlay}


def manual_gradcam(model, target_layer, x, sign=1.0):
    """Grad-CAM written out by hand (for understanding; the library does the same thing).

    1. forward pass, keep the target layer's feature maps A^k
    2. backprop the class score y to get dy/dA^k
    3. weights alpha_k = global-average-pool of the gradients over H, W
    4. cam = ReLU( sum_k alpha_k * A^k ), upsample to input size, scale to [0, 1]
    """
    store = {}
    fwd = target_layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    bwd = target_layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0]))
    try:
        model.zero_grad()
        score = sign * model(x).squeeze()
        score.backward()
    finally:
        fwd.remove()
        bwd.remove()

    weights = store["grad"].mean(dim=(2, 3), keepdim=True)               # [1, K, 1, 1]
    cam = F.relu((weights * store["act"]).sum(dim=1, keepdim=True))      # [1, 1, h, w]
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy()


def save_heatmap_grid(explainer, paths, true_labels, title, out_path, cols=4):
    """Each cell shows the original image above its Grad-CAM overlay."""
    n = len(paths)
    if n == 0:
        print(f"no images for '{title}' - skipped")
        return False
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows * 2, cols, figsize=(cols * 2.6, rows * 5.4))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for i, (path, label) in enumerate(zip(paths, true_labels)):
        r, c = (i // cols) * 2, i % cols
        image = Image.open(path)
        result = explainer.explain(image)
        axes[r, c].imshow(to_display_rgb(image))
        axes[r, c].set_title(f"true: {config.CLASS_NAMES[label]}", fontsize=8)
        axes[r + 1, c].imshow(result["overlay"])
        axes[r + 1, c].set_title(f"pred: {config.CLASS_NAMES[result['pred']]} "
                                 f"(P(fake)={result['prob_fake']:.2f})", fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print("saved", out_path)
    return True


def pick_correct_examples(explainer, data_dir, n, class_name, max_scan=500):
    """First `n` test images of a class that the model classifies correctly."""
    dataset = DeepfakeDataset(data_dir, "test", max_per_class=max_scan)
    label = config.CLASS_TO_IDX[class_name]
    picked = []
    with torch.no_grad():
        for path, y in dataset.samples:
            if y != label:
                continue
            x = explainer.transform(Image.open(path).convert("RGB")).unsqueeze(0).to(explainer.device)
            if int(torch.sigmoid(explainer.model(x)).item() >= 0.5) == y:
                picked.append(path)
            if len(picked) == n:
                break
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=config.CHECKPOINT_DIR / "efficientnet_best.pth")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--n", type=int, default=12, help="images per grid")
    args = parser.parse_args()

    device = config.get_device()
    model, ckpt = load_checkpoint(args.checkpoint, device)
    name = ckpt["model_name"]
    explainer = GradCAMExplainer(model, name, device)

    for class_name in config.CLASS_NAMES:
        paths = pick_correct_examples(explainer, args.data_dir, args.n, class_name)
        out = config.HEATMAP_DIR / f"{name}_correct_{class_name}.png"
        save_heatmap_grid(explainer, paths, [config.CLASS_TO_IDX[class_name]] * len(paths),
                          f"{name}: correctly classified {class_name} faces", out)

    errors_csv = config.MISCLASSIFIED_DIR / f"{name}_most_confident_errors.csv"
    if errors_csv.exists():
        errors = pd.read_csv(errors_csv).head(args.n)
        out = config.HEATMAP_DIR / f"{name}_misclassified.png"
        save_heatmap_grid(explainer, errors["path"].tolist(), errors["true"].tolist(),
                          f"{name}: Grad-CAM on misclassified images", out)
    else:
        print(f"{errors_csv} not found - run `python -m src.evaluate` first for error heatmaps")


if __name__ == "__main__":
    main()
