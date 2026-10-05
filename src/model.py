"""Model definitions: EfficientNet-B0 and ResNet-18 with a single-logit binary head.

    python -m src.model      # prints parameter counts for each training phase
"""

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

MODEL_NAMES = ("efficientnet_b0", "resnet18")


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    """Load an ImageNet-pretrained backbone and replace its head with Linear(*, 1)."""
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        # original classifier = Sequential(Dropout(0.2), Linear(1280, 1000))
        in_features = model.classifier[1].in_features          # 1280
        model.classifier[1] = nn.Linear(in_features, 1)
    elif name == "resnet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features                      # 512
        model.fc = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"unknown model '{name}', choose from {MODEL_NAMES}")
    return model


def get_head(model: nn.Module, name: str) -> nn.Module:
    return model.classifier if name == "efficientnet_b0" else model.fc


def get_last_blocks(model: nn.Module, name: str, n: int = 2):
    """The last `n` feature-extraction blocks that get unfrozen in the fine-tuning phase."""
    if name == "efficientnet_b0":
        # features = [stem, stage1 ... stage7, final 1x1 conv]  (9 children)
        return list(model.features.children())[-n:]
    # ResNet: layer1 .. layer4
    return [model.layer1, model.layer2, model.layer3, model.layer4][-n:]


def get_target_layer(model: nn.Module, name: str) -> nn.Module:
    """Last convolutional block - the layer Grad-CAM hooks into."""
    if name == "efficientnet_b0":
        return model.features[-1]
    return model.layer4[-1]


# ---------------------------------------------------------------- freezing
def set_requires_grad(module: nn.Module, flag: bool):
    for p in module.parameters():
        p.requires_grad = flag


def freeze_backbone(model: nn.Module, name: str):
    """Phase 1 (warm-up): only the new classification head is trainable."""
    set_requires_grad(model, False)
    set_requires_grad(get_head(model, name), True)


def unfreeze_last_blocks(model: nn.Module, name: str, n: int = 2):
    """Phase 2 (fine-tune): head + last `n` backbone blocks are trainable."""
    freeze_backbone(model, name)
    for block in get_last_blocks(model, name, n):
        set_requires_grad(block, True)


def set_frozen_bn_eval(model: nn.Module):
    """Keep BatchNorm layers whose weights are frozen in eval mode.

    Otherwise model.train() would keep updating their running mean/var with our
    data even though the layer is meant to be frozen.
    """
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ---------------------------------------------------------------- checkpoints
def save_checkpoint(model, name, path, **extra):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_name": name, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path, device="cpu"):
    """Rebuild the right architecture from a checkpoint and load its weights."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_name"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt


if __name__ == "__main__":
    for model_name in MODEL_NAMES:
        m = build_model(model_name, pretrained=False)
        out = m(torch.randn(2, 3, 224, 224))
        total, _ = count_parameters(m)
        freeze_backbone(m, model_name)
        _, frozen_trainable = count_parameters(m)
        unfreeze_last_blocks(m, model_name)
        _, ft_trainable = count_parameters(m)
        print(f"{model_name:16s} output {tuple(out.shape)} | total {total:,} | "
              f"phase-1 trainable {frozen_trainable:,} | phase-2 trainable {ft_trainable:,}")
