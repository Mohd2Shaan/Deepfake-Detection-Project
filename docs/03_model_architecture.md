# 03 — Model Architecture

**Code:** [`src/model.py`](../src/model.py)

```bash
python -m src.model    # builds both models, prints parameter counts per training phase
```

---

## 1. Convolutional Neural Networks (CNNs) — quick recap

A CNN processes images with a stack of layers:

| Layer | What it does |
|-------|--------------|
| **Convolution** | slides small learnable filters (e.g. 3×3) over the image; each filter detects a pattern (edge, texture, …) and outputs a *feature map* |
| **Batch Normalization** | normalises each channel to zero-mean/unit-variance within a batch, then rescales with learnable γ, β — stabilises and speeds up training |
| **Activation** (ReLU, SiLU) | adds non-linearity so the network can learn complex functions |
| **Pooling / stride** | reduces spatial size, increases receptive field |
| **Global Average Pooling** | averages each feature map to a single number → fixed-length vector regardless of input size |
| **Fully-connected (Linear)** | maps the feature vector to the output |

Early layers learn generic features (edges, colours), deeper layers learn more abstract,
task-specific features. That is exactly why **transfer learning** works: the generic early
layers can be reused.

---

## 2. Model A — EfficientNet-B0 (primary model)

*Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019.*

### 2.1 Compound scaling
Previous work scaled networks in one dimension only (deeper *or* wider *or* higher
resolution). EfficientNet scales **depth (d), width (w) and resolution (r) together** with a
single coefficient φ:

```
d = α^φ,  w = β^φ,  r = γ^φ      subject to  α · β² · γ² ≈ 2
```

B0 is the baseline (φ = 0) found by neural architecture search; B1–B7 are scaled-up versions.
B0 has only **~5.3M parameters** (≈4.0M once the 1000-class head is replaced by our 1-output head) yet reaches ~77% ImageNet top-1 — better than ResNet-50
(25M parameters).

### 2.2 MBConv block (the building block)
*Mobile Inverted Bottleneck Convolution* (from MobileNetV2) + Squeeze-and-Excitation:

```
input ─► 1×1 conv (expand channels ×6) ─► depthwise k×k conv ─► Squeeze-Excitation ─► 1×1 conv (project down) ─► (+ input)
```

- **Depthwise convolution:** one filter per channel — far cheaper than a normal convolution.
- **Squeeze-and-Excitation (SE):** global-pools each channel, passes the vector through a
  tiny 2-layer network, and uses the output (0–1) to re-weight channels → "channel attention".
- **Skip connection** when input and output shapes match.
- **SiLU / Swish** activation: `x · sigmoid(x)`.

### 2.3 torchvision structure

```
model.features      # 9 children
  [0] stem: 3×3 conv, 32 ch, stride 2
  [1]–[7] MBConv stages (16 → 24 → 40 → 80 → 112 → 192 → 320 channels)
  [8] 1×1 conv 320 → 1280   ← last conv block (Grad-CAM target)
model.avgpool       # global average pool → 1280-d vector
model.classifier    # Dropout(0.2) → Linear(1280, 1000)
```

**Our change:** `classifier[1] = nn.Linear(1280, 1)` — one output (logit) for binary classification.
The original dropout (p = 0.2) is kept as regularisation.

---

## 3. Model B — ResNet-18 (comparison model)

*He et al., "Deep Residual Learning for Image Recognition", CVPR 2016.*

### 3.1 The residual idea
Very deep plain networks are hard to train (vanishing gradients, degradation). ResNet adds
**skip connections**:

```
output = F(x) + x
```

The block only has to learn the *residual* F(x) = desired − x. If an identity mapping is
best, it just learns F(x) ≈ 0. Gradients flow directly through the "+ x" path, so very deep
networks become trainable.

### 3.2 Structure (~11.7M parameters)

```
conv1 7×7, 64, stride 2 → maxpool
layer1: 2 × BasicBlock(64)
layer2: 2 × BasicBlock(128)
layer3: 2 × BasicBlock(256)
layer4: 2 × BasicBlock(512)   ← layer4[-1] is the Grad-CAM target
avgpool → fc: Linear(512, 1000)
```

A BasicBlock = 3×3 conv → BN → ReLU → 3×3 conv → BN → (+ skip) → ReLU.

**Our change:** `fc = nn.Linear(512, 1)`.

---

## 4. Why a single output instead of two?

For binary classification there are two equivalent options:

| Option | Head | Loss | Probability |
|--------|------|------|-------------|
| 2 outputs | `Linear(n, 2)` | CrossEntropyLoss | softmax |
| **1 output (ours)** | `Linear(n, 1)` | **BCEWithLogitsLoss** | **sigmoid** |

With one logit *z*: `P(fake) = σ(z) = 1 / (1 + e^{-z})`. The decision rule `P ≥ 0.5` is the
same as `z ≥ 0`. It is simpler, has fewer parameters, and is mathematically equivalent to
softmax over 2 classes.

---

## 5. Freezing and unfreezing

Each parameter tensor has a flag `requires_grad`. If it is `False`, no gradient is computed
and the optimizer never changes it — the layer is **frozen**.

| Function | Effect |
|----------|--------|
| `freeze_backbone` | everything frozen except the new head (phase 1) |
| `unfreeze_last_blocks(n=2)` | head + last 2 backbone blocks trainable (phase 2) |

"Last two blocks" means:
- **EfficientNet-B0:** `features[7]` (last MBConv stage, 320 ch) and `features[8]` (1×1 conv → 1280)
- **ResNet-18:** `layer3` and `layer4`

### BatchNorm in frozen layers
BatchNorm layers keep **running mean/variance** statistics which are updated in `train()` mode
*even if their weights are frozen*. `set_frozen_bn_eval()` puts the frozen BN layers back into
`eval()` mode during training so they keep their ImageNet statistics — otherwise the "frozen"
backbone would still silently change.

---

## 6. Loading pretrained weights

We use the modern torchvision API:

```python
models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
```

(`pretrained=True` from the plan is the older, now-deprecated spelling of the same thing.)

## 7. Checkpoints

`save_checkpoint` stores a dict `{"model_name", "state_dict", ...extra}`. Because the model name
is saved, `load_checkpoint(path)` can rebuild the correct architecture automatically — the
evaluation script, Grad-CAM and app don't need to be told which model a file contains.

## 8. Parameter counts (from `python -m src.model`)

| Model | Total | Phase 1 trainable | Phase 2 trainable |
|-------|-------|-------------------|-------------------|
| EfficientNet-B0 | 4,008,829 | 1,281 (0.03%) | 1,130,673 (28%) |
| ResNet-18 | 11,177,025 | 513 (0.005%) | 10,493,953 (94%) |

Note the big difference in phase 2: ResNet's `layer3` + `layer4` contain most of its weights
(94%), while EfficientNet's last two blocks hold only 28%. This is worth mentioning when
comparing the two models in the report.
