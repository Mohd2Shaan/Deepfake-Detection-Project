# 06 — Explainability with Grad-CAM

**Code:** [`src/gradcam.py`](../src/gradcam.py) · **Notebook:** [`notebooks/04_gradcam.ipynb`](../notebooks/04_gradcam.ipynb)

```bash
python -m src.gradcam --checkpoint checkpoints/efficientnet_best.pth --n 12
python -m src.gradcam --checkpoint checkpoints/resnet_best.pth --n 8
```

---

## 1. Why explainability?

A CNN is a "black box": it outputs 0.98 but doesn't say *why*. For a deepfake detector we want
to know whether it uses meaningful evidence (artifacts on the face, hair, background) or a
**shortcut** (a watermark, image border, compression). Visual explanations help to:

- trust (or distrust) the model,
- debug errors,
- explain results in the report and viva.

## 2. Grad-CAM — Gradient-weighted Class Activation Mapping

*Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
Localization", ICCV 2017.*

### Intuition
The last convolutional layer has *K* feature maps (EfficientNet-B0: K = 1280 maps of size 7×7).
Each map responds to some pattern and keeps **spatial information** (where in the image).
Grad-CAM asks: *how important is each feature map for the output score?* and then adds the maps
up weighted by that importance.

### Algorithm

Let *A^k* be the k-th feature map of the target layer, and *y* the score we want to explain.

1. **Forward pass** — record the activations *A^k*.
2. **Backward pass** — compute gradients ∂y/∂A^k.
3. **Importance weights** — global-average-pool the gradients:

   ```
   α_k = (1/Z) Σ_i Σ_j  ∂y / ∂A^k_ij
   ```
4. **Weighted combination + ReLU**:

   ```
   L_GradCAM = ReLU( Σ_k α_k · A^k )
   ```
   ReLU keeps only regions with a **positive** influence on the score.
5. **Upsample** the 7×7 map to 224×224, normalise to [0, 1], colour it (jet colormap: blue = low,
   red = high) and overlay on the image.

`manual_gradcam()` in `src/gradcam.py` implements exactly these steps with PyTorch hooks;
notebook 04 checks that it agrees with the `pytorch-grad-cam` library.

### Which score do we explain? (binary single-logit model)
Our model has one logit *z* (high = fake).
- To explain **"fake"**, use *y = z* → regions that increase the fake score.
- To explain **"real"**, use *y = −z* → regions that push towards real.

`BinaryClassifierOutputTarget(1)` / `(0)` from the library does exactly this. By default we
explain the **predicted** class.

## 3. Choosing the target layer

| Model | Target layer | Output size |
|-------|--------------|-------------|
| EfficientNet-B0 | `model.features[-1]` (1×1 conv → 1280 ch) | 1280 × 7 × 7 |
| ResNet-18 | `model.layer4[-1]` (last BasicBlock) | 512 × 7 × 7 |

Why the **last** conv layer?
- It has the most **high-level, class-specific** features.
- It still keeps **spatial** layout (after it comes global pooling, which throws location away).
- Earlier layers give noisy, edge-like maps; using the classifier layer gives no spatial map at all.

The trade-off: 7×7 is coarse, so heatmaps are blob-like rather than pixel-precise.

## 4. What to look for

From literature and experience, StyleGAN fakes often trigger attention around:

| Region | Typical artifact |
|--------|------------------|
| Eyes | mismatched reflections, irregular pupils |
| Hairline / hair strands | blurry, "painted" texture, strands merging into background |
| Teeth | merged or irregular teeth |
| Ears / earrings | asymmetric, malformed |
| Background edges | warped, nonsensical textures, "water droplet" blobs |

For **real** faces, the model often spreads attention over natural skin texture or the
background (real camera noise).

When analysing **misclassified** images, ask:
- Is the model looking at the face at all, or at the background/borders?
- Did a real photo have an unusual feature (filter, make-up, blur) in the highlighted region?

Generated grids:

| File | Content |
|------|---------|
| `outputs/heatmaps/<model>_correct_real.png` | 12 correctly classified real faces |
| `outputs/heatmaps/<model>_correct_fake.png` | 12 correctly classified fake faces |
| `outputs/heatmaps/<model>_misclassified.png` | most confident errors from Module 5 |

## 5. Limitations of Grad-CAM

- **Coarse resolution** (7×7 upsampled) — shows *roughly where*, not the exact pixels.
- Shows **where**, not **what** (e.g. it can't say "the frequency pattern of the hair").
- Gradients can saturate; the map depends on the chosen layer.
- A heatmap is a *hint* about the model's reasoning, not a proof.

Possible extensions: Grad-CAM++, Score-CAM, Integrated Gradients.
