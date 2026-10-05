# 04 — Training (Transfer Learning in Two Phases)

**Code:** [`src/train.py`](../src/train.py) · **Notebook:** [`notebooks/02_training.ipynb`](../notebooks/02_training.ipynb)

```bash
python -m src.train --model efficientnet_b0
python -m src.train --model resnet18
python -m src.train --model resnet18 --resume                       # continue after a disconnect
python -m src.train --model efficientnet_b0 --max-per-class 500 \
       --epochs-frozen 1 --epochs-finetune 1                         # quick smoke test on CPU
```

---

## 1. The training loop — what happens in one step

```
for each batch (images, labels):
    1. forward pass        logits = model(images)
    2. compute loss        loss = BCEWithLogits(logits, labels)
    3. zero old gradients  optimizer.zero_grad()
    4. backward pass       loss.backward()           # ∂loss/∂w for every trainable w (backpropagation)
    5. update weights      optimizer.step()          # w ← w − lr · (adjusted) gradient
```

One pass over the whole training set is an **epoch**. After every epoch we run the model on
the **validation set** (no gradient, `model.eval()`) to see how well it generalises.

`model.train()` vs `model.eval()`:
- **Dropout** is active only in train mode.
- **BatchNorm** uses batch statistics in train mode, stored running statistics in eval mode.

---

## 2. Loss function — Binary Cross-Entropy with Logits

For a true label *y* ∈ {0, 1} and predicted probability *p* = σ(z):

```
BCE(y, p) = −[ y · log(p) + (1 − y) · log(1 − p) ]
```

- If the image is fake (y = 1), loss = −log p → small when p is close to 1.
- If real (y = 0), loss = −log(1 − p) → small when p is close to 0.
- Confident wrong predictions are punished heavily (log → −∞).

`BCEWithLogitsLoss` takes the raw logit *z* and combines sigmoid + BCE in one numerically
stable formula (uses the log-sum-exp trick), which is why the model has **no sigmoid layer**
at the end — we only apply `sigmoid` at prediction time.

---

## 3. Optimizer — Adam

Plain gradient descent: `w ← w − lr · g`.

**Adam** (Kingma & Ba, 2015) keeps two running averages per parameter:

```
m = β1·m + (1−β1)·g          # momentum: smoothed gradient direction
v = β2·v + (1−β2)·g²         # smoothed squared gradient (scale)
w ← w − lr · m̂ / (√v̂ + ε)    # m̂, v̂ = bias-corrected m, v
```

- Momentum smooths noisy mini-batch gradients.
- Dividing by √v gives every parameter its own adaptive step size.
- Works well out of the box with little tuning — good default for fine-tuning.

We only give the optimizer parameters with `requires_grad=True`, so frozen layers are untouched.

---

## 4. Two-phase transfer learning

| Phase | Epochs | Trainable | Learning rate | Why |
|-------|--------|-----------|---------------|-----|
| 1. Warm-up | 3 | new head only | 1e-4 | The new `Linear` layer starts random. If we trained everything at once, its large random gradients would flow back and **damage the pretrained features**. So first we let the head learn on top of fixed features. |
| 2. Fine-tune | 7 | head + last 2 blocks | 1e-5 | Now the head is sensible, we let the last (most task-specific) blocks adapt to deepfake artifacts. A **10× smaller lr** makes small, careful changes so we don't destroy useful pretrained knowledge ("catastrophic forgetting"). |

Why only the **last** blocks? Early layers detect generic edges/textures that are useful as-is;
later layers encode ImageNet-specific concepts (dog ears, car wheels) that need to change most.
Fine-tuning fewer layers also means less overfitting risk and faster epochs.

When the phase changes, a **new optimizer** is created (new parameter set, new lr).

---

## 5. Learning-rate scheduler — ReduceLROnPlateau

Monitors the validation loss. If it does not improve for `patience=1` epoch, the learning rate
is multiplied by `factor=0.5`. Smaller steps help the model settle into a minimum once
progress stalls.

---

## 6. Checkpointing — saving the best model

- **`<model>_best.pth`** — saved whenever validation loss reaches a new minimum. This is the
  model we evaluate and deploy. Choosing by validation loss (not training loss) is a form of
  **early stopping**: if later epochs overfit, we still keep the best-generalising weights.
- **`<model>_last.pth`** — saved every epoch, including the optimizer, scheduler, history and
  current phase. With `--resume` training continues from the next epoch.

On Colab, pass `--checkpoint-dir /content/drive/MyDrive/...` so both files are written to Google
Drive and survive a session disconnect.

### Why validation *loss* and not accuracy?
Accuracy only changes when a prediction crosses 0.5; loss also reflects **how confident** the
model is. Near 99% accuracy, accuracy barely moves, while loss still shows whether the model is
improving or becoming over-confident.

---

## 7. Overfitting — what to watch in the curves

`outputs/plots/training_curves_<model>.png` shows loss and accuracy per epoch (dashed line =
start of fine-tuning).

| Pattern | Meaning |
|---------|---------|
| train ↓, valid ↓ | learning well |
| train ↓, valid ↑ | **overfitting** — memorising training data |
| both high, flat | **underfitting** — model/lr too weak |

Our defences against overfitting: pretrained weights, data augmentation, dropout in the
EfficientNet head, freezing most layers, small fine-tune lr, early stopping by best val loss.

---

## 8. Mixed precision (GPU only)

On a GPU, `torch.autocast` runs most operations in 16-bit floats (faster, half the memory) while
keeping sensitive ones (loss, reductions) in 32-bit. `GradScaler` multiplies the loss before
`backward()` so small fp16 gradients don't underflow to zero, then un-scales before the update.
This roughly speeds up training ~2× on Colab's T4. On CPU it is disabled automatically.

---

## 9. Reproducibility

`set_seed(42)` fixes Python, NumPy and PyTorch random generators so data shuffling, augmentation
and head initialisation are the same across runs (GPU kernels may still introduce tiny
non-determinism).

## 10. Outputs

| File | Content |
|------|---------|
| `checkpoints/efficientnet_best.pth`, `resnet_best.pth` | best weights |
| `checkpoints/*_last.pth` | resume state |
| `outputs/history_<model>.json` | loss/acc/lr per epoch |
| `outputs/plots/training_curves_<model>.png` | curves |

## 11. Expected time (Colab T4, full dataset, batch 32)

Roughly 3,125 training batches per epoch. Frozen epochs are faster (no backbone gradients).
Expect on the order of 5–10 minutes per epoch for EfficientNet-B0 with mixed precision, so a full
10-epoch run is about 1–1.5 hours per model. Use `--resume` if the session drops.
