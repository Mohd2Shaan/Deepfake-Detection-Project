# 01 — Dataset, Preprocessing & Augmentation

**Code:** [`src/dataset.py`](../src/dataset.py), [`src/config.py`](../src/config.py),
[`scripts/download_dataset.py`](../scripts/download_dataset.py)

---

## 1. Getting the data

```bash
python scripts/download_dataset.py                     # uses kagglehub
# or, if you downloaded the zip manually:
python scripts/download_dataset.py --source path/to/real_vs_fake
```

The script finds the folder holding `train/ valid/ test/` and copies it into `data/`.

| Split | Real | Fake | Total |
|-------|------|------|-------|
| train | 50,000 | 50,000 | 100,000 |
| valid | 10,000 | 10,000 | 20,000 |
| test  | 10,000 | 10,000 | 20,000 |

### Why we never reshuffle the splits (data leakage)

**Data leakage** = information from the test set "leaks" into training, so test scores look
better than the model really is. Some real/fake images in this dataset are related (e.g. share
a seed face). If we merged and re-split randomly, near-duplicates could land on both sides
and the model would be partially *memorising* instead of *generalising*. So we use the
provided splits exactly as given.

---

## 2. Labels

| Folder | Label | Meaning |
|--------|-------|---------|
| `real/` | 0 | real photograph |
| `fake/` | 1 | StyleGAN-generated |

Labels are returned as **float** tensors (`0.0` / `1.0`) because the loss we use,
`BCEWithLogitsLoss`, compares a real-valued logit with a float target.
With this labelling, `sigmoid(model_output)` directly means **P(image is fake)**.

---

## 3. Preprocessing

### 3.1 Resize to 224×224

The dataset images are 256×256. EfficientNet-B0 and ResNet-18 were pretrained on ImageNet at
224×224. Using the same size:
- matches the receptive-field statistics the pretrained filters expect,
- saves ~23% compute per image (224² vs 256²).

(CNNs with global average pooling *can* take any size, so 256 would not crash — it is just
wasteful and slightly mismatched.)

### 3.2 `ToTensor()`

Converts a PIL image (H×W×C, values 0–255) into a PyTorch tensor (C×H×W, values 0.0–1.0).

### 3.3 Normalisation with ImageNet statistics

For every colour channel *c*:

```
x_norm = (x - mean_c) / std_c
mean = [0.485, 0.456, 0.406]     std = [0.229, 0.224, 0.225]     (R, G, B)
```

These are the mean/std of the ImageNet training set. The pretrained weights were learned on
inputs normalised this way, so we must feed inputs with the **same distribution**; otherwise
the first-layer activations are shifted and the pretrained features become less useful.

Normalised (zero-centred, unit-variance) inputs also keep gradients well-scaled and make
optimisation faster.

`denormalize()` reverses this so we can display tensors as images.

---

## 4. Data Augmentation (training set only)

Augmentation = applying random, label-preserving changes to training images so the model
sees a slightly different version of every image each epoch. It acts as a **regulariser**:
the network cannot memorise exact pixels and must learn more robust features.

| Transform | What it does | Why it makes sense here |
|-----------|--------------|-------------------------|
| `RandomHorizontalFlip(p=0.5)` | mirror left↔right | a mirrored face is still real/fake |
| `RandomRotation(10)` | rotate by a random angle in [-10°, +10°] | small head tilts occur in real photos |
| `ColorJitter(brightness=0.2, contrast=0.2)` | random ±20% brightness/contrast | different lighting/cameras; stops the model using global colour as a shortcut |

**Why no augmentation on valid/test?** Validation and test sets must measure performance on
the *real* data distribution, and must be **deterministic** so results are reproducible and
comparable across epochs and models. So they only get resize + normalise.

**Why not stronger augmentation (blur, JPEG compression, heavy crops)?** The fake-detection
signal often lives in fine, high-frequency details. Blurring or heavy compression could
destroy exactly the artifacts we want the model to learn. We keep augmentation mild.

---

## 5. PyTorch `Dataset` and `DataLoader`

### `Dataset`
A class that answers two questions:
- `__len__()` — how many samples are there?
- `__getitem__(i)` — give me sample *i* as `(image_tensor, label)`.

`DeepfakeDataset` builds a list of `(path, label)` pairs at start-up (cheap) and only opens an
image file when `__getitem__` is called (lazy loading) — so 140k images never sit in RAM.

The optional `max_per_class` argument picks a random, fixed (seeded) subset per class — useful
for quick experiments on a CPU.

### `DataLoader`
Wraps a dataset and handles:
- **batching** — stacks 32 samples into one tensor `[32, 3, 224, 224]`,
- **shuffling** — only for training, so each epoch sees batches in a new order
  (valid/test stay ordered so predictions line up with file paths for error analysis),
- **parallel loading** — `num_workers` background processes read/decode images while the GPU trains,
- **pin_memory** — speeds up CPU→GPU copies when CUDA is available.

### Why batch size 32?
A batch is a compromise:
- too small → noisy gradient estimates, slow (GPU under-used),
- too large → may not fit in GPU memory, and can generalise slightly worse.

32 fits comfortably in a Colab T4 GPU for both models at 224×224.

---

## 6. Sanity check

```bash
python -m src.dataset
```

prints the size and class counts of each split, the shape of one batch, and saves
`outputs/plots/sample_train_batch.png` — a grid of augmented training images with their labels.
Looking at this grid confirms that (1) images load correctly, (2) labels match folders, and
(3) augmentations look reasonable (not too strong).
