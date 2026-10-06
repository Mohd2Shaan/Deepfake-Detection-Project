# 02 — Exploratory Data Analysis (EDA)

**Code:** [`src/eda.py`](../src/eda.py) · **Notebook:** [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)

```bash
python -m src.eda                      # full dataset (takes a few minutes)
python -m src.eda --max-per-class 2000 # faster, on a random subset
```

---

## 1. Why do EDA before training?

EDA means *looking at the data before modelling it*. "Garbage in, garbage out": if the data
has problems (wrong labels, corrupted files, imbalance, duplicates), no model can fix them,
and the metrics we report would be misleading. EDA answers:

1. Is the data what we think it is?
2. Are there problems we must fix first?
3. Are there obvious "shortcuts" the model might exploit instead of learning the real task?

---

## 2. Checks we perform

### 2.1 Class balance
Counts of real vs fake per split. The dataset should be exactly **50/50**.

*Why it matters:* with imbalanced classes (say 90% real), a model that always says "real"
gets 90% accuracy while being useless. With a balanced dataset, accuracy is meaningful and
the default decision threshold of 0.5 is appropriate.

### 2.2 Image dimensions & colour mode
Expected: all 256×256, RGB. If some images were grayscale or a different size, the
transform pipeline must handle it (we always call `.convert("RGB")` and `Resize(224, 224)`).

### 2.3 Corrupted files
Each file is opened with `PIL.Image.verify()`, which checks the file structure without fully
decoding it. A single corrupted file would crash a training run hours in — better to find it
up front.

### 2.4 Duplicates
We compute the **MD5 hash** of every file's bytes. Identical hashes ⇒ byte-identical files.
Duplicates *across* splits are a form of **data leakage** (the model would be tested on an
image it trained on).

> MD5 only finds *exact* duplicates. Near-duplicates (same face, re-encoded) would need
> perceptual hashing (pHash) — mentioned as a possible extension.

### 2.5 Sample grids
Visual inspection of random real and fake faces. Humans usually *cannot* tell them apart at
a glance — which motivates the project.

### 2.6 File-size distribution (shortcut check)
If real and fake JPEGs had very different file sizes / compression levels, a model could learn
to detect **compression** instead of **generation artifacts** — a "shortcut". Plotting the
file-size histograms per class tells us whether such a trivial signal exists. Worth discussing
in the report either way.

### 2.7 Mean face & mean Fourier spectrum
- **Mean face:** the pixel-wise average of N images per class. Shows alignment — both
  datasets are face-centred and aligned (FFHQ-style alignment), so the average is a blurry face.
- **Mean frequency spectrum:** for each grayscale image we compute the 2D **Discrete Fourier
  Transform** and average the log-magnitude.

#### Quick Fourier theory
Any image can be written as a sum of 2D sine/cosine waves of different frequencies. The DFT
gives the amplitude of each frequency:

```
F(u, v) = Σ_x Σ_y  f(x, y) · e^{-2πi (ux/M + vy/N)}
```

After `fftshift`, low frequencies (smooth regions, overall shape) are in the centre and high
frequencies (edges, fine texture, noise) towards the borders.

GAN generators build images by **up-sampling** (transposed convolutions / interpolation +
convolution). This can leave periodic, grid-like traces that appear as **bright spots or lines
in the high-frequency part of the spectrum** (Durall et al., 2020; Frank et al., 2020). These
are often invisible in pixel space but are exactly the kind of signal a CNN can pick up.

---

## 3. Results (full dataset, run on Kaggle)

See the executed notebook [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb) for all plots.

| Item | Result |
|------|--------|
| Images per split | train 100,000 · valid 20,000 · test 20,000 |
| Class ratio | exactly 50 / 50 in every split |
| Image size / mode | all 256 × 256, RGB |
| Corrupted files | 0 |
| Exact duplicates (MD5) | 0 (within and across splits) |
| Mean JPEG size | real 28.26 KB (σ 3.88) · fake 27.76 KB (σ 3.86) |
| Full scan time | ~17 min on Kaggle CPU (140 images/s) |

### Findings
1. **Clean, balanced data** — no cleaning step needed; accuracy is a fair metric.
2. **No compression shortcut** — file-size distributions overlap almost completely (difference ≈ 0.13 σ).
   Fakes are marginally smaller, consistent with slightly smoother GAN textures.
3. **Visually indistinguishable** — random samples look like normal portraits; some fakes have odd
   backgrounds (warped objects, partial second faces) that Grad-CAM may later highlight.
4. **Mean faces** — both classes are aligned the same way; the mean *fake* face is sharper, i.e. StyleGAN
   outputs are more uniform in pose and alignment than real photos.
5. **Mean spectra are nearly identical** — no obvious up-sampling peaks in the *average* spectrum at
   256 px (the bright cross comes from image borders). The discriminative signal is subtle and local, which
   motivates a CNN instead of a hand-crafted frequency rule.
6. **Pipeline check passed** — labels match, augmentations are mild; rotation's black corners appear in both
   classes equally, so they don't leak the label.
