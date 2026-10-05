# 05 — Evaluation, Model Comparison & Error Analysis

**Code:** [`src/evaluate.py`](../src/evaluate.py) · **Notebook:** [`notebooks/03_evaluation.ipynb`](../notebooks/03_evaluation.ipynb)

```bash
python -m src.evaluate                       # all checkpoints/*_best.pth
python -m src.evaluate --checkpoints checkpoints/efficientnet_best.pth
```

The **test set** (20,000 images) is used **only here**, once training is finished. It was never
used to choose hyper-parameters or checkpoints, so it gives an unbiased estimate of performance
on unseen data.

---

## 1. Confusion matrix

Treating **fake as the positive class** (label 1):

|                 | Predicted real | Predicted fake |
|-----------------|----------------|----------------|
| **Actually real** | TN (true negative) | FP (false positive — real photo flagged as fake) |
| **Actually fake** | FN (false negative — fake slipped through) | TP (true positive) |

All metrics below are computed from these four numbers.

## 2. Metrics

| Metric | Formula | Question it answers |
|--------|---------|---------------------|
| **Accuracy** | (TP + TN) / total | What fraction of all images is classified correctly? |
| **Precision** | TP / (TP + FP) | When the model says "fake", how often is it right? |
| **Recall** (sensitivity, TPR) | TP / (TP + FN) | Of all fakes, how many did we catch? |
| **F1-score** | 2 · P · R / (P + R) | Harmonic mean of precision and recall — high only if *both* are high |
| **ROC-AUC** | area under ROC curve | How well does the model *rank* fakes above reals, over all thresholds? |

`classification_report` prints precision/recall/F1 for **each** class (real and fake) plus
macro and weighted averages.

### Which errors matter more?
- **False negative** (fake accepted as real) → misinformation spreads. Important for a
  content-moderation system.
- **False positive** (real flagged as fake) → a genuine person is wrongly accused.

The threshold (default 0.5) can be moved to trade one against the other — the ROC curve shows
this trade-off.

## 3. ROC curve and AUC

For every possible threshold *t* between 0 and 1, classify "fake" if P(fake) ≥ t and compute:

```
TPR (recall)  = TP / (TP + FN)
FPR           = FP / (FP + TN)
```

Plotting TPR vs FPR for all thresholds gives the **ROC curve** (Receiver Operating
Characteristic).

- Perfect classifier: goes straight up to (0, 1) → AUC = 1.0
- Random guessing: diagonal line → AUC = 0.5

**AUC interpretation:** the probability that a randomly chosen fake image gets a higher
P(fake) than a randomly chosen real image. It is **threshold-independent**, so it measures the
quality of the model's scores, not just of one decision rule.

Both models are drawn on the same plot: `outputs/plots/roc_curve.png`.

## 4. Model comparison table

Saved as `outputs/model_comparison.csv`:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|----|---------|
| EfficientNet-B0 | *fill after training* | | | | |
| ResNet-18 | | | | | |

Points to discuss in the report:
- accuracy per parameter (EfficientNet has ~3× fewer parameters),
- training time per epoch,
- how much was unfrozen in phase 2 (28% vs 94% of weights — see doc 03).

## 5. Error analysis — most confident mistakes

For every misclassified test image:

```
confidence = P(fake)       if predicted fake
           = 1 − P(fake)   if predicted real
```

We sort the errors by confidence and keep the top 25. These are the cases where the model
was **sure and wrong** — the most informative failures:

- real photos with unusual lighting, heavy make-up, filters, or low quality that "look generated",
- very clean StyleGAN images with no visible artifacts,
- images with unusual backgrounds or accessories.

Outputs:
- `outputs/misclassified/<model>_most_confident_errors.png` — image grid with true/pred/confidence
- `outputs/misclassified/<model>_most_confident_errors.csv` — paths + scores (used by Grad-CAM in Module 6)

## 6. "Why is accuracy so high?" — be honest in the report

StyleGAN (2019) faces are now considered **easy fakes**: they carry consistent generator
fingerprints, and train/test come from the **same generator**. 97–99% test accuracy is expected.
This does **not** mean the model detects deepfakes in general:

- a different generator (StyleGAN3, diffusion models, Midjourney) has different artifacts,
- re-compression, resizing or screenshots can erase the fingerprint,
- face-swap videos (FaceForensics++) are a different problem altogether.

This generalisation gap should be stated clearly in the Discussion/Limitations section.
