# 08 — Report Outline, Literature & Viva Preparation

## 1. Report structure (12–15 pages)

| Section | Content | Material in this repo |
|---------|---------|-----------------------|
| 1. Problem Statement | deepfakes, why detection matters, our binary task | `docs/00` |
| 2. Literature Review | 3–4 papers (see §2) | below |
| 3. Dataset & EDA | source, splits, balance, checks, sample grids, spectrum | `docs/02`, `outputs/plots/eda_*` |
| 4. Methodology | preprocessing, augmentation, architectures, 2-phase transfer learning, loss/optimizer | `docs/01`, `docs/03`, `docs/04` |
| 5. Results | training curves, metrics table, confusion matrices, ROC, comparison table | `docs/05`, `outputs/plots/`, `outputs/model_comparison.csv` |
| 6. Explainability | Grad-CAM theory + heatmap grids + observations | `docs/06`, `outputs/heatmaps/` |
| 7. Demo | app screenshots, OOD tests | `docs/07` |
| 8. Discussion | what worked, what didn't, limitations | §3 below |
| 9. Conclusion & Future work | | §4 below |
| References | | §2 below |

## 2. Literature review — papers to cite

1. **Rössler, A. et al. (2019).** *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV.
   Large benchmark of face-manipulation videos (Deepfakes, Face2Face, FaceSwap, NeuralTextures);
   shows CNNs (XceptionNet) beat hand-crafted features, and that compression strongly hurts detection.
2. **Karras, T., Laine, S., Aila, T. (2019).** *A Style-Based Generator Architecture for Generative
   Adversarial Networks (StyleGAN).* CVPR. The generator that produced our fake images; also introduced FFHQ.
3. **Wang, S.-Y. et al. (2020).** *CNN-generated images are surprisingly easy to spot… for now.* CVPR.
   A ResNet-50 trained on ProGAN images with augmentation generalises surprisingly well to other
   GANs — supports our transfer-learning approach and discusses generalisation.
4. **Survey:** Tolosana, R. et al. (2020). *DeepFakes and Beyond: A Survey of Face Manipulation and
   Fake Detection.* Information Fusion. Categorises entire-face synthesis vs. attribute/identity
   manipulation and summarises detection methods.
5. **Selvaraju, R. R. et al. (2017).** *Grad-CAM: Visual Explanations from Deep Networks via
   Gradient-based Localization.* ICCV. Our explainability method.
6. **Tan, M., Le, Q. (2019).** *EfficientNet: Rethinking Model Scaling for CNNs.* ICML.
7. **He, K. et al. (2016).** *Deep Residual Learning for Image Recognition.* CVPR.
8. (optional) **Frank, J. et al. (2020).** *Leveraging Frequency Analysis for Deep Fake Image Recognition.* ICML —
   GAN up-sampling artifacts in the frequency domain (links to our spectrum plot).

Dataset: *140k Real and Fake Faces*, Kaggle (xhlulu), built from FFHQ + StyleGAN images.

## 3. Discussion / limitations checklist

- Very high accuracy is expected because train and test come from **one generator** (StyleGAN).
- No test on other generators → **generalisation unknown**.
- Robustness to JPEG compression, resizing, blur, screenshots not evaluated.
- Faces are aligned/cropped like FFHQ; arbitrary photos need face detection + alignment first.
- Grad-CAM maps are coarse (7×7) and only indicate *where*, not *what*.
- Single images only — no temporal (video) cues.

## 4. Future work

- Train/test on multiple generators (StyleGAN2/3, diffusion: Stable Diffusion, Midjourney, DALL·E).
- Add a face detector (MTCNN / RetinaFace) before classification in the app.
- Robustness augmentation: JPEG compression, blur, resizing.
- Frequency-domain features or two-stream (RGB + FFT) models.
- Video deepfakes (FaceForensics++, DFDC) with temporal models.
- Model calibration (temperature scaling) so confidence values are more trustworthy.

## 5. Presentation (10–12 slides)

1. Title
2. Problem — what are deepfakes, why detect them
3. Dataset — 140k faces, splits, sample grid
4. Pipeline / methodology diagram
5. Architectures — EfficientNet-B0 vs ResNet-18
6. Training strategy — 2-phase transfer learning + curves
7. Results — metrics, confusion matrix, ROC
8. Model comparison table
9. Grad-CAM findings — real vs fake heatmaps, error heatmaps
10. Live demo
11. Limitations
12. Future work & conclusion

## 6. Viva — likely questions with short answers

| Question | One-line answer |
|----------|-----------------|
| Why EfficientNet over other architectures? | Best accuracy-per-parameter on ImageNet thanks to compound scaling and MBConv+SE blocks; ~4M params trains fast on free Colab. |
| Why also ResNet-18? | A well-known, simple residual baseline to show our result isn't architecture-specific and to compare cost vs accuracy. |
| Why transfer learning instead of training from scratch? | Pretrained low-level features transfer to any images; converges in a few epochs with far less data/compute and overfits less. |
| Why freeze first, then unfreeze? | A random new head produces large gradients that would wreck pretrained features; warm it up first, then fine-tune the last blocks with a 10× smaller lr. |
| Why BCEWithLogitsLoss? | Binary task with a single logit; it fuses sigmoid + BCE in a numerically stable way. |
| Why 224×224? | Matches ImageNet pretraining resolution; 256 works but wastes ~23% compute. |
| Why ImageNet mean/std normalisation? | Pretrained weights expect inputs with that distribution. |
| Why augment only the training set? | Augmentation is a regulariser for learning; evaluation must be on unmodified, deterministic data. |
| How did you avoid data leakage? | Used Kaggle's predefined splits without reshuffling; test set touched only once for final evaluation; checked duplicates via MD5. |
| Why choose the checkpoint by validation loss? | Early stopping — loss is more sensitive than accuracy near 99% and reflects confidence/over-fitting. |
| What does ROC-AUC mean? | Probability a random fake is scored higher than a random real; threshold-independent. |
| How does Grad-CAM work? | Weights last-conv feature maps by the average gradient of the class score, sums them, ReLU, upsample → heatmap. |
| Would this work on diffusion-model fakes? | Probably not reliably — different generators leave different artifacts; would need training data from those generators. |
| What's the computational cost? | EfficientNet-B0 ≈ 0.39 GFLOPs and ~4M params per 224×224 image — milliseconds on GPU, well under a second on CPU. ResNet-18 ≈ 1.8 GFLOPs, 11M params. |
| How would you deploy at scale? | Export to ONNX/TorchScript, serve behind a REST API with batching on GPU (e.g. TorchServe/Triton), add face detection, monitor drift and retrain on new generators. |
| Why is accuracy so high — is it overfitting? | No: it is measured on an unseen test set. StyleGAN fakes are "easy" because of consistent generator fingerprints; the real challenge is cross-generator generalisation. |
| What are the limitations? | Single generator, aligned faces only, no robustness tests, coarse explanations, over-confidence on OOD inputs. |
