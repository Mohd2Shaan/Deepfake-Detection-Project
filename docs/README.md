# Project Documentation

This folder explains **what** we are building, **why** each choice was made, and **how** every
piece works — module by module. Each file covers the theory first and then how it is
implemented in the code.

| #  | Document | What it covers |
|----|----------|----------------|
| 00 | [Project Overview](00_project_overview.md) | Problem statement, what a deepfake is, overall pipeline |
| 01 | [Dataset & Preprocessing](01_dataset_and_preprocessing.md) | Data layout, leakage, resize/normalise, augmentation, Dataset/DataLoader |
| 02 | [Exploratory Data Analysis](02_exploratory_data_analysis.md) | Balance, sizes, corruption, duplicates, shortcut checks, Fourier spectrum |
| 03 | [Model Architecture](03_model_architecture.md) | CNN basics, EfficientNet-B0, ResNet-18, single-logit head, freezing |
| 04 | [Training](04_training.md) | Training loop, BCE loss, Adam, two-phase fine-tuning, scheduler, checkpoints, overfitting |
| 05 | [Evaluation](05_evaluation.md) | Confusion matrix, precision/recall/F1, ROC-AUC, model comparison, error analysis |
| 06 | [Grad-CAM](06_gradcam.md) | Explainability, Grad-CAM maths, target layers, interpreting heatmaps, limitations |
| 07 | [Streamlit App](07_streamlit_app.md) | Demo app flow, caching, confidence, out-of-distribution testing |
| 08 | [Report & Viva](08_report_and_viva.md) | Report outline, literature review, limitations, slides, viva Q&A |

Read them in order — each builds on the previous one.
