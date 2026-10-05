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

New documents are added here as each module is implemented.
