# Changelog

All notable changes to this project are documented here, module by module.

## [Module 3] — Model definitions
- `src/model.py`: EfficientNet-B0 / ResNet-18 builders with `Linear(*, 1)` heads,
  freeze / unfreeze-last-blocks helpers, frozen-BatchNorm handling, Grad-CAM target layers,
  self-describing checkpoint save/load
- `docs/03_model_architecture.md` (CNN recap, compound scaling, MBConv/SE, residual blocks, parameter counts)

## [Module 2] — Exploratory Data Analysis
- `src/eda.py`: image scan (size, mode, corruption via `verify()`, MD5 duplicates), class balance,
  sample grids, file-size shortcut check, mean face and mean Fourier spectrum
- `notebooks/01_eda.ipynb` (runs locally or on Colab)
- `docs/02_exploratory_data_analysis.md`

## [Module 1] — Dataset, preprocessing & augmentation
- `src/config.py`: central paths and hyper-parameters (`DEEPFAKE_DATA_DIR` env override)
- `src/dataset.py`: `DeepfakeDataset`, train/eval transforms, DataLoader helpers, batch sanity grid
- `scripts/download_dataset.py`: download via kagglehub or arrange an existing extract into `data/`
- `docs/01_dataset_and_preprocessing.md`

## [Module 0] — Project scaffolding
- Created folder structure (`data/`, `src/`, `notebooks/`, `checkpoints/`, `outputs/`, `docs/`)
- Added `README.md`, `requirements.txt`, `.gitignore`, `LICENSE`
- Added `docs/00_project_overview.md` (problem statement, GAN/StyleGAN theory, pipeline)
