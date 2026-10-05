# Changelog

All notable changes to this project are documented here, module by module.

## [Module 1] — Dataset, preprocessing & augmentation
- `src/config.py`: central paths and hyper-parameters (`DEEPFAKE_DATA_DIR` env override)
- `src/dataset.py`: `DeepfakeDataset`, train/eval transforms, DataLoader helpers, batch sanity grid
- `scripts/download_dataset.py`: download via kagglehub or arrange an existing extract into `data/`
- `docs/01_dataset_and_preprocessing.md`

## [Module 0] — Project scaffolding
- Created folder structure (`data/`, `src/`, `notebooks/`, `checkpoints/`, `outputs/`, `docs/`)
- Added `README.md`, `requirements.txt`, `.gitignore`, `LICENSE`
- Added `docs/00_project_overview.md` (problem statement, GAN/StyleGAN theory, pipeline)
