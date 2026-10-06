# Changelog

All notable changes to this project are documented here, module by module.

## [Experiment] EfficientNet-B0 with 4 unfrozen blocks
- New `notebooks/02b_efficientnet_unfreeze4.ipynb`: retrains EfficientNet-B0 with `--unfreeze-blocks 4`
  (92% of weights, comparable to ResNet's 94%) and `--lr-finetune 1e-4`, because run 02 was under-fitting
- Run-02 EfficientNet kept for comparison as `efficientnet_2blocks.pth` / `outputs/history_efficientnet_2blocks.json`
- Notebooks 03/04 now take their models from notebook 02b's output

## [Results] Training on the full dataset
- `notebooks/02_training.ipynb` executed on Kaggle (Tesla T4), both models, 3 frozen + 7 fine-tune epochs
- Best validation accuracy: EfficientNet-B0 93.71% · ResNet-18 98.96%
- Training histories committed in `outputs/history_*.json`; analysis in `docs/04_training.md` §11

## [Results] EDA on the full dataset
- `notebooks/01_eda.ipynb` executed on Kaggle over all 140,000 images (outputs and plots included)
- Observations written in the notebook and in `docs/02_exploratory_data_analysis.md` §3:
  balanced, 0 corrupted, 0 duplicates, no file-size shortcut, near-identical mean spectra

## [Fix] Kaggle dataset detection
- Dataset lookup moved into `src/config.py` (`find_kaggle_dataset`) and searches `/kaggle/input` at any
  depth — Kaggle now mounts datasets under `/kaggle/input/datasets/<owner>/<slug>/...`
- `import_kaggle_checkpoints()` replaces the notebook-side checkpoint glob
- Notebook setup cell does `git pull` when the project is already cloned

## [Kaggle support]
- Notebook setup cell detects Kaggle / Colab / local; on Kaggle it clones the repo and uses the
  attached dataset directly via `DEEPFAKE_DATA_DIR`
- All four notebooks run on Kaggle; 03/04 pick up trained checkpoints from notebook 02's attached output
- `02_training.ipynb` zips the trained models into `results.zip`; EDA scans the full dataset by default
- README: step-by-step Kaggle instructions for every notebook

## [Module 8] — Report material & viva preparation
- `docs/08_report_and_viva.md`: report structure, literature list, limitations, future work,
  slide outline, viva Q&A
- README: usage table, results placeholder, documentation index

## [Module 7] — Streamlit demo app
- `app.py`: plain Streamlit UI — model dropdown, image upload, REAL/FAKE prediction,
  confidence, P(fake) bar, Grad-CAM overlay (cached model loading)
- `docs/07_streamlit_app.md` (how Streamlit reruns work, confidence vs probability, OOD testing)
- `requirements.txt`: `streamlit>=1.46`

## [Module 6] — Grad-CAM explainability
- `src/gradcam.py`: `GradCAMExplainer` (pytorch-grad-cam, binary single-logit targets),
  hand-written `manual_gradcam` (verified identical to the library), heatmap grids for correctly
  classified real/fake faces and for misclassified images
- `notebooks/04_gradcam.ipynb`
- `docs/06_gradcam.md`

## [Module 5] — Evaluation & error analysis
- `src/evaluate.py`: test-set predictions, classification report, accuracy/precision/recall/F1/ROC-AUC,
  confusion matrices, shared ROC plot, model comparison CSV, most-confident-mistakes grid + CSV
- `notebooks/03_evaluation.ipynb`
- `docs/05_evaluation.md`

## [Module 4] — Training pipeline
- `src/train.py`: two-phase transfer learning (frozen head warm-up → fine-tune last 2 blocks),
  BCEWithLogitsLoss, Adam, ReduceLROnPlateau, best-by-val-loss checkpoint, `--resume` from
  `<model>_last.pth`, mixed precision on GPU, training-curve plot + JSON history
- `notebooks/02_training.ipynb`: Colab workflow with checkpoints saved to Google Drive
- `docs/04_training.md`

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
