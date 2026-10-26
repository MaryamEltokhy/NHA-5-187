# Brain MRI Project — Dataset Documentation

**Task:** M1-T01 — Choose the datasets and get access
**Owner:** Alaa Emad
**Status:** Datasets downloaded ✅

## Project Overview
AI-Powered Brain MRI Tumor Analysis, Segmentation, and Quantitative Assessment System.

## Datasets Used

### Primary (Training)
**BraTS 2021 Task 1** — Kaggle mirror
https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1
- 1,251 training + 219 validation cases
- Modalities: T1, T1ce, T2, FLAIR + segmentation mask
- Labels: 1 (NCR/NET), 2 (ED), 4 (ET)

### Supplementary (Post-Treatment)
**BraTS 2024 Adult Glioma Post-Treatment** — Kaggle mirror
https://www.kaggle.com/datasets/i212385nomanarif/2024-brats-glioma
- ⚠️ Partial dataset: 700 cases only (official release has ~1,621 timepoints)
- Labels: 1 (NETC), 2 (SNFH), 3 (ET), 4 (RC)

### External Test Sets (Generalization)
Used to validate the model on independent, out-of-distribution data.

| Dataset | Subjects | Source |
|---|---|---|
| RHUH-GBM | 40 (3 timepoints each) | https://www.cancerimagingarchive.net/collection/rhuh-gbm/ |
| BraTS-Africa | 146 | https://www.cancerimagingarchive.net/collection/brats-africa/ |
| MU-Glioma-Post | 203 (~596 timepoints) | https://www.cancerimagingarchive.net/collection/mu-glioma-post/ |
| UTSW-Glioma | 625 | https://www.cancerimagingarchive.net/collection/utsw-glioma/ |

### Optional — MGMT Module
**ds007045** (OpenNeuro)
https://openneuro.org/datasets/ds007045
- 363 cases with complete MGMT methylation labels

## Important Notes
- BraTS 2021 and BraTS 2024 were downloaded from **Kaggle mirrors** (not the official TCIA/Synapse source) due to time constraints. These are community re-uploads, not the authoritative release.
- BraTS 2024 Kaggle mirror is **incomplete** (700/1,621 timepoints) and its license tag is inaccurate — the official license is CC BY-NC 4.0.
- No subject overlap was found between the primary training set and the external test sets (verified during dataset research phase).
- Label schemes differ slightly between datasets and require harmonization before training (see notebook).

## How to Reproduce
1. Open `datasets_notebook.ipynb` in Google Colab.
2. Add your Kaggle API token as a Colab Secret (`KAGGLE_API_TOKEN`).
3. Run cells top to bottom. External TCIA datasets require manually copying each collection's download link into the notebook (see instructions inside).

## License Compliance
All datasets used are permitted for non-commercial academic research. No raw data files are stored in this repository — only download/processing scripts.
