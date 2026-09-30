# PROJECT_STRUCTURE.md

**Project:** AI-Powered Brain MRI Tumor Analysis, Segmentation, and Quantitative Assessment System
**Program:** Digital Egypt Pioneers Initiative (DEPI) — AI & Data Science Track — Machine Learning
**Team:** Mohamed Tagy, Alaa Emad, Hagar Aiman, Yasmin Mahmoud, Maryam Ahmed, Essam Mohamed
**Scope of this document:** repository layout, code organization, dataset/model/experiment organization, naming conventions, Git strategy, checkpoint policy, environment management, secrets policy, and reproducibility strategy.

This document is the engineering contract for the repository. It is referenced by task IDs in the project plan and the Notion task board (for example `M1-T02` creates the skeleton described here). The original proposal is unchanged; this document only defines *how* the proposal is implemented.

---

## 1. Repository Layout

The structure below adapts the generic template to the actual proposal architecture (3D segmentation → quantitative analysis → longitudinal comparison → radiomics/MGMT → uncertainty/explainability → FastAPI/Docker/Azure → MLflow).

```text
brain-mri-tumor-system/
├── README.md                         # Project overview, quick start, safety statement (decision-support prototype, not diagnosis)
├── LICENSE                           # Code license (team decision; default MIT for code only — datasets keep their own licenses)
├── CITATION.cff                      # How to cite the project
├── .gitignore                        # Excludes data/, models/, secrets, MLflow artifacts, notebooks' checkpoints
├── .gitattributes                    # Git LFS rules (small demo assets only — see §7)
├── .env.example                      # Names of required environment variables, NO values
├── .pre-commit-config.yaml           # black, isort, ruff, nbstripout, secrets scan
├── pyproject.toml                    # Package metadata + tool configs (ruff/black/pytest)
├── environment/
│   ├── environment.yml               # Conda spec (pinned) for training workstation / Azure GPU VM
│   ├── requirements-train.txt        # pip lock for training (torch, monai, nnunetv2, nibabel, simpleitk ...)
│   ├── requirements-api.txt          # pip lock for inference API (no training-only deps)
│   ├── requirements-dev.txt          # pytest, ruff, black, pre-commit, mkdocs
│   └── cuda_notes.md                 # CUDA / driver / torch build matrix actually used
├── configs/
│   ├── data/
│   │   ├── brats2021.yaml            # Paths, modality file suffixes, label map, cohort role
│   │   ├── brats2024_post.yaml
│   │   ├── external_*.yaml           # One file per external / longitudinal cohort
│   │   └── label_harmonization.yaml  # Canonical label map → WT/TC/ET derivation rules per dataset
│   ├── preprocessing/
│   │   ├── harmonize_v1.yaml         # Orientation (RAS), spacing (1 mm iso), crop, normalization
│   │   └── augment_train_v1.yaml     # MONAI transform chain for training
│   ├── splits/
│   │   └── split_v1.yaml             # Seed, ratios, stratification, path to split manifest
│   ├── models/
│   │   ├── unet3d_baseline.yaml
│   │   ├── nnunet_3dfullres.yaml
│   │   ├── swin_unetr.yaml
│   │   └── segresnet_fallback.yaml   # Fallback if Swin UNETR is infeasible
│   ├── training/
│   │   ├── exp_template.yaml         # Every experiment starts from this template
│   │   └── EXP-*.yaml                # One frozen config per experiment ID
│   ├── inference/
│   │   └── inference_v1.yaml         # Sliding-window params, TTA, post-processing, thresholds
│   ├── quantitative/
│   │   └── morphology_v1.yaml        # Units, voxel-spacing source, diameter definition
│   ├── longitudinal/
│   │   └── registration_v1.yaml      # Rigid/affine params, similarity metric, QC thresholds
│   ├── radiomics/
│   │   └── pyradiomics_params.yaml   # PyRadiomics parameter file (bin width, resampling, filters)
│   ├── uncertainty/
│   │   └── mc_dropout_v1.yaml        # Samples, dropout rate, calibration bins
│   └── api/
│       └── api_settings.yaml         # Limits (upload size), timeouts, model version pins
├── data/                             # NOT committed (see .gitignore). Tracked by DVC (§4)
│   ├── raw/                          # Exact downloads, read-only, per source cohort
│   │   ├── brats2021/
│   │   ├── brats2024_post/
│   │   ├── external_<cohort>/
│   │   └── longitudinal_<cohort>/
│   ├── interim/                      # Reoriented / renamed / label-remapped but not yet normalized
│   ├── processed/                    # Harmonized volumes ready for training/inference (NIfTI or NPZ)
│   │   └── <cohort>/<subject_id>/<timepoint>/{t1,t1ce,t2,flair,seg}.nii.gz
│   ├── manifests/                    # CSV/Parquet indices (unified subject index, per-cohort manifests)
│   ├── splits/                       # split_v1_train.csv, split_v1_val.csv, split_v1_test.csv (patient-level)
│   ├── lockbox/                      # Held-out internal test + external test manifests; write-protected
│   └── demo/                         # ≤ 3 de-identified demo cases (small, permitted license only)
├── src/
│   └── bmts/                         # "brain MRI tumor system" importable package
│       ├── __init__.py
│       ├── common/                   # Logging, config loading, paths, seeds, typing, constants (label maps)
│       ├── data/
│       │   ├── indexing/             # Build unified subject index, overlap/leakage checks
│       │   ├── splitting/            # Patient-level splitting, stratification, lockbox creation
│       │   ├── harmonization/        # Orientation, spacing, modality naming, label remapping
│       │   ├── preprocessing/        # Intensity normalization, cropping/padding, tensor conversion
│       │   ├── augmentation/         # MONAI transform factories (training-only)
│       │   ├── datasets/             # PyTorch/MONAI Dataset + DataLoader builders (multi-channel T1/T1ce/T2/FLAIR)
│       │   └── quality/              # Data-quality checks, missing-modality handling
│       ├── segmentation/
│       │   ├── models/               # unet3d.py, swin_unetr.py, segresnet.py, nnunet_bridge.py
│       │   ├── losses/               # Dice, Dice+CE, Dice+Focal, region-based losses
│       │   ├── training/             # train loop, schedulers, checkpointing, AMP, resume
│       │   ├── evaluation/           # Dice/IoU/Sensitivity/Precision/HD95 per WT/TC/ET, lesion-wise optional
│       │   ├── postprocessing/       # Connected components, ET threshold rules, label reconstruction
│       │   └── inference/            # Sliding-window inference, TTA, model loading, ensemble
│       ├── quantitative/             # Volumes, diameters, surface area, sphericity, compactness, ratios
│       ├── longitudinal/
│       │   ├── verification/         # Subject-identity + date checks; reject/flag logic
│       │   ├── registration/         # SimpleITK rigid/affine registration + QC metrics
│       │   ├── change/               # Absolute/percent change, rate per day, trend classification
│       │   └── changemaps/           # Voxel-wise change map generation
│       ├── radiomics/                # PyRadiomics wrappers, feature tables, stability/ICC analysis
│       ├── prediction/               # Optional MGMT: feature pipelines, RF/XGBoost, deep-feature models
│       ├── uncertainty/              # MC dropout, ensembles, calibration (ECE, reliability), uncertainty maps
│       ├── explainability/           # Overlays, saliency/attention maps, SHAP/feature importance
│       ├── pipeline/                 # End-to-end orchestrator: preprocess → segment → quantify → (longitudinal)
│       ├── api/                      # FastAPI app: routers, schemas (pydantic), services, middleware, monitoring
│       └── monitoring/               # Latency/failure/data-quality/confidence-distribution logging
├── app/                              # Frontend (Streamlit or React+Niivue/Cornerstone — team decision in M4)
│   ├── src/
│   ├── assets/
│   └── README.md
├── nnunet/                           # nnU-Net workspace (env vars nnUNet_raw / nnUNet_preprocessed / nnUNet_results) — NOT committed except dataset.json + plans
│   ├── dataset_conversion/           # Scripts to convert processed data → nnU-Net raw format
│   └── plans/                        # Exported nnUNetPlans.json for reproducibility
├── experiments/                      # One folder per experiment ID (configs + logs + metrics, NO large weights)
│   └── EXP-0001_unet3d_baseline/
│       ├── config.yaml               # Frozen copy of the training config
│       ├── git_commit.txt
│       ├── metrics_val.json
│       ├── metrics_test.json         # Only for frozen final models (M5)
│       ├── train_log.csv
│       └── notes.md
├── models/                           # Checkpoints — NOT committed; registered in MLflow (see §6)
│   ├── checkpoints/
│   └── registry/                     # model_card.md + manifest.json per registered model version
├── notebooks/                        # Exploration only; must be stripped of outputs before commit
│   ├── 01_eda/
│   ├── 02_preprocessing_checks/
│   ├── 03_model_debug/
│   ├── 04_quantitative/
│   ├── 05_longitudinal/
│   └── 06_radiomics/
├── tests/
│   ├── unit/                         # Pure functions: metrics, label remapping, morphology, change calc
│   ├── data/                         # Dataset integrity, manifest schema, missing-modality, shape tests
│   ├── integration/                  # Preprocess→infer→quantify on a tiny synthetic volume
│   ├── api/                          # FastAPI TestClient tests, failure handling
│   ├── longitudinal/                 # Subject verification, registration QC thresholds
│   ├── reproducibility/              # Same seed → same metrics on a 2-case smoke run
│   └── fixtures/                     # Tiny synthetic NIfTI volumes (generated, ≤ 1 MB)
├── results/                          # Tables, figures, galleries — small files only
│   ├── eda/
│   ├── segmentation/
│   ├── quantitative/
│   ├── longitudinal/
│   ├── radiomics/
│   ├── uncertainty/
│   ├── final_validation/             # M5 frozen-protocol outputs
│   └── galleries/
├── docs/
│   ├── dataset_research.md           # Copy of the dataset research
│   ├── datasheets/                   # One datasheet per cohort (license, access, preprocessing, known issues)
│   ├── preprocessing_pipeline.md
│   ├── model_development.md
│   ├── evaluation_protocol.md        # Frozen before M5 test evaluation
│   ├── quantitative_analysis.md
│   ├── longitudinal_analysis.md
│   ├── radiomics.md
│   ├── uncertainty_explainability.md
│   ├── api_reference.md              # Generated from OpenAPI + narrative
│   ├── deployment.md
│   ├── mlops.md
│   ├── safety_and_limitations.md     # Decision-support framing, privacy, bias, generalization limits
│   ├── final_report/                 # LaTeX/Markdown sources of the final technical report
│   └── presentation/                 # Slides sources, demo script
├── deployment/
│   ├── docker/
│   │   ├── Dockerfile.api            # CPU/GPU inference image
│   │   ├── Dockerfile.train          # Training image (optional, for Azure ML jobs)
│   │   └── docker-compose.yml        # api + app + mlflow (local)
│   ├── azure/
│   │   ├── bicep_or_cli/             # Container Apps / App Service / ACI definitions (no secrets)
│   │   └── README.md                 # Step-by-step deployment; cost notes
│   └── ci/
│       └── github-actions/           # lint + tests + docker build workflows
├── scripts/                          # Thin CLIs that call src/bmts (download, index, split, preprocess, train, evaluate, infer)
└── dvc.yaml / .dvc/                  # Data + model version tracking (DVC, remote = Azure Blob)
```

### 1.1 Why this layout

- `src/bmts` mirrors the proposal's pipeline order (data → segmentation → quantitative → longitudinal → radiomics/prediction → uncertainty/explainability → pipeline/api). Each proposal module maps to exactly one package, so ownership in the plan maps to folders.
- `data/lockbox/` is separate from `data/splits/` so the held-out internal test set and external cohorts are physically and procedurally isolated (Milestone 5 leakage control).
- `nnunet/` is isolated because nnU-Net v2 owns its own preprocessing and folder conventions; we bridge to it instead of forcing it into `src/bmts`.
- `experiments/` holds *small* reproducibility artifacts per experiment ID; weights live in `models/` + MLflow, never in Git.

---

## 2. Code Organization Rules

1. **Library vs. scripts:** all logic lives in `src/bmts`; `scripts/` and `notebooks/` are thin callers. Nothing in a notebook is a deliverable until it is moved into `src/bmts` with a test.
2. **Config-driven:** every stage reads a YAML config from `configs/`; no hard-coded paths, seeds, or label values. Label integers are defined once in `src/bmts/common/constants.py` and `configs/data/label_harmonization.yaml`.
3. **Interfaces first (parallel work):** the following contracts are written in Milestone 1–2 so backend/UI/analysis can proceed with mock outputs while models train:
   - `SegmentationResult` (pydantic): `subject_id`, `timepoint`, `label_map_path`, `probability_map_path?`, `label_scheme = "brats_canonical"`, `model_version`, `spacing_mm`, `orientation`.
   - `QuantitativeReport` (pydantic): per-region volumes (mm³/cm³), max diameter, surface area, sphericity, compactness, boundary irregularity, tumor-to-edema ratio, units.
   - `LongitudinalReport` (pydantic): `verification_status` (verified/flagged/rejected + reasons), `interval_days?`, per-region absolute/percent change, rate per 30 days when dates exist, `trend` ∈ {decreased, stable, increased}, change-map path, registration QC metrics.
   - `UncertaintyReport` (pydantic): mean voxel entropy per region, uncertainty-map path, calibration summary, confidence flag.
4. **Medical-safety strings:** every user-facing output includes the fixed disclaimer constant `DECISION_SUPPORT_NOTICE` (research and clinical decision-support prototype, not an autonomous diagnosis). Tests assert its presence in API responses.
5. **Units:** volumes stored in mm³ and reported in cm³ (mL); distances in mm; time in days. Voxel spacing is read from the NIfTI header of the *processed* volume, never assumed.
6. **Type hints + docstrings** on all public functions; `ruff` + `black` enforced by pre-commit.

---

## 3. Dataset Organization

### 3.1 Cohort roles (fixed vocabulary)

| Role code | Meaning |
| --- | --- |
| `DEV_TRAIN` | Primary development cohort — training split |
| `DEV_VAL` | Primary development cohort — validation split (model selection, early stopping, tuning) |
| `DEV_TEST` | Primary development cohort — held-out internal test (lockbox; used once in M5) |
| `SUPP_TRAIN` | Supplementary cohort admitted to training only after harmonization + overlap check |
| `EXT_TEST` | External generalization cohort (different institution/scanner); never used for training or tuning |
| `LONG` | Longitudinal cohort (same-patient sequential studies) for the longitudinal module |
| `MOL` | Molecular cohort (MGMT labels) for the optional prediction module |
| `DEMO` | De-identified demo cases for the live demonstration |

A subject has exactly one role within the development cohort (patient-level), and a subject that appears in `EXT_TEST` or `LONG` must not appear in any `DEV_*` or `SUPP_*` role (enforced by `tests/data/test_leakage.py`).

### 3.2 Canonical subject and file naming

- Canonical subject ID: `<cohort>__<original_id>` (double underscore), e.g. `brats2021__BraTS2021_00495`, `rhuh__RHUH-0012`. The original ID is kept verbatim for traceability.
- Canonical timepoint: `tp00`, `tp01`, … in acquisition order (`tp00` = earliest available). Single-timepoint cohorts use `tp00`.
- Processed file names: `t1.nii.gz`, `t1ce.nii.gz`, `t2.nii.gz`, `flair.nii.gz`, `seg.nii.gz` (canonical labels), `seg_raw.nii.gz` (original labels retained).
- Canonical label scheme (`brats_canonical`): `0` background, `1` NCR/NET (non-enhancing tumor core / necrosis), `2` ED (peritumoral edema / FLAIR hyperintensity), `3` ET (enhancing tumor), `4` RC (resection cavity, post-treatment cohorts only). Evaluation regions: **WT** = {1,2,3}, **TC** = {1,3}, **ET** = {3}. RC handling is defined per experiment in `label_harmonization.yaml` (default: excluded from WT/TC/ET, retained as its own label for post-treatment analysis).

### 3.3 Manifests

- `data/manifests/unified_subject_index.csv` — one row per (subject, timepoint); schema in `docs/templates/DATASET_MANIFEST_TEMPLATE.csv`.
- `data/manifests/<cohort>_manifest.csv` — per-cohort raw file inventory with SHA-256 checksums.
- `data/splits/split_v<N>_<role>.csv` — patient-level split files; each split version is immutable once used by an experiment.
- `data/lockbox/README.md` — who may read the lockbox, when (M5 only), and the sign-off record.

### 3.4 Storage policy

- Raw and processed data live outside Git (`.gitignore`) and are versioned with **DVC** pointing to an Azure Blob Storage remote (or a shared external drive if cloud storage is unavailable). Manifests and DVC pointer files are committed.
- Estimated footprints must be recorded in the Dataset Access/Download Manifest (`M1-D02`). Plan for ≥ 250 GB of local scratch on the training machine.
- No dataset may be redistributed. Team members download from the official source using their own accepted data-use agreement.

---

## 4. Data Versioning (DVC)

- `dvc init` in the repo; remote `azure://<container>/dvc` configured via environment variables, not committed.
- Tracked artifacts: `data/processed/<cohort>`, `data/manifests/`, `data/splits/`, `models/checkpoints/<registered>`.
- Every experiment config records `dataset_version` = the DVC commit hash of `data/processed` and `split_version` = the split file name.
- `dvc.yaml` stages: `index → check_overlap → split → harmonize → preprocess → (nnunet_convert)`, so preprocessing is reproducible by `dvc repro`.

---

## 5. Model Organization

```text
models/
├── checkpoints/
│   └── EXP-0007_nnunet_3dfullres_fold0/
│       ├── best_val_dice.pt          # Selected on DEV_VAL only
│       ├── last.pt
│       └── checkpoint_meta.json      # exp id, git commit, dataset/split versions, epoch, val metrics
└── registry/
    └── seg-model/
        ├── v1/                       # "Selected Model v1" (end of M2)
        │   ├── model_card.md
        │   ├── manifest.json         # checksum, source experiment, config path, MLflow run id
        │   └── inference_config.yaml
        └── v2/                       # "Final Model" (M5, after frozen protocol)
```

- **Model versions** are immutable folders; a new selection creates a new version.
- **MLflow Model Registry** is the system of record: stage transitions `None → Staging (M2 selected v1) → Production (M5 final)`. The `registry/` folder mirrors the registry entry for offline reproducibility.
- The API loads models only by registry version pinned in `configs/api/api_settings.yaml`.

---

## 6. Experiment Organization

- **Experiment ID format:** `EXP-<4-digit>` (e.g. `EXP-0001`). Sub-runs (folds, seeds): `EXP-0001-f0`, `EXP-0001-s42`.
- Every experiment has: a frozen config in `configs/training/EXP-xxxx.yaml`, a folder in `experiments/EXP-xxxx_<slug>/`, an MLflow run tagged with the same ID, and a row in `docs/templates/EXPERIMENT_TRACKER_TEMPLATE.csv` (the CSV is the human-readable index; MLflow is the machine record).
- Mandatory fields per experiment (recorded automatically by the training script): experiment ID, git commit (dirty tree forbidden — the script refuses to start if `git status` is dirty unless `--allow-dirty` is set and logged), dataset version, split version, architecture, input modalities, preprocessing config hash, augmentation config hash, loss, optimizer, learning rate, scheduler, batch size, patch size, random seed, training duration, per-region validation metrics (Dice/IoU/Sensitivity/Precision/HD95 for WT/TC/ET), test metrics (only when protocol is frozen), checkpoint path, notes.
- **Test-set discipline:** the evaluation script refuses to run on `DEV_TEST` or `EXT_TEST` unless `--frozen-protocol docs/evaluation_protocol.md` is passed and the protocol file's hash matches the recorded frozen hash (M5 control).

---

## 7. Git Strategy

- **Repository:** the team's existing shared repository; nobody creates a new repo.
- **Default branch:** `main` (protected; PR + one review required).
- **Branch naming:** one branch per task, named after the task ID from the Notion task board, e.g. `M1-T05-subject-index`, `M2-T03-unet-baseline`.
- **Commit messages:** start with the task ID, e.g. `M2-T03: add 3D U-Net baseline`.
- **Pull requests:** template includes task ID, deliverable ID, test evidence, and a "no data / no secrets committed" checkbox. The deliverable owner's reviewer is listed in the plan (usually the milestone reviewer).
- **Tags:** `m1-approved`, `m2-approved`, … at each approval gate; `model-v1`, `model-final`; `submission-final`.
- **What is never committed:** `data/` (except tiny fixtures and manifests), `models/checkpoints/`, `mlruns/`, `.env`, Azure credentials, notebook outputs, files > 10 MB (pre-commit hook blocks them).
- **Git LFS:** used only for small demo assets (result-gallery PNGs, slide images, ≤ 50 MB total). Model weights and NIfTI volumes are *not* put in LFS (quota and reproducibility reasons); they go to DVC/Azure Blob.

---

## 8. Model Checkpoint Policy

1. Save `best_val_dice.pt` (selection metric = mean of WT/TC/ET Dice on `DEV_VAL`) and `last.pt`; keep at most 3 intermediate checkpoints per run (rolling).
2. Every checkpoint ships with `checkpoint_meta.json` (see §5) — a checkpoint without metadata is treated as invalid by the loader.
3. Checkpoints are uploaded to the DVC/Azure remote at the end of each run; local copies may be deleted after upload verification (`dvc status`).
4. Registered models (`v1`, final) are additionally exported to a deployment format (TorchScript or ONNX where supported) with a numerical-equivalence test against the PyTorch checkpoint on 3 validation cases.
5. Retention: all registered versions kept; unregistered experiment checkpoints may be pruned after the M2 approval gate, provided their metrics are preserved in MLflow and the tracker CSV.

---

## 9. Environment Management

- **Python 3.10 or 3.11**, **PyTorch** (CUDA build matched to the GPU driver), **MONAI**, **nnU-Net v2**, **NiBabel**, **SimpleITK**, **PyRadiomics**, **scikit-learn**, **XGBoost**, **MLflow**, **FastAPI**, **uvicorn**, **pydantic**, **DVC**.
- One Conda environment spec for training (`environment/environment.yml`) and a slim pip lock for the API image. Both are pinned with exact versions after the M2 baseline run succeeds; changes require a PR and a re-run of `tests/reproducibility`.
- GPU reality check: the team should record the actual GPU(s) available (local, Kaggle/Colab, Azure NC-series). Patch size, batch size and architecture choices in `configs/models/*.yaml` carry a `min_gpu_memory_gb` field, and Swin UNETR is gated on that field (see fallback strategy in the plan).
- Docker images pin the base image digest; `Dockerfile.api` is CPU-capable so the app can run on low-cost Azure services when GPU is unavailable (slower but functional for the demo).

---

## 10. Secrets and Credentials Policy

- **Never** commit Azure credentials, storage keys, MLflow tokens, or Kaggle/Synapse/TCIA credentials. `.env` is git-ignored; `.env.example` lists variable names only.
- Secrets are supplied by environment variables locally and by Azure Key Vault / Container Apps secrets in deployment.
- A pre-commit secret scanner (e.g. `detect-secrets` or `gitleaks`) runs on every commit; CI fails on any detected secret.
- Data-use agreements (BraTS/Synapse, TCIA collections) are personal; account credentials are never shared between members.

---

## 11. Reproducibility Strategy

| Element | Mechanism |
| --- | --- |
| Code | Git commit hash recorded in every experiment, model, and result file |
| Data | DVC hash of `data/processed` + immutable split files + SHA-256 per raw file in manifests |
| Configuration | Frozen YAML per experiment; config hash logged in MLflow |
| Randomness | Fixed seeds for Python/NumPy/PyTorch; `torch.backends.cudnn.deterministic=True` for the reproducibility smoke test (documented as slower); nondeterministic kernels listed in `docs/model_development.md` |
| Environment | Pinned Conda/pip locks; Docker image digest for API |
| Metrics | Metric implementations unit-tested against MONAI/MedPy reference values on synthetic masks |
| Model selection | Only `DEV_VAL` metrics may influence selection; the evaluation script enforces the frozen-protocol guard for test cohorts |
| Reproducibility test | `tests/reproducibility/test_smoke_train.py`: 2-case, 2-epoch run with fixed seed reproduces loss to a tolerance on the same hardware |
| Final package | `docs/reproducibility_package/` = configs + environment + split files + model manifest + inference instructions + commands to regenerate every table in the final report |

---

## 12. Naming Conventions Summary

| Item | Convention | Example |
| --- | --- | --- |
| Task | `M<milestone>-T<nn>` | `M2-T04` |
| Deliverable | `M<milestone>-D<nn>` | `M3-D07` |
| Experiment | `EXP-<nnnn>[-f<fold>][-s<seed>]` | `EXP-0012-f0` |
| Split version | `split_v<N>` | `split_v1` |
| Dataset version | DVC commit hash + tag `data-v<N>` | `data-v2` |
| Model version | `seg-model/v<N>` | `seg-model/v1` |
| Cohort code | lowercase, underscores | `brats2021`, `rhuh_gbm` |
| Canonical subject ID | `<cohort>__<original_id>` | `ucsf_pdgm__UCSF-PDGM-0123` |
| Timepoint | `tp<nn>` | `tp01` |
| Config files | `<purpose>_v<N>.yaml` | `harmonize_v1.yaml` |
| Result tables | `results/<area>/<deliverable-id>_<slug>.csv` | `results/segmentation/M2-D08_model_comparison.csv` |
| Figures | `results/<area>/fig_<slug>.png` | `results/longitudinal/fig_changemap_case03.png` |

---

## 13. Testing Layout (summary; details in the plan's testing strategy)

- `tests/unit` — metrics, label remapping, morphology math, change calculations, trend rules.
- `tests/data` — manifest schema, modality presence, shape/orientation/spacing, label-value validity, leakage check.
- `tests/integration` — synthetic 3-case pipeline: preprocess → infer (tiny model) → quantify → longitudinal.
- `tests/api` — endpoints, validation errors, missing modality, oversize upload, disclaimer presence.
- `tests/longitudinal` — subject-verification decision table, registration QC thresholds.
- `tests/reproducibility` — seed reproducibility smoke test; config/hash checks.
- CI (GitHub Actions): lint + unit + data-schema + api tests on every PR; integration tests nightly or before merging into `main`.

---

*Claude Recommended Addition — not in the proposal but required to execute it safely:* the lockbox mechanism (§3.1, §6), the frozen-protocol guard, the dirty-tree refusal in the training script, and the mandatory disclaimer constant are additions that operationalize the proposal's leakage-prevention and decision-support requirements. They do not expand the scientific scope.
