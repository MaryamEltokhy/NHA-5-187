# Data Loader (M1-T05)

Turns the cleaned scans from M1-T04 into training batches: the 4 MRI types stacked as channels, plus the tumor mask
as 3 region channels. M2-T03 (baseline) and M2-T04 (nnU-Net comparison) build on it.

## Input

| What | Where (Colab) | Made by |
| --- | --- | --- |
| Cleaned scans | `/content/processed/<dataset>/<patient>/<timepoint>/{t1,t1ce,t2,flair,seg}.nii.gz` | `BrainMRI_Data/pipeline/standardize.py` (M1-T04): RAS, 240×240×155, z-scored per MRI type, canonical labels |
| Patient split | `BrainMRI_Data/qc/split_v1.csv` (`dataset_source`, `canonical_subject_id`, `split`, `lockbox`) | M1-T04 |
| Settings | `configs/data/loader_v1.yaml` | this task |

## Output

| Loader | Each batch |
| --- | --- |
| Training | `image` (B, 4, 128, 128, 128) float32 in the order T1, T1ce, T2, FLAIR; `label` (B, 3, 128, 128, 128) binary channels **TC, WT, ET**. Random patches, two thirds centred on tumor; flips, small rotation/scaling, brightness and gamma changes. |
| Validation | Whole volume, batch size 1: `image` (1, 4, 240, 240, 155), `label` (1, 3, 240, 240, 155). Use sliding-window inference. |

Regions follow the canonical labels (1 NCR, 2 ED, 3 ET, 4 RC): TC = {1, 3}, WT = {1, 2, 3}, ET = {3}. The resection cavity (4, MU-Glioma-Post only) is in no region.

## Use

```python
from bmts.data.datasets.brats import build_loaders, load_config

cfg = load_config("configs/data/loader_v1.yaml")          # or override: load_config(path, roi_size=[96, 96, 96])
train_loader, val_loader = build_loaders(cfg)
for batch in train_loader:
    image, label = batch["image"], batch["label"]
```

- **Locked test patients are refused.** Asking for `split: test`, or for any patient with `lockbox: True`, raises `LockboxError`. Only the final evaluation (M5-T02) may pass `allow_lockbox=True` to `list_cases`.
- **Training uses BraTS 2021 only** (`datasets: [brats2021]`), following rule 4 of `docs/dataset_research.md` §8.2, so the other cohorts stay valid external tests.
- **Only the patients you clean are needed.** With `on_missing: error` the loader stops if a listed patient has no cleaned files; use `on_missing: skip` while only part of the data is cleaned.
- **Small GPUs:** use `roi_size: [96, 96, 96]` and `batch_size: 1` below 8 GB.
- **Keeping patients in RAM (M2-T03):** `smart_cache_num: N` keeps N patients in memory (about 30 MB each: cropped to the brain, image stored as float16, mask as uint8) and swaps `smart_cache_replace_rate` of them after every epoch; `sampling.samples_per_case` draws several patches per loaded patient. On Windows/macOS the cache loads in the main process (MONAI limitation); on Colab (Linux) it uses the worker processes.

## Checks

- `tests/data/test_loader.py` (7 tests, synthetic scans in the cleaned layout): split filtering, lockbox refusal, missing files, region channels, batch shapes and values. Run with `pytest tests/data`.
- `scripts/smoke_train.py` trains a tiny 3D U-Net for 2 epochs and writes a JSON report; it stops on the first batch with a wrong shape, non-binary label or NaN.
- `notebooks/03_dataloader_smoke_test.ipynb` runs that smoke test in Colab on real BraTS 2021 patients (cleaned with Yasmin's pipeline). It passes when it prints `SMOKE TEST PASSED`.

Tested on 28 Sep 2026 (CPU, torch 2.14, MONAI 1.6.1): all 7 tests pass. The smoke test also passes on 6 real glioma patients from OpenNeuro ds007045, put through Yasmin's `standardize_file`, with 96³ and 128³ patches.

**Colab, real BraTS 2021 (28 Sep 2026, Tesla T4):** 16 training + 4 validation patients cleaned with `standardize.py`, 2 epochs, 128³ patches, batch 2: passed. About 33 s per epoch of 8 batches plus 7 s validation; peak GPU memory 0.78 GB with the tiny test model (the real baseline will need more). Report: `BrainMRI_Data/runs/M1-T05_smoke/smoke_report.json`.
