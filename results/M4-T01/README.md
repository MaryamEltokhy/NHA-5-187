# M4-T01 results: seg-model v1 packaged for fast prediction

Made by `notebooks/09_m4_t01_export_model.ipynb` on 28 Oct 2026, on a Colab Tesla T4 GPU with torch 2.11 and
nnU-Net 2.8.1. The method is described in [`docs/DEPLOY_MODEL.md`](../../docs/DEPLOY_MODEL.md).

> Research prototype for decision support. These are validation numbers on a public dataset, not clinical performance.

## The export

| Check | Result |
| --- | --- |
| Checkpoint SHA-256 | `7dcb450b…f5e0`, the registered seg-model v1 (`models/registry/seg-model/v1/manifest.json`) |
| Traced network vs original on a test tile | max difference **0.0** |
| Export folder on Drive | `BrainMRI_Data/models/seg-model-v1-torchscript/` (model file 125 MB, SHA-256 `e5ca77e1…dfb9`) |

## Comparison on 20 validation patients

The 20 patients are the first 20 BraTS 2021 validation patients by ID. No locked test patients were used.

The reference ("original") is seg-model v1 run the way `nnUNetv2_predict` runs it: loaded from the 250 MB
checkpoint, with the 8 mirror flips. Times are for one scan, from the files on disk to the mask in the scan's grid.
They are medians over patients 2–20; patient 1 is left out because its time includes GPU warm-up.

| | Original (v1) | Export, `tta=True` | Export, `tta=False` (default) |
| --- | --- | --- | --- |
| Time per scan, median | 8.7 s | 8.6 s | **2.1 s** |
| of which the network | – | 7.5 s | 1.0 s |
| Loading the model, once | 4.9 s | 1.0 s | 1.0 s |
| Voxels different from v1, median (max) | – | **0 (0)** | 1,979 (11,204) |
| Dice against v1's mask, lowest of the 20 patients (TC / WT / ET) | – | 1.000 / 1.000 / 1.000 | 0.942 / 0.968 / 0.942 |
| Dice against v1's mask, mean (TC / WT / ET) | – | 1.000 / 1.000 / 1.000 | 0.987 / 0.989 / 0.983 |
| Dice against the expert, mean (TC / WT / ET) | 0.927 / 0.940 / 0.870 | 0.927 / 0.940 / 0.870 | 0.925 / 0.939 / 0.869 |
| Mean Dice against the expert | 0.9125 | 0.9125 | 0.9112 |

What this shows:

- **`tta=True` reproduces seg-model v1 exactly.** On all 20 patients, every voxel is the same as the original
  prediction. It is also about 4 s quicker to load, because the 250 MB training checkpoint is not read.
- **`tta=False` is about 4x faster per scan** (8x less network work). It agrees closely with v1: the median
  difference is about 2,000 voxels, or 2 cm³, mostly at the tumor boundary. Against the expert masks it is slightly
  lower on average: mean Dice 0.911 instead of 0.913, and per region by 0.0004 (WT) to 0.002 (TC). Per patient the
  change ranges from −0.039 to +0.022. The largest drop is TC in BraTS2021_00134, which is also the patient whose mask
  differs most (11,204 voxels).
- The remaining time per scan (about 1 s) is spent reading and preprocessing the scan and writing the mask back to
  its grid. It is the same in all three columns.

Choice for the app (M4-T02): use `tta=False` for interactive use, and `tta=True` when the result must equal the
validated seg-model v1. Both are one argument of `FastSegmenter.from_dir`.

## Files

| File | Content |
| --- | --- |
| `per_case.csv` | One row per patient: times, voxels different from v1, Dice against v1's mask and against the expert, for each setting |
| `summary.json` | The summary above, with the export and checkpoint SHA-256 values and the versions |
