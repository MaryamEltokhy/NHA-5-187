# 3D U-Net Baseline (M2-T03, experiment EXP-0001)

The first segmentation model: it learns to draw the three tumor regions (TC, WT, ET) from the 4 MRI types, on the
BraTS 2021 training patients, and is scored on the validation patients. M2-T04 compares nnU-Net against it.

## What runs

| Part | File | Choice |
| --- | --- | --- |
| Experiment | `configs/training/EXP-0001.yaml` | 80 epochs × 200 steps, AdamW 3e-4 with 3 warm-up epochs then cosine decay, Dice + BCE loss, mixed precision, gradient clipping |
| Model | `configs/models/unet3d_baseline.yaml`, `src/bmts/segmentation/models/unet3d.py` | MONAI residual 3D U-Net, 5 levels (32→320 feature maps), instance norm, dropout 0.1 (reused for the confidence map in M3-T04) |
| Data | `configs/data/loader_v1.yaml` + overrides | 128³ patches, 2 per step from one patient (two thirds centred on tumor), flips/rotation/scaling/brightness/gamma; 150 patients kept in RAM, a quarter swapped after every epoch |
| Training loop | `src/bmts/segmentation/training/trainer.py` | Checkpoint to Drive after every epoch (`last.pt`), best model by validation mean Dice (`best.pt`), resume on rerun, clean stop at `--max-hours` |
| Scores | `src/bmts/segmentation/training/metrics.py` | Dice, IoU, sensitivity, precision, HD95 per region, with the same conventions as the team's `pipeline/metrics.py` (M2-T02) |
| Run it | `notebooks/04_train_unet_baseline.ipynb` (Colab T4) or `python scripts/train.py --exp configs/training/EXP-0001.yaml` | |

Prediction on a whole scan uses sliding windows of 128³ with 50 % overlap; a voxel belongs to a region when the
sigmoid output is above 0.5.

## What is logged in MLflow

Store: `BrainMRI_Data/mlflow.db` (SQLite on Drive), experiment `BrainMRI_3D_UNet`, run name `EXP-0001`.

| Kind | Content |
| --- | --- |
| Parameters | every setting of the experiment, model and data configs; number of patients and model parameters |
| Tags | task, git commit (and whether the code had uncommitted changes), split file + its SHA-256 + split version, config hash, device, where the checkpoints are, run state |
| Every epoch | `train_loss`, `lr`, `epoch_seconds`, `peak_gpu_memory_gb` |
| Every 5 epochs | `val_dice_TC/WT/ET`, `val_mean_dice` on the first 40 validation patients; `best_val_mean_dice` when it improves |
| At the end | `final_val_{dice,iou,sensitivity,precision,hd95}_{TC,WT,ET}` and `final_val_mean_dice` for the best checkpoint on **all** validation patients; `evaluation/` artifacts: summary JSON, per-patient CSV, resolved config |

A run that stops at `--max-hours` is marked *paused*; rerunning the same command continues the same run.

## Rules it follows

- **Never reads locked test patients:** the loader refuses `split: test` and `lockbox: True`; the final evaluation uses the validation split only.
- **Trains on BraTS 2021 only** (`datasets: [brats2021]`), so the other cohorts stay valid external tests.
- **Validation during training uses 40 patients** to keep each check short; the reported scores are the final evaluation on all validation patients.

## Checks

`tests/segmentation/test_training.py`: scoring conventions (both-empty, one-empty, partial overlap); a tiny end-to-end
run on synthetic scans that checks MLflow parameters/tags/metrics, the memory cache, checkpoint files, that the test
patient is never scored, resuming into the same run, and pausing at the time budget.
