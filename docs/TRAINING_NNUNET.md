# nnU-Net vs 3D U-Net (M2-T04)

nnU-Net is the better model: on the 188 validation patients it reaches a mean Dice of 0.908 against 0.765 for the
3D U-Net baseline (M2-T03, EXP-0001), and it scores higher on 97% of patients. It wins in all three regions (TC, WT, ET)
and its HD95 is about three times lower. nnU-Net is the Milestone 2 model.

## Setup

Both models share the same pieces, so the comparison is fair:

| Part | Choice |
| --- | --- |
| Data | cleaned BraTS 2021 patients (M1-T04): T1, T1ce, T2, FLAIR + expert segmentation |
| Split | `split_v1` (seed 42): 875 training, 188 validation; the 188 locked test patients are never read |
| Regions | TC (labels 1, 3), WT (labels 1, 2, 3), ET (label 3), from the team's `REGIONS` |
| Metrics | `region_scores` and `summarize` from `src/bmts/segmentation/training/metrics.py` (M2-T02 conventions): Dice, IoU, sensitivity, precision, HD95; NaN values are left out of averages and counted separately |

## What runs

| Part | Choice |
| --- | --- |
| Notebook | `notebooks/05_train_nnunet_compare.ipynb` (Colab T4) |
| Model | nnU-Net v2, `3d_fullres`, fold 0, plans chosen automatically by nnU-Net |
| Budget | `nnUNetTrainer_50epochs`: 50 epochs x 250 steps = 12,500 steps (baseline: 60 x 200 = 12,000 steps) |
| Split | custom `splits_final.json`: fold 0 = `split_v1` (train 875 / val 188) |
| Prediction | `checkpoint_best.pth` on the 188 validation patients, nnU-Net's default mirroring test-time augmentation |
| Scoring | team metrics on the predictions; baseline numbers from `final_val_per_case.csv` (M2-T03), averaged with the same `summarize` |

Checkpoints go to Drive after training epochs, and re-running the training cell resumes from the last checkpoint.

## Results (188 validation patients)

Higher is better for Dice, IoU, sensitivity and precision; lower is better for HD95 (mm).

| Region | Metric | 3D U-Net | nnU-Net |
| --- | --- | --- | --- |
| TC | Dice | 0.7173 | 0.9171 |
| TC | IoU | 0.5980 | 0.8674 |
| TC | Sensitivity | 0.7406 | 0.9050 |
| TC | Precision | 0.7923 | 0.9463 |
| TC | HD95 (mm) | 16.18 | 4.60 |
| WT | Dice | 0.8482 | 0.9346 |
| WT | IoU | 0.7528 | 0.8821 |
| WT | Sensitivity | 0.8421 | 0.9170 |
| WT | Precision | 0.8864 | 0.9597 |
| WT | HD95 (mm) | 18.84 | 5.62 |
| ET | Dice | 0.7282 | 0.8717 |
| ET | IoU | 0.6066 | 0.8016 |
| ET | Sensitivity | 0.8169 | 0.8769 |
| ET | Precision | 0.7145 | 0.8964 |
| ET | HD95 (mm) | 12.57 | 4.29 |

Mean Dice over TC, WT, ET: **0.765** (3D U-Net) vs **0.908** (nnU-Net). Paired Wilcoxon test on per-patient mean Dice:
p = 3.2e-31, nnU-Net better on 97% of patients. **Chosen model: nnU-Net.**

## Notes and limitations

- **Budget matched in steps, not in time.** An nnU-Net epoch took about 7 minutes, the baseline's about 3 minutes, on a T4.
- **Inference differs.** nnU-Net predicts with mirroring test-time augmentation; the baseline uses sliding windows without it.
- **Checkpoint choice.** nnU-Net's `checkpoint_best.pth` is chosen by its own pseudo Dice on the validation fold, the
  baseline's `best.pt` by Dice on the first 40 validation patients. Both selections look at validation data. In the nnU-Net
  log the last epoch was also the best one.
- **Empty predictions.** HD95 and precision averages skip cases where a region is empty in the prediction (NaN): 1 to 5
  such cases per metric for the baseline, 0 to 3 for nnU-Net. Dice is not affected.
- **Short training.** nnU-Net's default is 1000 epochs; its pseudo Dice was still improving after 50, so its score is
  probably not its ceiling.
- **Single fold.** One split and one run; no repeated runs or cross-validation.

## Files

| Item | Location |
| --- | --- |
| Small results | `results/M2-T04/` (`chosen_model.json`, `comparison_table.csv`, `comparison_per_case.csv`, `comparison_dice.png`, `scores_nnunet_per_case.csv`, `splits_final.json`) |
| nnU-Net checkpoints (Drive) | `BrainMRI_Data/runs/M2-T04_nnunet/nnUNet_results/Dataset501_BraTS2021/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres/fold_0/` |
| nnU-Net validation predictions (Drive) | `BrainMRI_Data/runs/M2-T04_nnunet/pred_nnunet_val/` |
| 3D U-Net baseline (Drive) | `BrainMRI_Data/runs/EXP-0001_unet3d_baseline/` (`best.pt`, `final_val_per_case.csv`, `final_val_metrics.json`) |

Model weights and predicted segmentations stay on Drive; only code, this report and the small result files are in the repo.
