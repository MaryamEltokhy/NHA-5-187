# M2-T04 results: nnU-Net vs 3D U-Net

Small result files of the comparison (details: `docs/TRAINING_NNUNET.md`). Chosen model: **nnU-Net** (mean Dice 0.908 vs 0.765).

| File | Content |
| --- | --- |
| `chosen_model.json` | chosen model, selection metric, mean Dice of both, where the weights are |
| `comparison_table.csv` | mean of every metric per region, both models |
| `comparison_dice.png` | Dice boxplots per region |
| `scores_nnunet_per_case.csv` | nnU-Net per-patient scores |
| `splits_final.json` | train/val patient IDs used by nnU-Net (= `split_v1`) |

Model weights (`checkpoint_best.pth`) and predicted segmentations are not stored in git; they stay on Drive.
