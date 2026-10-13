# M3-T01: tumor volume per region

`tumor_volumes_val.csv` has the tumor volume of every region for the 188 validation patients, calculated from the
nnU-Net predicted masks (M2-T04, the model in `models/registry/seg-model/v1/`).

- **How:** volume = number of voxels in the region x volume of one voxel, in cm3 (1 cm3 = 1000 mm3).
  The voxel size is read from each file's header. Here every voxel is 1 x 1 x 1 mm, so 1000 voxels = 1 cm3.
- **Regions** (they overlap, ET is inside TC and TC is inside WT): TC = labels 1 + 3, WT = labels 1 + 2 + 3, ET = label 3.
- **Code:** `src/bmts/quantitative/volumes.py`. Tests: `tests/quantitative/test_volumes.py`.
  Run on the real data in `notebooks/06_m3_t01_tumor_volume.ipynb`.

| Column | Meaning |
| --- | --- |
| `case` | Patient id, for example BraTS2021_00032 |
| `voxel_mm3` | Volume of one voxel in mm3 (1.0 for all 188 cases) |
| `TC_voxels`, `WT_voxels`, `ET_voxels` | Number of voxels in each region |
| `TC_cm3`, `WT_cm3`, `ET_cm3` | Volume of each region in cm3 |

Summary over the 188 patients (cm3): WT 5.66 to 266.06 (mean 96.17), TC 0.11 to 116.39 (mean 36.07),
ET 0.06 to 115.84 (mean 21.78).

**Checks:** the 5 tests pass. On all 188 cases, ET <= TC <= WT. On 3 cases (BraTS2021_00032, 00059, 00077) the numbers
match an independent voxel count done without the helper functions.

**Limits:**
- These are volumes of the model's predicted masks, not of the expert masks, so they carry the model's errors
  (mean Dice 0.908 on these patients).
- The model predicted enhancing tumor in 3 patients (BraTS2021_01483, 01509, 01535) who have none in the expert mask,
  so their ET volume is not real tumor.
- Only the 188 validation patients of BraTS 2021 are covered. The locked test patients are not included.
