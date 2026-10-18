# M3-T03: radiomics features per tumor region

`radiomics_features_val.csv` has shape, first-order and texture features of every tumor region for the 188 validation
patients, calculated with PyRadiomics from the nnU-Net predicted masks (M2-T04, the model in `models/registry/seg-model/v1/`).

- **Rows:** 564 = 188 patients x 3 regions. **Columns:** 293 features plus 4 id columns.
- **Regions** (they overlap): TC = labels 1 + 3, WT = labels 1 + 2 + 3, ET = label 3 (same definition as `bmts.common.constants.REGIONS`).
- **Images:** the cleaned scans from M1-T04 (RAS, 240 x 240 x 155, 1 mm voxels, z-score inside the brain), image types T1ce, FLAIR and T2.
- **Settings:** `configs/radiomics/pyradiomics_params.yaml`. Fixed bin width 0.25 (the images are z-scored, so the default
  width of 25 would put the whole tumor in one bin), no extra normalization and no resampling (already done by the cleaning step),
  3D features, original image only, all 7 feature classes.
- **Code:** `notebooks/NN_m3_t03_radiomics.ipynb` (self-contained, run on the real data in Colab).
  Install: `environment/requirements-radiomics.txt`.
- **Versions:** PyRadiomics 3.1.1.dev111+g8ed579383 (from GitHub, commit pinned in the requirements file), Python 3.13.16, Google Colab.
  The run took 66.9 minutes with 2 processes.

| Column | Meaning |
| --- | --- |
| `patient_id` | Patient id, for example BraTS2021_00032 |
| `region` | TC, WT or ET |
| `voxels` | Number of voxels in the region (1 voxel = 1 mm3, so 1000 voxels = 1 cm3) |
| `skip_reason` | Empty when the region was computed, otherwise why not. Empty for all 564 rows |
| `shape_*` (14) | Shape of the region (volume, surface area, sphericity, axis lengths, ...). From the mask only, so computed once per region |
| `t1ce_*`, `flair_*`, `t2_*` | For each image type: `firstorder` (18), `glcm` (24), `glrlm` (16), `glszm` (16), `gldm` (14), `ngtdm` (5) |

Example column names: `shape_Sphericity`, `t1ce_firstorder_Mean`, `flair_glcm_Contrast`, `t2_glszm_ZoneEntropy`.

**Checks:** the table has no duplicates, no missing values and no infinite values. The 188 patients are exactly
the validation patients of `split_v1.csv`, so no locked test patient is included. The voxel count of every region matches
`results/M3-T01/tumor_volumes_val.csv` for all 188 patients (same masks), and ET <= TC <= WT everywhere.

**Limits:**
- These are features of the model's predicted masks, not of the expert masks, so they carry the model's errors (mean Dice 0.908 on these patients).
- The model predicted enhancing tumor in 3 patients (BraTS2021_01483, 01509, 01535) who have none in the expert mask, so their ET features do not describe real tumor.
- Some regions are small: the smallest ET has 60 voxels (BraTS2021_01509), then 109 (00540) and 211 (01535). Texture features of regions this small are unreliable and should be used with care.
- In 5 patients (BraTS2021_00284, 00496, 00540, 01077, 01159) TC and ET are the same region, because the predicted mask has no NCR label.
- Only the 188 validation patients of BraTS 2021 are covered. The locked test patients are not included.
