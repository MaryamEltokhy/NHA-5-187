# M3-T02: tumor volume change between scans of the same patient

`tumor_change_long.csv` gives, for 27 RHUH-GBM patients, how the volume of each tumor region changed between their three
scans. Every number is given twice: from the **expert masks** (the reference) and from **seg-model v1** (nnU-Net)
predictions on the same scans.

> Research prototype for decision support. These are differences between labelled volumes on two scans, not a
> treatment-response assessment and not a diagnosis.

## How

- **Volumes** per region: voxels x voxel volume, in cm3. Regions are the team's `REGIONS`: TC = labels 1 + 3,
  WT = 1 + 2 + 3, ET = 3. Code: `src/bmts/quantitative/volumes.py` (M3-T01).
- **Change** between an earlier scan A and a later scan B, per region: `change_cm3 = V_B - V_A` and
  `change_pct = (V_B - V_A) / V_A x 100`.
  - The percentage is only given when V_A is at least **1 cm3**; otherwise it is empty and only the cm3 change counts.
  - The first version used 0.1 cm3, which turned a few voxels of difference into changes of several thousand percent.
  - Code: `src/bmts/quantitative/longitudinal.py` (`percent_change`, `change_rows`), tested in
    `tests/quantitative/test_longitudinal.py`.
- **Lining the scans up:** all cleaned scans (M1-T04) of a patient share the same 240 x 240 x 155, 1 mm grid, and a
  volume does not depend on where the tumor sits, so no registration is needed for volume change.
  `same_grid()` checks the grid before scans are compared. A voxel-by-voxel change map would need registration; it
  is not part of this task.
- **Scans:** RHUH-GBM timepoints 0, 1, 2, named by the dataset as pre-op, early post-op (within 72 h after surgery)
  and recurrence.
- **Patients:** the 27 RHUH patients of the training and validation split with all three cleaned scans. The other
  7 development patients (RHUH-0001, 0010, 0014, 0018, 0024, 0025, 0029) have incomplete cleaned folders; RHUH-0014
  and 0025 lack `t1ce.nii.gz`. No locked test patient is included.
- **Run:** `notebooks/07_m3_t02_scan_comparison.ipynb` (Colab). The same computation for any patient, from scans to
  table, is now part of the analysis pipeline: `scripts/analyze.py --dataset rhuh --subject RHUH-0002` (M3-T05).

## Columns of `tumor_change_long.csv`

| Column | Meaning |
| --- | --- |
| `patient`, `pair`, `tp_a`, `tp_b`, `region` | Patient, the two scans compared (earlier, later) and the region |
| `vol_a_cm3_*`, `vol_b_cm3_*` | Volume on the earlier and the later scan; `_expert` or `_nnunet` |
| `change_cm3_*`, `change_pct_*` | Change in cm3 and in percent (empty when the earlier volume is below 1 cm3) |
| `pct_diff_points` | nnU-Net percentage minus expert percentage, in percentage points |
| `days_preop_to_surgery` | Pre-op -> early post-op only: days from the earliest imaging to surgery, from the dataset's clinical file. The post-op scan is up to 72 h after surgery, so the real gap is this number to 3 days more. The clinical file has no recurrence-scan date, so the other pairs have no interval |

## Results

**Model vs expert volume on each scan** (27 patients). r is the correlation over patients.

| Scan | Region | r | Expert median (cm3) | nnU-Net median (cm3) | Median absolute error (cm3) |
| --- | --- | --- | --- | --- | --- |
| pre-op | WT | 0.996 | 99.48 | 105.83 | 1.69 |
| pre-op | TC | 0.999 | 31.18 | 30.08 | 0.86 |
| pre-op | ET | 0.997 | 17.39 | 16.37 | 0.78 |
| early post-op | WT | 0.797 | 24.99 | 84.15 | 57.46 |
| early post-op | TC | 0.032 | 0.00 | 32.19 | 32.19 |
| early post-op | ET | 0.359 | 0.00 | 2.13 | 2.13 |
| recurrence | WT | 0.888 | 44.13 | 83.06 | 11.00 |
| recurrence | TC | 0.578 | 9.66 | 24.80 | 1.98 |
| recurrence | ET | 0.541 | 8.74 | 11.08 | 1.49 |

- **Pre-op scans** (the kind the model was trained on): model and expert volumes agree closely.
- **Early post-op scans:** the model is not usable. It labels the surgical area as tumor core (median 32 cm3 where
  the expert marks none).
- **Recurrence scans:** in between.

**Change per pair** (median over the 27 patients; "same direction" is the share of patients where the model's change
has the same sign as the expert's):

| Pair | Region | Expert change (cm3) | nnU-Net change (cm3) | Same direction | Expert change (%) | nnU-Net change (%) | Patients with a % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pre-op -> early post-op | WT | -68.14 | -18.70 | 74 % | -73.9 | -16.0 | 27 |
| pre-op -> early post-op | TC | -31.18 | +2.39 | 44 % | -100.0 | +7.4 | 27 |
| pre-op -> early post-op | ET | -17.28 | -14.34 | 93 % | -100.0 | -85.1 | 27 |
| early post-op -> recurrence | WT | +36.32 | +5.43 | 70 % | +145.6 | +6.6 | 27 |
| early post-op -> recurrence | TC | +9.66 | -5.85 | 44 % | (1 patient) | | 1 |
| early post-op -> recurrence | ET | +8.74 | +6.25 | 78 % | (1 patient) | | 1 |
| pre-op -> recurrence | WT | -18.16 | -5.91 | 89 % | -23.4 | -6.6 | 27 |
| pre-op -> recurrence | TC | -8.73 | +7.06 | 81 % | -55.1 | +22.3 | 27 |
| pre-op -> recurrence | ET | -0.95 | +2.74 | 93 % | -9.8 | +22.2 | 27 |

**Reading the change table:**
- **Use the expert-based numbers as the reference.** The model's changes are only close where both scans are of a
  kind it handles: the pre-op -> recurrence pair, where its direction matches in 81–93 % of patients.
- **Avoid the model's numbers for any pair with the early post-op scan.**
- **Percentages need a meaningful earlier volume.** After surgery, TC and ET are usually 0 cm3 in the expert masks, so
  for early post-op -> recurrence only 1 patient has a percentage; use the cm3 change.
- **The recurrence scan is not always bigger.** In 16 of 27 patients the expert whole-tumor volume on the recurrence
  scan is smaller than before surgery.

**Check against the dataset's own numbers (narrow).** The RHUH-GBM clinical file gives two volumes per patient, only
for the pre-op and early post-op scans. The notebook compares each with the matching part of the expert masks:

| Clinical file column | Compared with | Largest difference over 27 patients |
| --- | --- | --- |
| contrast-enhancing tumor | TC = labels 1 + 3 (necrosis + enhancing tumor) | 5.40 cm3 pre-op (one patient), 0.10 cm3 post-op |
| T2/FLAIR abnormality | WT - TC = label 2 | 0.01 cm3 pre-op and post-op |

The near-exact T2/FLAIR match shows the masks are read with the dataset's own label meaning. The check does not cover
ET (label 3) or necrosis (label 1) on their own, or the recurrence scan, because the clinical file has no such
volumes. The one 5.40 cm3 pre-op difference was not investigated further.

## Limits

- The model was trained on pre-treatment BraTS 2021 scans only. Post-operative and recurrence scans are outside what
  it has seen (surgical cavity, blood products).
- RHUH-GBM label 2 is "peritumoral signal" (edema plus non-enhancing tumor), so TC (labels 1 + 3) leaves out
  non-enhancing tumor. WT and ET are the most comparable regions.
- Only the pre-op -> early post-op pair has an interval, and it is a lower bound. No dates are used.
- 27 patients is a small sample. The medians above describe this sample only.
