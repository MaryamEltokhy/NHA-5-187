# Analysis pipeline (M3-T05)

One command takes all scans of one patient and gives back every Milestone 3 result.

> Research prototype for decision support. The numbers describe the model's masks; they are not a diagnosis.

```bash
# a patient from the team data (checked against qc/split_v1.csv first)
python scripts/analyze.py --dataset rhuh --subject RHUH-0002 --out out/RHUH-0002
# any scan folders, in time order (each: t1, t1ce, t2, flair .nii.gz, optionally seg.nii.gz = expert mask)
python scripts/analyze.py --scan exam1 --scan exam2 --patient P1 --out out/P1 [--raw]
```

In Colab, `notebooks/08_m3_t05_analysis_pipeline.ipynb` does the same with a few settings and shows the results.

## Steps (all reuse the team's tested code)

| Step | Code | Task |
| --- | --- | --- |
| Refuse locked test patients and external cohorts | `bmts.data.splits.check_subject` | M1-T04 rules |
| Cleaning, only with `--raw`: RAS, 240 x 240 x 155, z-score inside the brain | `bmts.data.cleaning.standardize` | M1-T04 |
| Mask + confidence map: seg-model v1 run once per mirror flip, mask identical to `nnUNetv2_predict` | `bmts.uncertainty.nnunet_tta.predict_with_confidence` | M2, M3-T04 |
| Volumes of the regions TC, WT, ET and of each label: NCR (1), ED (2), ET (3), RC (4, resection cavity, never tumor) | `bmts.quantitative.volumes.all_volumes` | M3-T01 |
| Radiomics on T1ce, FLAIR, T2 with `configs/radiomics/pyradiomics_params.yaml`, with a `quality_note` per region | `bmts.radiomics.features.extract_case` | M3-T03 |
| Change between every pair of scans of the patient for TC, WT, ET, NCR and ED, from the predicted masks and, if every scan has one, the expert masks | `bmts.quantitative.longitudinal.change_rows` | M3-T02 |
| Dice against the expert mask, when given (checking only) | `bmts.pipeline.analyze` | M2-T02 conventions |

**Raw input:** `--raw` runs the M1-T04 cleaning step first. That step does not skull-strip, co-register or resample,
so raw scans must already be skull-stripped, co-registered and 1 mm, as in BraTS. For raw BraTS 2021 expert masks
(where ET is label 4), add `--dataset brats2021` so the cleaning step maps ET to label 3.

**Percent change:** only given when the earlier volume is at least 1 cm3 (`--min-baseline-cm3`); the cm3 change is
always given.

## Outputs

| File | Content |
| --- | --- |
| `<scan>/<scan>_seg.nii.gz` | Predicted mask (0 background, 1 NCR, 2 ED, 3 ET), in the input scan's grid |
| `<scan>/<scan>_uncertainty.nii.gz`, `_disagreement.nii.gz` | Confidence maps, 0..1 (`docs/UNCERTAINTY.md`) |
| `volumes.csv` | One row per scan: cm3 and voxels of TC, WT, ET and of the single labels NCR, ED, RC (TC = NCR + ET, WT = NCR + ED + ET), run time |
| `uncertainty.csv` | One row per scan: uncertain volume and mean uncertainty per region (`maps.case_summary`) |
| `radiomics.csv` | One row per scan and region (columns as in `results/M3-T03`) |
| `changes.csv` | Patients with 2+ scans: change per pair for TC, WT, ET, NCR and ED (column `region`); `source` = prediction or expert; `same_grid` |
| `metrics.csv` | Scans with an expert mask: Dice per region and the expert volumes (`<name>_cm3_expert`, regions and single labels) |
| `run_summary.json` | Scans, model (from the registry manifest), settings, repo commit, Python version, run time |

## Checks

`tests/pipeline/test_analyze.py` runs the whole pipeline with a stand-in model and radiomics extractor. It checks that:
- a 3-scan patient gets every output, and the volumes nest (ET ≤ TC ≤ WT);
- the changes cover all 3 pairs, from both sources;
- a locked test patient and an external-cohort scan are refused before anything is read;
- raw scans are cleaned to the M1-T04 grid first.

The real model runs in Colab (T4 GPU), with the notebook above.

## History

- **First version:** Maryam's notebook `08_unified_brain_mri_pipeline.ipynb` (commit 19c00a4).
- **The Milestone 3 review moved its steps into the package so they call the tested code.** It also fixed several
  problems:
  - The confidence map is now M3-T04's, not a separate noise-based method.
  - Radiomics uses the team settings file. The earlier run fell back to PyRadiomics defaults (bin width 25).
  - Every input is checked against the split before it is read.
  - The pipeline no longer edits repo or nnU-Net files at run time.
  - One run gives all five outputs.
