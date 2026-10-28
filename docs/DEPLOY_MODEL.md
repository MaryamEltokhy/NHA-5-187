# seg-model v1 packaged for fast prediction (M4-T01)

One function takes a scan and returns the tumor mask. It is the model the M4 app (M4-T02) will call.

> Research prototype for decision support, not a diagnostic device.

```python
from bmts.deploy.segmenter import FastSegmenter

segmenter = FastSegmenter.from_dir("BrainMRI_Data/models/seg-model-v1-torchscript", device="cuda")   # load once
seg, timings = segmenter.segment_to_file([t1, t1ce, t2, flair], "case_seg.nii.gz")                   # per scan
```

```bash
python scripts/export_model.py --out <export folder>                           # once, in Colab
python scripts/segment.py --export-dir <export folder> --scan scans/exam1 --out out/ [--tta] [--device cpu]
```

`notebooks/09_m4_t01_export_model.ipynb` runs the export in Colab and compares the result with the original model.

## What the export contains

`scripts/export_model.py` checks the checkpoint against the SHA-256 in `models/registry/seg-model/v1/manifest.json`.
It then traces the network once with TorchScript (`torch.jit.trace` on one 1 × 4 × 128 × 128 × 128 tile) and writes:

| File | Content |
| --- | --- |
| `seg_model.torchscript.pt` | The network with its weights |
| `plans.json`, `dataset.json` | nnU-Net's description of the preprocessing, copied from the trained model folder |
| `deploy_manifest.json` | Source checkpoint and its SHA-256, the export file's SHA-256, the configuration, patch size and mirror axes, the traced-vs-original check, and the torch / nnU-Net versions |

At run time the segmenter needs neither the 250 MB training checkpoint nor the nnU-Net results folder or trainer class.
It checks the model file against `deploy_manifest.json` before loading it.

## How a scan is segmented

The traced network is the only new part. Everything around it is nnU-Net 2.8.1's own code, so the steps are the same
as `nnUNetv2_predict`:

1. Preprocessing: crop to the brain, then z-score inside the brain. The scans are already 1 mm, so there is no resampling.
2. Sliding window: 128³ tiles with 50 % overlap and Gaussian weighting.
3. Back to the scan's grid: the labels are 0 background, 1 NCR, 2 ED, 3 ET.

The input must be T1, T1ce, T2 and FLAIR, cleaned as in M1-T04. The model was trained on cleaned BraTS scans.

Measured on a Colab T4 GPU with 20 validation patients ([`results/M4-T01`](../results/M4-T01/README.md)):

| Setting | Network passes per tile | Time per scan (median) | Mask compared with seg-model v1 | Mean Dice vs expert (TC / WT / ET) |
| --- | --- | --- | --- | --- |
| Original v1 (`nnUNetv2_predict`, from the checkpoint) | 8 | 8.7 s | – | 0.927 / 0.940 / 0.870 |
| `tta=True` (`--tta`) | 8 (all mirror flips, averaged) | 8.6 s | **Identical**: 0 voxels differ on all 20 patients | 0.927 / 0.940 / 0.870 |
| `tta=False` (default) | 1 | **2.1 s** | Median 1,979 voxels differ (about 2 cm³, at the boundary); lowest Dice to v1 0.94 | 0.925 / 0.939 / 0.869 |

Loading the exported model takes 1.0 s, against 4.9 s for the 250 MB checkpoint. For the app, use `tta=False` for
interactive use and `tta=True` when the result must equal the validated seg-model v1.

## Checks

`tests/deploy/test_deploy.py` builds a trained-model folder from seg-model v1's own plans
(`models/registry/seg-model/v1/nnunet/`) with random weights, exports it and checks that:

- the export folder is complete and its SHA-256 values are right;
- the exported model gives **the same mask** (and logits within 1e-4) as nnU-Net's own predictor, with and without
  the mirror flips;
- the saved mask keeps the scan's grid and orientation;
- a checkpoint with the wrong SHA-256, or a model file changed after export, is refused;
- the full-size v1 network (32 to 320 features) traces with a difference under 1e-4.

The tests run on a CPU with a smaller network. On the GPU, the notebook compares the real model with the original on
validation patients: it counts the voxels that differ, the Dice between the two masks, the Dice against the expert
masks, and the time per scan. Locked test patients are never loaded. The 28 Oct 2026 run is in
[`results/M4-T01`](../results/M4-T01/README.md).

## Notes

- On the GPU, nnU-Net runs the network in mixed precision (`torch.autocast`), for the original and the exported
  model alike. The identical masks with `tta=True` show the exported model goes through the same computation.
- PyTorch 2.14 marks TorchScript as deprecated (it still works). If a later version removes it, only
  `bmts/deploy/export.py` has to change, for example to `torch.export`. The segmenter only needs a module that maps a
  1 × 4 × 128³ tile to 4 logit channels.
