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

| Setting | Network passes per tile | Mask |
| --- | --- | --- |
| `tta=True` (`--tta`) | 8 (all mirror flips, averaged) | Computed exactly as seg-model v1 (`nnUNetv2_predict` default) |
| `tta=False` (default) | 1 | Faster; may differ from v1 on a few voxels at the tumor boundary |

The measured time per scan and the difference from v1 are written to `BrainMRI_Data/runs/M4-T01_export/summary.json`.

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
masks, and the time per scan. Locked test patients are never loaded.

## Notes

- On the GPU, nnU-Net runs the network in mixed precision (`torch.autocast`). The original model does too. Any
  remaining difference shows in the notebook's voxel count.
- PyTorch 2.14 marks TorchScript as deprecated (it still works). If a later version removes it, only
  `bmts/deploy/export.py` has to change, for example to `torch.export`. The segmenter only needs a module that maps a
  1 × 4 × 128³ tile to 4 logit channels.
