"""One call from a scan to its tumor mask, with the exported seg-model v1 (M4-T01).

    seg = FastSegmenter.from_dir("export_dir").segment_to_file(image_files, "out/case_seg.nii.gz")

The network is the TorchScript file written by bmts.deploy.export; everything around it is nnU-Net's own code
(preprocessing, sliding window with Gaussian weighting, resampling back to the scan's grid), so the mask is computed
the same way as with `nnUNetv2_predict`.

Speed setting:
- tta=True: the 8 mirror flips are averaged, as `nnUNetv2_predict` does by default. Same mask as seg-model v1.
- tta=False (default): one pass, about 8x less network work. The mask can differ from v1 on a few boundary voxels;
  docs/DEPLOY_MODEL.md gives the measured difference.

Input: T1, T1ce, T2, FLAIR in that order, cleaned as in M1-T04 (the model was trained on cleaned BraTS scans).
Output: labels 0 background, 1 NCR, 2 ED, 3 ET in the input scan's grid; arrays in SimpleITK order (z, y, x).
Research prototype for decision support, not a diagnostic device.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from bmts.deploy.export import MANIFEST_FILE


class FastSegmenter:
    def __init__(self, network, plans, dataset_json, manifest, device="cuda", tta=False, tile_step_size=0.5):
        import torch
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

        device = torch.device(device)
        axes = manifest.get("mirror_axes")
        if tta and not axes:
            raise ValueError("this model has no mirror axes; use tta=False")
        self.tta, self.manifest = tta, manifest
        self.predictor = nnUNetPredictor(tile_step_size=tile_step_size, use_gaussian=True, use_mirroring=tta,
                                         perform_everything_on_device=device.type == "cuda", device=device,
                                         verbose=False, verbose_preprocessing=False, allow_tqdm=False)
        plans_manager = PlansManager(plans)
        self.predictor.manual_initialization(network, plans_manager, plans_manager.get_configuration(
            manifest["configuration"]), None, dataset_json, manifest["source"]["trainer"],
            tuple(axes) if axes else None)
        if self.predictor.label_manager.has_regions:
            raise ValueError("region-based models are not supported")

    @classmethod
    def from_dir(cls, export_dir, device="cuda", tta=False, tile_step_size=0.5, check_sha=True):
        """Load an export folder (bmts.deploy.export). check_sha verifies the model file against its manifest."""
        import torch

        from bmts.deploy.export import sha256

        export_dir = Path(export_dir)
        manifest = json.loads((export_dir / MANIFEST_FILE).read_text())
        model_path = export_dir / manifest["model_file"]
        if check_sha and sha256(model_path) != manifest["model_sha256"]:
            raise ValueError(f"{model_path} does not match the SHA-256 in {MANIFEST_FILE}")
        network = torch.jit.load(str(model_path), map_location=torch.device(device))
        plans = json.loads((export_dir / "plans.json").read_text())
        dataset_json = json.loads((export_dir / "dataset.json").read_text())
        return cls(network, plans, dataset_json, manifest, device=device, tta=tta, tile_step_size=tile_step_size)

    def logits(self, data):
        """Preprocessed case (C, z, y, x) -> logits in network space, on the CPU."""
        import torch

        return self.predictor.predict_sliding_window_return_logits(
            torch.from_numpy(np.asarray(data, dtype=np.float32))).to("cpu")

    def segment(self, image_files):
        """Scan files (T1, T1ce, T2, FLAIR) -> (mask (z, y, x) uint8, nnU-Net properties, timings in seconds)."""
        from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

        from bmts.uncertainty.nnunet_tta import preprocess

        p = self.predictor
        t0 = time.perf_counter()
        data, properties = preprocess(p, image_files)
        t1 = time.perf_counter()
        logits = self.logits(data)
        t2 = time.perf_counter()
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, p.plans_manager, p.configuration_manager, p.label_manager, properties, return_probabilities=False)
        t3 = time.perf_counter()
        timings = {"preprocess_s": t1 - t0, "network_s": t2 - t1, "export_s": t3 - t2, "total_s": t3 - t0}
        return np.asarray(seg, dtype=np.uint8), properties, timings

    def segment_to_file(self, image_files, out_path):
        """Like segment(), and also saves the mask as NIfTI in the scan's geometry. Returns (mask, timings)."""
        from bmts.uncertainty.nnunet_tta import write_like

        seg, properties, timings = self.segment(image_files)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        write_like(seg, out_path, properties)
        return seg, timings
