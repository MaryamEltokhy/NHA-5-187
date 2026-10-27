"""Export seg-model v1 (nnU-Net v2) to TorchScript (M4-T01).

The trained network is traced once on a patch-sized input and saved with the two nnU-Net files that describe the
preprocessing (plans.json, dataset.json). The export folder is all the segmenter needs at run time: no checkpoint
(.pth), no nnU-Net results folder and no trainer class.

    export_dir/
        seg_model.torchscript.pt   the network with its weights
        plans.json, dataset.json   copied from the trained model folder
        deploy_manifest.json       source checkpoint and its SHA-256, export check, versions
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

MODEL_FILE = "seg_model.torchscript.pt"
MANIFEST_FILE = "deploy_manifest.json"
EXPORT_TOLERANCE = 1e-3          # max |traced - original| logit difference accepted on the check input


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _version(package):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(package)
    except PackageNotFoundError:
        return None


def export_model(model_dir, out_dir, fold=0, checkpoint="checkpoint_best.pth", expected_sha256=None, seed=0):
    """Trace the network of a trained nnU-Net folder and write the export folder. Returns the manifest.

    expected_sha256: when given (e.g. from models/registry/seg-model/v1/manifest.json), the checkpoint must match it.
    """
    import torch

    from bmts.uncertainty.nnunet_tta import load_predictor

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    ckpt_path = model_dir / f"fold_{fold}" / checkpoint
    ckpt_sha = sha256(ckpt_path)
    if expected_sha256 is not None and ckpt_sha != expected_sha256:
        raise ValueError(f"{ckpt_path} has SHA-256 {ckpt_sha}, expected {expected_sha256}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    predictor = load_predictor(model_dir, fold=fold, checkpoint=checkpoint, device="cpu")
    network = predictor.network
    network.load_state_dict(predictor.list_of_parameters[0])
    network.eval()

    patch = list(predictor.configuration_manager.patch_size)
    channels = len(predictor.dataset_json["channel_names"])
    generator = torch.Generator().manual_seed(seed)
    example = torch.randn(1, channels, *patch, generator=generator)
    with torch.no_grad():
        traced = torch.jit.trace(network, example)
        check = torch.randn(1, channels, *patch, generator=generator)
        diff = float((traced(check) - network(check)).abs().max())
    if diff > EXPORT_TOLERANCE:
        raise RuntimeError(f"traced network differs from the original by {diff:.2e} (> {EXPORT_TOLERANCE})")

    out_dir.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_dir / MODEL_FILE))
    for name in ("plans.json", "dataset.json"):
        shutil.copyfile(model_dir / name, out_dir / name)
    axes = ckpt.get("inference_allowed_mirroring_axes")
    manifest = {
        "format": "TorchScript (torch.jit.trace)",
        "model_file": MODEL_FILE,
        "model_sha256": sha256(out_dir / MODEL_FILE),
        "source": {"model_dir": str(model_dir), "fold": fold, "checkpoint": checkpoint, "checkpoint_sha256": ckpt_sha,
                   "trainer": ckpt["trainer_name"]},
        "configuration": ckpt["init_args"]["configuration"],
        "patch_size": patch,
        "input_channels": channels,
        "output_channels": int(predictor.label_manager.num_segmentation_heads),
        "mirror_axes": list(axes) if axes is not None else None,
        "export_check_max_abs_diff": diff,
        "versions": {"torch": torch.__version__, "nnunetv2": _version("nnunetv2"), "python": platform.python_version()},
        "exported_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return manifest
