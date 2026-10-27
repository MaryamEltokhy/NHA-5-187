"""Export seg-model v1 to TorchScript for fast prediction (M4-T01).

    python scripts/export_model.py --out /content/drive/MyDrive/BrainMRI_Data/models/seg-model-v1-torchscript

The checkpoint is checked against the SHA-256 in models/registry/seg-model/v1/manifest.json before anything is traced.
See src/bmts/deploy/export.py for the folder it writes and docs/DEPLOY_MODEL.md for how to use it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bmts.deploy.export import export_model  # noqa: E402

DRIVE = "/content/drive/MyDrive/BrainMRI_Data"
MODEL_DIR = (f"{DRIVE}/runs/M2-T04_nnunet/nnUNet_results/Dataset501_BraTS2021/"
             "nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres")
REGISTRY_MANIFEST = REPO / "models/registry/seg-model/v1/manifest.json"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=MODEL_DIR, help="trained nnU-Net folder (plans.json, dataset.json, fold_0/)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoint_best.pth")
    p.add_argument("--no-sha-check", action="store_true",
                   help="do not compare the checkpoint with the registry manifest (only for other models)")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    expected = None
    if not args.no_sha_check:
        expected = json.loads(REGISTRY_MANIFEST.read_text())["checkpoint_file"]["sha256"]
    manifest = export_model(args.model_dir, args.out, fold=args.fold, checkpoint=args.checkpoint,
                            expected_sha256=expected)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
