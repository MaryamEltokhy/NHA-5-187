"""M1-T05 smoke test: prove the data loader feeds a model without errors (2 tiny epochs).

    python scripts/smoke_train.py --config configs/data/loader_v1.yaml --cases 8 --epochs 2

It trains a deliberately small 3D U-Net on a few training patients, runs one validation pass on a few
validation patients, and writes a JSON report. The scores mean nothing after 2 epochs: the point is that
every batch loads with the right shapes and values. Locked test patients are never read.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from monai.inferers import sliding_window_inference  # noqa: E402
from monai.losses import DiceLoss  # noqa: E402
from monai.metrics import DiceMetric  # noqa: E402
from monai.networks.nets import UNet  # noqa: E402
from monai.utils import set_determinism  # noqa: E402

from bmts.common.constants import MODALITIES, REGIONS  # noqa: E402
from bmts.data.datasets.brats import build_loaders, load_config  # noqa: E402


def check_batch(batch, roi):
    """Fail loudly if a training batch does not have the shapes and values the model expects."""
    image, label = batch["image"], batch["label"]
    assert image.ndim == 5 and image.shape[1] == len(MODALITIES), f"image shape {tuple(image.shape)}"
    assert label.shape[1] == len(REGIONS) and label.shape[2:] == image.shape[2:], f"label shape {tuple(label.shape)}"
    assert list(image.shape[2:]) == list(roi), f"patch {tuple(image.shape[2:])} != roi {roi}"
    assert torch.isfinite(image).all(), "image has NaN/Inf"
    assert set(label.unique().tolist()) <= {0.0, 1.0}, "label is not binary"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/data/loader_v1.yaml")
    p.add_argument("--processed-root", help="override processed_root")
    p.add_argument("--split-csv", help="override split_csv")
    p.add_argument("--cases", type=int, default=8, help="training patients to use")
    p.add_argument("--val-cases", type=int, default=2, help="validation patients to use")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--roi", type=int, nargs=3, help="patch size, e.g. 96 96 96")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--report", default="smoke_report.json")
    args = p.parse_args()

    cfg = load_config(args.config, processed_root=args.processed_root, split_csv=args.split_csv,
                      roi_size=args.roi, batch_size=args.batch_size, num_workers=args.num_workers)
    set_determinism(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    print(f"device: {device}{' (' + torch.cuda.get_device_name(0) + ')' if amp else ''}")

    train_loader, val_loader = build_loaders(cfg, train_limit=args.cases, val_limit=args.val_cases)
    print(f"training cases: {len(train_loader.dataset)}, validation cases: {len(val_loader.dataset)}, "
          f"patch {cfg['roi_size']}, batch {cfg['batch_size']}")

    model = UNet(spatial_dims=3, in_channels=len(MODALITIES), out_channels=len(REGIONS),
                 channels=(16, 32, 64, 128), strides=(2, 2, 2), num_res_units=1).to(device)
    loss_fn = DiceLoss(sigmoid=True, smooth_nr=1e-5, smooth_dr=1e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    report = {"device": str(device), "config": cfg, "epochs": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        start, losses = time.time(), []
        for step, batch in enumerate(train_loader):
            check_batch(batch, cfg["roi_size"])
            if epoch == 1 and step == 0:
                img, lab = batch["image"], batch["label"]
                print(f"first batch: image {tuple(img.shape)} {img.dtype}, label {tuple(lab.shape)}, "
                      f"tumor voxels per region {[int(v) for v in lab.sum(dim=(0, 2, 3, 4))]}")
            image, label = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                loss = loss_fn(model(image), label)
            assert torch.isfinite(loss), f"loss is {loss.item()}"
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        train_time = time.time() - start

        model.eval()
        dice = DiceMetric(include_background=True, reduction="mean_batch")
        start = time.time()
        with torch.no_grad(), torch.autocast(device.type, enabled=amp):
            for batch in val_loader:
                image, label = batch["image"].to(device), batch["label"].to(device)
                logits = sliding_window_inference(image, cfg["roi_size"], 1, model, overlap=0.25)
                dice((torch.sigmoid(logits.float()) > 0.5).float(), label)
        scores = dice.aggregate().tolist() if len(val_loader.dataset) else []
        entry = {"epoch": epoch, "train_batches": len(losses), "mean_loss": sum(losses) / len(losses),
                 "train_seconds": round(train_time, 1), "val_seconds": round(time.time() - start, 1),
                 "val_dice": dict(zip(REGIONS, [round(s, 4) for s in scores]))}
        report["epochs"].append(entry)
        print(f"epoch {epoch}: {entry['train_batches']} batches, loss {entry['mean_loss']:.4f}, "
              f"{entry['train_seconds']}s train, {entry['val_seconds']}s val, val Dice {entry['val_dice']}")

    if amp:
        report["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        print(f"peak GPU memory: {report['peak_gpu_memory_gb']} GB")
    report["result"] = "passed"
    Path(args.report).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"SMOKE TEST PASSED: every batch loaded with the right shapes and values. Report: {args.report}")


if __name__ == "__main__":
    main()
