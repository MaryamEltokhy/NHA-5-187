"""Tumor mask for one or more scans with the exported seg-model v1 (M4-T01).

    python scripts/segment.py --export-dir <export folder> --scan scans/exam1 --scan scans/exam2 --out out/
    python scripts/segment.py --export-dir <export folder> --scan scans/exam1 --tta --out out/   # same mask as v1

Each scan folder holds t1, t1ce, t2, flair .nii.gz, cleaned as in M1-T04. Writes out/<folder name>_seg.nii.gz
(0 background, 1 NCR, 2 ED, 3 ET) and out/timings.csv. Research prototype for decision support only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmts.common.constants import MODALITIES  # noqa: E402
from bmts.deploy.segmenter import FastSegmenter  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--export-dir", required=True, help="folder written by scripts/export_model.py")
    p.add_argument("--scan", action="append", required=True, help="scan folder; repeat for several scans")
    p.add_argument("--tta", action="store_true", help="average the 8 mirror flips (slower, identical to seg-model v1)")
    p.add_argument("--tile-step-size", type=float, default=0.5)
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    segmenter = FastSegmenter.from_dir(args.export_dir, device=args.device, tta=args.tta,
                                       tile_step_size=args.tile_step_size)
    out, rows = Path(args.out), []
    for folder in map(Path, args.scan):
        files = [folder / f"{m}.nii.gz" for m in MODALITIES]
        missing = [f.name for f in files if not f.is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} is missing {missing}")
        _, timings = segmenter.segment_to_file(files, out / f"{folder.name}_seg.nii.gz")
        rows.append({"scan": folder.name, "tta": args.tta, **timings})
        print(f"{folder.name}: {timings['total_s']:.1f} s", flush=True)
    pd.DataFrame(rows).to_csv(out / "timings.csv", index=False)


if __name__ == "__main__":
    main()
