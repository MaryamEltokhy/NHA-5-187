"""One command for the whole analysis (M3-T05): scans in -> mask, volumes, change, radiomics and confidence map out.

A patient from the team data (checked against the split file; locked test and external scans are refused):
    python scripts/analyze.py --dataset rhuh --subject RHUH-0002 --out /content/analysis/RHUH-0002
    python scripts/analyze.py --dataset brats2021 --subject BraTS2021_00032 --out /content/analysis/BraTS2021_00032

Any scan folders (each with t1, t1ce, t2, flair .nii.gz and optionally seg.nii.gz), in time order:
    python scripts/analyze.py --scan scans/exam1 --scan scans/exam2 --out out/  [--raw to clean them first]

Outputs and steps: see src/bmts/pipeline/analyze.py and docs/PIPELINE.md. Research prototype for decision support only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmts.data.splits import check_subject  # noqa: E402
from bmts.pipeline.analyze import Scan, analyze  # noqa: E402
from bmts.quantitative.longitudinal import MIN_BASELINE_CM3  # noqa: E402

DRIVE = "/content/drive/MyDrive/BrainMRI_Data"
MODEL_DIR = (f"{DRIVE}/runs/M2-T04_nnunet/nnUNet_results/Dataset501_BraTS2021/"
             "nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres")
TIMEPOINT_NAMES = {"rhuh": {0: "pre-op", 1: "early post-op", 2: "recurrence"}}   # RHUH-GBM's own naming


def timepoint_value(name):
    """Folder names such as '0', '0.0' or 'Timepoint_2' -> a number for sorting."""
    digits = "".join(ch for ch in name if ch.isdigit() or ch == ".")
    try:
        return float(digits)
    except ValueError:
        return name


def subject_scans(dataset, subject, processed_root, split_csv):
    """All cleaned scans of one patient from processed_root/<dataset>/<subject>/<timepoint>/ after the lockbox check."""
    split = check_subject(split_csv, dataset, subject)
    folders = sorted((p for p in (Path(processed_root) / dataset / subject).glob("*") if p.is_dir()),
                     key=lambda p: timepoint_value(p.name))
    if not folders:
        raise FileNotFoundError(f"no cleaned scans in {Path(processed_root) / dataset / subject}")
    names = TIMEPOINT_NAMES.get(dataset, {})
    scans = []
    for folder in folders:
        tp = timepoint_value(folder.name)
        label = names.get(int(tp), folder.name) if isinstance(tp, float) else folder.name
        scan_id = subject if len(folders) == 1 else f"{subject}_tp{folder.name}"
        scan = Scan.from_folder(folder, scan_id=scan_id, timepoint=tp, label=label, patient=subject)
        scan.extra.update(dataset=dataset, split=split)
        scans.append(scan)
    print(f"{dataset}/{subject} ({split}): {len(scans)} scan(s) -> {[s.label for s in scans]}", flush=True)
    return scans


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--subject", help="patient id from the team data, e.g. RHUH-0002 (needs --dataset)")
    src.add_argument("--scan", action="append", help="scan folder; repeat in time order for change over time")
    p.add_argument("--dataset", help="dataset of --subject (brats2021, rhuh, mu); with --raw it also picks the label "
                                     "remap of the cleaning step (brats2021: ET 4 -> 3)")
    p.add_argument("--patient", default="patient", help="patient name for --scan folders")
    p.add_argument("--raw", action="store_true", help="--scan folders are not cleaned yet: run the M1-T04 cleaning")
    p.add_argument("--processed-root", default="/content/processed")
    p.add_argument("--split-csv", default=f"{DRIVE}/qc/split_v1.csv")
    p.add_argument("--model-dir", default=MODEL_DIR)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoint_best.pth")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--no-radiomics", action="store_true")
    p.add_argument("--min-baseline-cm3", type=float, default=MIN_BASELINE_CM3,
                   help="smallest earlier volume for which a percentage change is given")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.subject:
        if not args.dataset:
            p.error("--subject needs --dataset")
        scans = subject_scans(args.dataset, args.subject, args.processed_root, args.split_csv)
    else:
        scans = []
        for i, folder in enumerate(Path(f) for f in args.scan):
            if args.raw:     # raw files: same names, cleaned (and labels remapped for --dataset) by the pipeline
                seg = folder / "seg.nii.gz"
                scan = Scan(folder.name, {m: str(folder / f"{m}.nii.gz") for m in ("t1", "t1ce", "t2", "flair")},
                            i, folder.name, str(seg) if seg.is_file() else None, args.patient)
            else:
                scan = Scan.from_folder(folder, timepoint=i, label=folder.name, patient=args.patient)
            scan.extra["dataset"] = args.dataset or ""
            scans.append(scan)

    from bmts.uncertainty.nnunet_tta import load_predictor
    predictor = load_predictor(args.model_dir, args.fold, args.checkpoint, args.device)
    summary = analyze(scans, args.out, predictor, clean=args.raw, radiomics=not args.no_radiomics,
                      min_baseline_cm3=args.min_baseline_cm3)
    print("ANALYSIS FINISHED\n" + json.dumps(summary["outputs"], indent=1))


if __name__ == "__main__":
    main()
