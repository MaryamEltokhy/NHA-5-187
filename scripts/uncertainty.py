"""Confidence maps for seg-model v1 on the validation patients, and a check that they point at the mistakes (M3-T04).

    python scripts/uncertainty.py --model-dir <nnUNet_results>/Dataset501_BraTS2021/nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres

Each validation scan is predicted 8 times (one pass per mirror flip, see src/bmts/uncertainty/nnunet_tta.py). The
mask is the same as seg-model v1's; the passes give the confidence map. Against the expert masks the script measures:
calibration (ECE), whether uncertain voxels are the wrong ones (AUROC), and whether uncertain cases have lower Dice.
Locked test patients are never loaded. Rerunning continues where it stopped (state kept in --out).

Writes to --out: per_case.csv, summary.json, figures/*.png, slices/<case>.npz (one axial slice per case), and for the
first --save-maps cases maps/<case>_{seg,uncertainty,disagreement}.nii.gz.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmts.common.constants import MODALITIES, REGIONS, SEG  # noqa: E402
from bmts.data.datasets.brats import list_cases  # noqa: E402
from bmts.uncertainty import figures as F  # noqa: E402
from bmts.uncertainty import nnunet_tta as T  # noqa: E402
from bmts.uncertainty.maps import CalibrationCounter, ErrorDetectionCounter, case_summary  # noqa: E402

DRIVE = "/content/drive/MyDrive/BrainMRI_Data"
MODEL_DIR = (f"{DRIVE}/runs/M2-T04_nnunet/nnUNet_results/Dataset501_BraTS2021/"
             "nnUNetTrainer_50epochs__nnUNetPlans__3d_fullres")


def dice(pred, truth):
    s = pred.sum() + truth.sum()
    return 1.0 if s == 0 else float(2 * (pred & truth).sum() / s)


def tumor_slice_index(seg, gt):
    """Axial slice (axis 0 in z, y, x order) with the most predicted or expert tumor."""
    area = ((seg > 0) | (gt > 0)).reshape(seg.shape[0], -1).sum(1)
    return int(area.argmax())


def evaluate_case(predictor, case, out, save_maps):
    files = [case[m] for m in MODALITIES]
    data, props = T.preprocess(predictor, files)
    seg, maps, n_passes = T.predict_case_tta(predictor, data, props)
    gt = T.read_sitk(case[SEG], np.uint8)
    flair = T.read_sitk(case["flair"], np.float32)
    brain = np.zeros(gt.shape, bool)
    for m in MODALITIES:
        brain |= T.read_sitk(case[m]) != 0
    spacing = props["sitk_stuff"]["spacing"]
    row = {"case": case["id"].split("/")[0].split("__")[-1], "passes": n_passes,
           **case_summary(seg, maps, voxel_ml=float(np.prod(spacing)) / 1000)}
    for name, labels in REGIONS.items():
        row[f"dice_{name}"] = dice(np.isin(seg, labels), np.isin(gt, labels))
        row[f"gt_vol_{name}_ml"] = float(np.isin(gt, labels).sum() * np.prod(spacing) / 1000)
    z = tumor_slice_index(seg, gt)
    np.savez_compressed(out / "slices" / f"{row['case']}.npz", flair=flair[z].astype(np.float16), seg=seg[z],
                        gt=gt[z], uncertainty=maps["uncertainty"][z].astype(np.float16), z=z)
    if save_maps:
        T.write_like(seg, out / "maps" / f"{row['case']}_seg.nii.gz", props)
        T.write_like(maps["uncertainty"], out / "maps" / f"{row['case']}_uncertainty.nii.gz", props)
        T.write_like(maps["disagreement"], out / "maps" / f"{row['case']}_disagreement.nii.gz", props)
    return row, maps["mean_probs"], seg, gt, brain


def summarize(df, calibration, detector, out):
    rho = {r: float(stats.spearmanr(df[f"uncertainty_{r}"], df[f"dice_{r}"]).statistic) for r in REGIONS}
    summary = {"cases": int(len(df)), "passes_per_case": int(df.passes.median()),
               "dice_mean": {r: round(float(df[f"dice_{r}"].mean()), 4) for r in REGIONS},
               "ece": {r: round(calibration.ece(r), 5) for r in REGIONS},
               "error_auroc": {r: round(detector.auroc(r), 4) for r in REGIONS},
               "errors_flagged": {}, "spearman_uncertainty_vs_dice": {r: round(v, 3) for r, v in rho.items()},
               "worst_quarter_caught": {}, "seconds_per_case": round(float(df.seconds.median()), 1)}
    for r in REGIONS:
        errors, everything = detector.share_of_errors_flagged(r)
        summary["errors_flagged"][r] = {"share_of_wrong_voxels_uncertain": round(errors, 4),
                                        "share_of_all_voxels_uncertain": round(everything, 5)}
        # of the quarter of cases with the lowest Dice, how many are in the most uncertain quarter
        worst = df[f"dice_{r}"] <= df[f"dice_{r}"].quantile(0.25)
        unsure = df[f"uncertainty_{r}"] >= df[f"uncertainty_{r}"].quantile(0.75)
        summary["worst_quarter_caught"][r] = round(float((worst & unsure).sum() / max(worst.sum(), 1)), 3)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    F.reliability(calibration, fig_dir)
    F.uncertainty_vs_dice(df, rho, fig_dir)
    F.error_detection(detector, fig_dir)
    mean_dice = df[[f"dice_{r}" for r in REGIONS]].mean(axis=1)
    picks = [("best", mean_dice.idxmax()), ("typical", (mean_dice - mean_dice.median()).abs().idxmin()),
             ("worst", mean_dice.idxmin())]
    rows = []
    for label, i in picks:
        r = df.loc[i]
        sl = dict(np.load(out / "slices" / f"{r.case}.npz"))
        rows.append((f"{r.case} ({label}, mean Dice {mean_dice[i]:.2f})", {k: sl[k].astype(np.float32) for k in
                                                                           ("flair", "seg", "gt", "uncertainty")}))
    F.gallery(rows, fig_dir)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=MODEL_DIR, help="nnU-Net trainer folder (contains plans.json, fold_0/)")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoint_best.pth")
    p.add_argument("--split-csv", default=f"{DRIVE}/qc/split_v1.csv")
    p.add_argument("--processed-root", default="/content/processed")
    p.add_argument("--out", default=f"{DRIVE}/runs/M3-T04_uncertainty")
    p.add_argument("--cases", type=int, help="only the first N validation patients (default: all 188)")
    p.add_argument("--save-maps", type=int, default=10, help="write NIfTI maps for the first N cases")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-hours", type=float, default=10.0, help="stop cleanly after this long; rerun to continue")
    args = p.parse_args()

    out = Path(args.out)
    for sub in ("slices", "maps"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cases = list_cases(args.split_csv, args.processed_root, splits=("val",), datasets=("brats2021",),
                       limit=args.cases)                     # refuses locked test patients
    state_file, table_file = out / "state.pkl", out / "per_case.csv"
    if state_file.exists() and table_file.exists():
        calibration, detector = pickle.loads(state_file.read_bytes())
        done = pd.read_csv(table_file)
    else:
        calibration, detector, done = CalibrationCounter(), ErrorDetectionCounter(), pd.DataFrame()
    finished = set(done.case) if len(done) else set()
    print(f"{len(cases)} validation patients, {len(finished)} already done", flush=True)

    predictor = T.load_predictor(args.model_dir, args.fold, args.checkpoint, args.device)
    start, rows = time.time(), done.to_dict("records")
    for i, case in enumerate(cases):
        name = case["id"].split("/")[0].split("__")[-1]
        if name in finished:
            continue
        if (time.time() - start) / 3600 > args.max_hours:
            print("time budget reached; rerun to continue", flush=True)
            break
        t0 = time.time()
        row, mean_probs, seg, gt, brain = evaluate_case(predictor, case, out, save_maps=i < args.save_maps)
        calibration.add(mean_probs, gt, brain)
        detector.add(mean_probs, seg, gt, brain)
        row["seconds"] = round(time.time() - t0, 1)
        rows.append(row)
        pd.DataFrame(rows).to_csv(table_file, index=False)          # table and counters saved together after
        state_file.write_bytes(pickle.dumps((calibration, detector)))  # every case, so a rerun can continue
        print(f"[{len(rows)}/{len(cases)}] {name}: Dice WT {row['dice_WT']:.3f} TC {row['dice_TC']:.3f} "
              f"ET {row['dice_ET']:.3f} | uncertainty {row['uncertainty_near_tumor']:.3f} | {row['seconds']} s",
              flush=True)

    df = pd.DataFrame(rows)
    if len(df) == len(cases):
        summary = summarize(df, calibration, detector, out)
        print("UNCERTAINTY FINISHED\n" + json.dumps(summary, indent=1))
    else:
        print(f"{len(df)} of {len(cases)} done; rerun the same command to continue")


if __name__ == "__main__":
    main()
