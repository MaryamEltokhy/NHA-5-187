"""Exploratory data analysis of the cleaned scans: tables, figures and summary numbers for the EDA report.

    python scripts/eda.py --out /content/drive/MyDrive/BrainMRI_Data/eda

Only training and validation scans are opened. Locked test patients and the external cohorts appear in patient
counts only (read from the split file), so the lockbox stays untouched.

Writes to --out:
    cohort_patients.csv, cohort_scans.csv, cohort_modalities.csv   counts per cohort and split
    tumor_per_scan.csv          volumes, composition, parts, position of the tumor in every scan (mask only)
    intensity_per_scan.csv      brain volume, tumor share, mean z-score per MRI type and tissue (random sample)
    raw_vs_clean.csv            intensity percentiles of a few BraTS scans before and after cleaning (optional)
    difficult_cases.csv         smallest tumors, no enhancing tumor, many separate parts, tumor at the edge
    summary.json                every number quoted in docs/EDA_REPORT.md
    figures/f01..f15_*.png
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmts.common.constants import ED, ET, MODALITIES, NCR, SEG  # noqa: E402
from bmts.data.datasets.brats import list_cases  # noqa: E402
from bmts.eda import figures as F  # noqa: E402
from bmts.eda.stats import (HIST_EDGES, TISSUES, best_axial_slice, intensity_stats, nonzero_percentiles,  # noqa: E402
                            tumor_stats)

DRIVE = "/content/drive/MyDrive/BrainMRI_Data"
CANONICAL_SHAPE = (240, 240, 155)


def load(path, dtype=np.float32):
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    return (data.astype(dtype, copy=False) if dtype else np.asarray(data)), tuple(float(z) for z in img.header.get_zooms()[:3])


def tumor_slice(seg):
    """Axial slice with the most tumor core (so all three labels show), or the most whole tumor if there is no core."""
    core = np.isin(seg, (NCR, ET))
    return best_axial_slice(core if core.any() else np.isin(seg, (NCR, ED, ET)))


def subject_of(case_id):
    """'brats2021__BraTS2021_00032/tp00' -> 'BraTS2021_00032'."""
    return case_id.split("/")[0].split("__", 1)[-1]


# ---------------------------------------------------------------- counts (no image is opened)

def cohort_tables(split_csv, index_csv):
    split = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
    patients = split.groupby(["dataset_source", "split"]).canonical_subject_id.nunique().unstack(fill_value=0)
    scans = modalities = per_patient = None
    if index_csv and Path(index_csv).is_file():
        index = pd.read_csv(index_csv, dtype=str, keep_default_na=False)
        if "split" not in index or index.split.eq("").all():
            index = index.drop(columns="split", errors="ignore").merge(
                split[["canonical_subject_id", "split"]], on="canonical_subject_id", how="left")
        scans = index.groupby(["dataset_source", "split"]).size().unstack(fill_value=0)
        flags = [c for c in ("T1_available", "T1ce_available", "T2_available", "FLAIR_available", "segmentation_available")
                 if c in index]
        modalities = index.groupby("dataset_source")[flags].agg(lambda s: int((s.astype(str) == "1").sum()))
        modalities.insert(0, "scans", index.groupby("dataset_source").size())
        per_patient = index.groupby(["dataset_source", "canonical_subject_id"]).size().rename("n").reset_index()
    return patients, scans, modalities, per_patient


# ---------------------------------------------------------------- per-scan work (runs in worker processes)

def _mask_chunk(cases):
    rows, freq = [], {}
    for case in cases:
        seg, spacing = load(case[SEG], dtype=None)
        row = {"id": case["id"], "dataset": case["dataset"], "split": case["split"],
               "shape": "x".join(map(str, seg.shape)), "spacing_mm": "x".join(f"{s:g}" for s in spacing)}
        row.update(tumor_stats(seg, spacing))
        rows.append(row)
        if seg.shape == CANONICAL_SHAPE:
            acc = freq.setdefault(case["dataset"], np.zeros(CANONICAL_SHAPE, np.uint16))
            acc += np.isin(seg, (NCR, ED, ET))
    return rows, freq


def _intensity_chunk(cases):
    rows, hist, flair_sum, n_flair = [], {}, None, 0
    for case in cases:
        images = {m: load(case[m])[0] for m in MODALITIES}
        seg, spacing = load(case[SEG], dtype=None)
        row, h = intensity_stats(images, seg, spacing)
        row = {"id": case["id"], "dataset": case["dataset"], "split": case["split"], **row}
        rows.append(row)
        acc = hist.setdefault(case["dataset"], {m: {t: np.zeros(len(HIST_EDGES) - 1, np.int64) for t in TISSUES}
                                                for m in MODALITIES})
        for m in MODALITIES:
            for t in TISSUES:
                acc[m][t] += h[m][t]
        if case["dataset"] == "brats2021" and images["flair"].shape == CANONICAL_SHAPE:
            flair_sum = images["flair"].astype(np.float64) if flair_sum is None else flair_sum + images["flair"]
            n_flair += 1
    return rows, hist, flair_sum, n_flair


def _chunks(items, n):
    return [items[i::n] for i in range(n) if items[i::n]]


def run_parallel(func, cases, workers):
    chunks = _chunks(cases, max(workers * 8, 1))
    if workers <= 1:
        return [func(c) for c in chunks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, chunks))


# ---------------------------------------------------------------- raw scans (optional)

def raw_sample(tar_path, subjects, workdir):
    """Pull the four MRI types of a few patients out of the raw BraTS 2021 archive (streamed, stops when found)."""
    wanted = {(s, m) for s in subjects for m in MODALITIES}
    pattern = re.compile(r"(BraTS2021_\d+)_(t1|t1ce|t2|flair)\.nii\.gz$")
    found = {}
    with tarfile.open(tar_path, "r|*") as tar:
        for member in tar:
            match = pattern.search(member.name)
            if not match or (match.group(1), match.group(2)) not in wanted:
                continue
            target = Path(workdir) / f"{match.group(1)}_{match.group(2)}.nii.gz"   # our own name, never the member path
            target.write_bytes(tar.extractfile(member).read())
            found[(match.group(1), match.group(2))] = target
            if len(found) == len(wanted):
                break
    return found


# ---------------------------------------------------------------- summary helpers

def _q(series):
    s = pd.Series(series).dropna()
    return {"median": round(float(s.median()), 2), "q1": round(float(s.quantile(0.25)), 2),
            "q3": round(float(s.quantile(0.75)), 2), "min": round(float(s.min()), 3), "max": round(float(s.max()), 1)} \
        if len(s) else None


def summarize_cohort(df):
    out = {"scans": int(len(df)), "patients": int(df.id.map(lambda i: i.split("/")[0]).nunique())}
    for col in ("vol_wt_ml", "vol_tc_ml", "vol_et_ml", "vol_ed_ml", "vol_ncr_ml", "vol_rc_ml"):
        out[col] = _q(df[col])
    for t in ("ncr", "ed", "et"):
        out[f"share_{t}_of_wt"] = _q(df[f"share_{t}_of_wt"] * 100)
    no_et = ~df.et_present.astype(bool)
    out["no_et"] = {"n": int(no_et.sum()), "pct": round(float(no_et.mean() * 100), 1)}
    no_tumor = df.vol_wt_ml == 0
    out["no_tumor"] = int(no_tumor.sum())
    multi = df.wt_components >= 2
    out["multifocal"] = {"n": int(multi.sum()), "pct": round(float(multi.mean() * 100), 1)}
    out["touches_border"] = int(df.touches_border.astype(bool).sum())
    out["has_resection_cavity"] = int((df.vol_rc_ml > 0).sum())
    out["shapes"] = df["shape"].value_counts().to_dict()
    out["spacings_mm"] = df["spacing_mm"].value_counts().to_dict()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=f"{DRIVE}/eda")
    p.add_argument("--split-csv", default=f"{DRIVE}/qc/split_v1.csv")
    p.add_argument("--index-csv", default=f"{DRIVE}/qc/unified_subject_index_v4.csv", help="for scan counts (optional)")
    p.add_argument("--processed-root", default="/content/processed")
    p.add_argument("--datasets", nargs="+", default=["brats2021", "mu", "rhuh"])
    p.add_argument("--intensity-sample", type=int, default=150, help="scans per cohort for the intensity statistics")
    p.add_argument("--raw-tar", default=f"{DRIVE}/raw/brats2021/BraTS2021_Training_Data.tar",
                   help="raw BraTS 2021 archive for the before/after-cleaning figure ('' to skip)")
    p.add_argument("--raw-sample", type=int, default=10)
    p.add_argument("--scores-csv", default=str(Path(__file__).resolve().parents[1] / "results/M2-T04/scores_nnunet_per_case.csv"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out, fig_dir = Path(args.out), Path(args.out) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    summary, start = {}, time.time()

    def log(msg):
        print(f"[{(time.time() - start) / 60:5.1f} min] {msg}", flush=True)

    # 1. counts per cohort (split file + subject index; no scan is opened)
    patients, scans, modalities, per_patient = cohort_tables(args.split_csv, args.index_csv)
    patients.to_csv(out / "cohort_patients.csv")
    summary["patients"] = {c: patients.loc[c].astype(int).to_dict() for c in patients.index}
    if scans is not None:
        scans.to_csv(out / "cohort_scans.csv")
        modalities.to_csv(out / "cohort_modalities.csv")
        summary["scans"] = {c: scans.loc[c].astype(int).to_dict() for c in scans.index}
        summary["modalities"] = {c: modalities.loc[c].astype(int).to_dict() for c in modalities.index}
        F.cohorts(patients, scans, fig_dir)
        F.timepoints(per_patient, fig_dir)
    log("cohort counts done")

    # 2. training + validation scans that are on this disk
    cases = []
    for dataset in args.datasets:
        for split in ("train", "val"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                found = list_cases(args.split_csv, args.processed_root, splits=(split,), datasets=(dataset,),
                                   on_missing="skip")
            if caught:
                log(f"{dataset}/{split}: {str(caught[0].message).splitlines()[0]}")
            cases += [{**c, "dataset": dataset, "split": split} for c in found]
            log(f"{dataset}/{split}: {len(found)} scans")
    if not cases:
        sys.exit("No cleaned scans found under --processed-root; restore or clean them first.")

    # 3. masks of every scan: volumes, composition, separate parts, position, location map
    results = run_parallel(_mask_chunk, cases, args.workers)
    tumor = pd.DataFrame([r for rows, _ in results for r in rows]).sort_values("id")
    freq = {}
    for _, f in results:
        for k, v in f.items():
            freq[k] = freq.get(k, 0) + v.astype(np.uint32)
    tumor.to_csv(out / "tumor_per_scan.csv", index=False)
    summary["tumor"] = {d: summarize_cohort(tumor[tumor.dataset == d]) for d in F._cohort_order(tumor.dataset)}
    log(f"tumor statistics of {len(tumor)} scans done")

    # 4. intensity statistics on a random sample per cohort
    sample = []
    for d in F._cohort_order(tumor.dataset):
        pool = [c for c in cases if c["dataset"] == d]
        sample += [pool[i] for i in rng.choice(len(pool), min(args.intensity_sample, len(pool)), replace=False)]
    results = run_parallel(_intensity_chunk, sample, args.workers)
    intensity = pd.DataFrame([r for rows, *_ in results for r in rows]).sort_values("id")
    intensity.to_csv(out / "intensity_per_scan.csv", index=False)
    hist, flair_sum, n_flair = {}, None, 0
    for _, h, fs, n in results:
        for d, per_mod in h.items():
            acc = hist.setdefault(d, {m: {t: 0 for t in TISSUES} for m in MODALITIES})
            for m in MODALITIES:
                for t in TISSUES:
                    acc[m][t] = acc[m][t] + per_mod[m][t]
        if fs is not None:
            flair_sum = fs if flair_sum is None else flair_sum + fs
            n_flair += n
    np.savez_compressed(out / "intensity_histograms.npz", edges=HIST_EDGES,
                        **{f"{d}_{m}_{t}": hist[d][m][t] for d in hist for m in MODALITIES for t in TISSUES})
    summary["intensity"] = {}
    for d in hist:
        totals = {t: float(hist[d]["flair"][t].sum()) for t in TISSUES}    # voxel counts per tissue
        brain = sum(totals.values())
        sub = intensity[intensity.dataset == d]
        summary["intensity"][d] = {
            "sample_scans": int(len(sub)),
            "brain_ml": _q(sub.brain_ml),
            "wt_share_of_brain_pct": _q(sub.wt_share_of_brain * 100),
            "tissue_share_of_brain_pct": {t: round(totals[t] / brain * 100, 3) for t in TISSUES},
            "mean_z": {m: {t: round(float(sub[f"mean_{m}_{t}"].mean()), 2) for t in TISSUES} for m in MODALITIES},
        }
    log(f"intensity statistics of {len(intensity)} sample scans done")

    # 5. figures from the tables
    F.volumes(tumor, fig_dir)
    F.composition(tumor, fig_dir)
    F.components(tumor, fig_dir)
    rho = sstats.spearmanr(*tumor.loc[tumor.dataset == "brats2021", ["vol_tc_ml", "vol_ed_ml"]].to_numpy().T).statistic
    summary["spearman_ed_vs_tc_brats2021"] = round(float(rho), 2)
    F.ed_vs_core(tumor, rho, fig_dir)
    if "brats2021" in hist:
        F.intensity_by_tissue(hist["brats2021"], fig_dir)
        b = summary["intensity"]["brats2021"]
        F.imbalance({t: b["tissue_share_of_brain_pct"][t] / 100 for t in TISSUES},
                    intensity.loc[intensity.dataset == "brats2021", "wt_share_of_brain"].to_numpy(), fig_dir)
    if "brats2021" in freq:
        n_brats = int((tumor.dataset == "brats2021").sum())
        F.location(freq["brats2021"], flair_sum / n_flair if n_flair else None, n_brats, fig_dir)
        np.save(out / "location_frequency_brats2021.npy", freq["brats2021"].astype(np.uint16))
    brats = tumor[tumor.dataset == "brats2021"]
    if brats.split.nunique() == 2:
        summary["train_vs_val_ks_p"] = {}
        for col in ("vol_wt_ml", "vol_tc_ml", "vol_et_ml"):
            summary["train_vs_val_ks_p"][col] = round(float(sstats.ks_2samp(
                brats.loc[brats.split == "train", col], brats.loc[brats.split == "val", col]).pvalue), 3)
        F.train_vs_val(brats, summary["train_vs_val_ks_p"]["vol_wt_ml"], fig_dir)
    log("table figures done")

    # 6. example patients, three views, difficult cases (BraTS 2021 training patients)
    by_id = {c["id"]: c for c in cases}
    train = brats[brats.split == "train"].sort_values("vol_wt_ml").reset_index(drop=True)
    if len(train):
        def volumes_of(case_id):
            c = by_id[case_id]
            return {**{m: load(c[m])[0] for m in MODALITIES}, SEG: load(c[SEG], dtype=None)[0]}

        picks = [train.iloc[int(q * (len(train) - 1))] for q in (0.1, 0.5, 0.9)]
        rows = []
        for r, label in zip(picks, ("small", "typical", "large")):
            v = volumes_of(r.id)
            rows.append((f"{subject_of(r.id)}\n{label}: WT {r.vol_wt_ml:.0f} mL", v, tumor_slice(v[SEG])))
        F.examples(rows, fig_dir)
        mid = picks[1]
        v = rows[1][1]
        centre = [mid[f"centroid_{a}"] * (s - 1) for a, s in zip("xyz", v[SEG].shape)]
        F.multiplanar(v, centre, subject_of(mid.id), fig_dir)

        flagged = []
        everyone = brats.sort_values("vol_wt_ml")
        tumors = everyone[everyone.vol_wt_ml > 0]
        flagged.append(("smallest whole tumor", tumors.iloc[0]))
        no_et = everyone[~everyone.et_present.astype(bool) & (everyone.vol_wt_ml > 0)]
        if len(no_et):
            flagged.append(("no enhancing tumor", no_et.iloc[len(no_et) // 2]))
        flagged.append(("most separate parts", everyone.sort_values(["wt_components", "vol_wt_ml"]).iloc[-1]))
        edge = everyone[everyone.touches_border.astype(bool)]
        flagged.append(("tumor touches the scan edge", edge.iloc[-1]) if len(edge) else ("largest whole tumor", tumors.iloc[-1]))
        gallery = []
        for reason, r in flagged:
            v = volumes_of(r.id)
            z = tumor_slice(v[SEG])
            gallery.append((subject_of(r.id), f"{reason}\nWT {r.vol_wt_ml:.1f} mL, {int(r.wt_components)} part(s)",
                            v["flair"][:, :, z], v[SEG][:, :, z]))
        F.difficult(gallery, fig_dir)

    hard = pd.concat([
        tumor[tumor.vol_wt_ml > 0].nsmallest(10, "vol_wt_ml").assign(reason="10 smallest whole tumors"),
        tumor[~tumor.et_present.astype(bool)].assign(reason="no enhancing tumor in the expert mask"),
        tumor[tumor.wt_components >= 3].assign(reason="3 or more separate tumor parts"),
        tumor[tumor.touches_border.astype(bool)].assign(reason="tumor touches the scan edge"),
    ])[["reason", "id", "dataset", "split", "vol_wt_ml", "vol_tc_ml", "vol_et_ml", "wt_components"]]
    hard.to_csv(out / "difficult_cases.csv", index=False)
    summary["difficult_cases"] = hard.groupby("reason").size().to_dict()
    log("example and difficult-case figures done")

    # 7. size against the selected model's validation Dice (validation patients only)
    if args.scores_csv and Path(args.scores_csv).is_file():
        scores = pd.read_csv(args.scores_csv)
        val = brats[brats.split == "val"].assign(case=lambda d: d.id.map(subject_of))
        merged = val.merge(scores, on="case", how="inner")
        if len(merged):
            F.volume_vs_dice(merged, fig_dir)
            summary["dice_vs_volume"] = {"cases": int(len(merged))}
            for region, col in (("WT", "vol_wt_ml"), ("TC", "vol_tc_ml"), ("ET", "vol_et_ml")):
                d = merged[merged[col] > 0]
                small = d[col] < d[col].quantile(0.25)
                summary["dice_vs_volume"][region] = {
                    "spearman": round(float(sstats.spearmanr(d[col], d[f"dice_{region}"]).statistic), 2),
                    "median_dice_smallest_quarter": round(float(d.loc[small, f"dice_{region}"].median()), 3),
                    "median_dice_other": round(float(d.loc[~small, f"dice_{region}"].median()), 3)}
        log("volume vs Dice done")

    # 8. raw vs cleaned intensities of a few BraTS training patients (streams the raw archive)
    if args.raw_tar and Path(args.raw_tar).is_file() and len(train):
        subjects = sorted(train.id.map(subject_of))[: args.raw_sample]
        with tempfile.TemporaryDirectory() as tmp:
            found = raw_sample(args.raw_tar, subjects, tmp)
            rows = []
            for (s, m), path in sorted(found.items()):
                clean = by_id[train.loc[train.id.map(subject_of) == s, "id"].iloc[0]][m]
                rows.append({"subject": s, "modality": m, "stage": "raw", **nonzero_percentiles(load(path)[0])})
                rows.append({"subject": s, "modality": m, "stage": "clean", **nonzero_percentiles(load(clean)[0])})
        rc = pd.DataFrame(rows)
        rc.to_csv(out / "raw_vs_clean.csv", index=False)
        if len(rc):
            F.raw_vs_clean(rc[rc.stage == "raw"], rc[rc.stage == "clean"], fig_dir)
            raw = rc[rc.stage == "raw"]
            summary["raw_p99_range"] = {m: [round(float(raw.loc[raw.modality == m, "p99"].min()), 1),
                                            round(float(raw.loc[raw.modality == m, "p99"].max()), 1)] for m in MODALITIES}
        log(f"raw vs cleaned: {len(found)} raw files")

    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    log(f"EDA FINISHED: {len(list(fig_dir.glob('*.png')))} figures and summary.json in {out}")


if __name__ == "__main__":
    main()
