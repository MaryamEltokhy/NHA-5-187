"""One analysis pipeline (M3-T05): scans in -> mask, volumes, change over time, radiomics and confidence map out.

Steps for every scan, all with the project's own code:
1. (optional) cleaning, M1-T04: RAS orientation, 240 x 240 x 155 grid, z-score inside the brain
   (bmts.data.cleaning.standardize). Raw input must already be skull-stripped, co-registered and 1 mm
   (the BraTS convention); the cleaning step does not resample or skull-strip.
2. mask + confidence map, M3-T04: seg-model v1 (nnU-Net) predicted once per mirror flip
   (bmts.uncertainty.nnunet_tta.predict_with_confidence); the mask equals nnUNetv2_predict's.
3. volumes per region (TC, WT, ET) and per label (NCR, ED, RC), M3-T01 (bmts.quantitative.volumes.all_volumes).
4. radiomics per region, M3-T03 (bmts.radiomics.features.extract_case, settings configs/radiomics/).
5. if an expert mask is given: Dice per region and the expert volumes (for checking only).
Then, when two or more scans of the same patient are given: volume change between every pair, M3-T02
(bmts.quantitative.longitudinal.change_rows), from the predicted masks and, when present, from the expert masks.

Outputs in out_dir: <scan>/<scan>_{seg,uncertainty,disagreement}.nii.gz, volumes.csv, uncertainty.csv, radiomics.csv,
changes.csv (2+ scans), metrics.csv (expert masks given), run_summary.json.

A research prototype for decision support. The numbers describe the model's masks; they are not a diagnosis.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from bmts.common.constants import MODALITIES, REGIONS, SEG
from bmts.quantitative.longitudinal import MIN_BASELINE_CM3, change_rows, same_grid
from bmts.quantitative.volumes import all_volumes

MEASURES = ("TC", "WT", "ET", "NCR", "ED", "RC")   # regions, then single labels (ET is both)
CHANGE_MEASURES = ("TC", "WT", "ET", "NCR", "ED")
from bmts.uncertainty import nnunet_tta as T
from bmts.uncertainty.maps import case_summary

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Scan:
    """One MRI exam: the four MRI types (and optionally the expert mask) of one patient at one time."""
    scan_id: str                      # unique name, used for output files
    files: dict                       # {"t1": path, "t1ce": path, "t2": path, "flair": path}
    timepoint: object = 0             # sortable time order of the patient's scans (0, 1, 2 or dates)
    label: str = ""                   # readable timepoint name, e.g. "pre-op"
    expert_seg: str | None = None     # optional expert mask, canonical labels 0-4 (checking only)
    patient: str = "patient"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_folder(cls, folder, scan_id=None, timepoint=0, label="", patient="patient"):
        """A cleaned scan folder: {t1,t1ce,t2,flair}.nii.gz and optionally seg.nii.gz (expert mask)."""
        folder = Path(folder)
        files = {m: folder / f"{m}.nii.gz" for m in MODALITIES}
        missing = [m for m, p in files.items() if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"{folder} lacks {', '.join(missing)}")
        seg = folder / f"{SEG}.nii.gz"
        return cls(scan_id or folder.name, {m: str(p) for m, p in files.items()}, timepoint, label,
                   str(seg) if seg.is_file() else None, patient)


def clean_scan(scan: Scan, out_dir) -> Scan:
    """Run the M1-T04 cleaning step on one raw scan; returns the same scan pointing at the cleaned files."""
    from bmts.data.cleaning import standardize as S

    out = Path(out_dir) / "cleaned" / scan.scan_id
    jobs = [(m, p) for m, p in scan.files.items()] + ([(SEG, scan.expert_seg)] if scan.expert_seg else [])
    cleaned = {}
    for kind, path in jobs:
        target = out / f"{kind}.nii.gz"
        status, error = S.standardize_file(SimpleNamespace(path=str(path), out_path=str(target), scan=kind,
                                                           dataset=scan.extra.get("dataset", "")))
        if status == "failed":
            raise RuntimeError(f"cleaning {scan.scan_id}/{kind} failed: {error}")
        cleaned[kind] = str(target)
    seg = cleaned.pop(SEG, None)
    return Scan(scan.scan_id, cleaned, scan.timepoint, scan.label, seg, scan.patient, scan.extra)


def dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 1.0 if s == 0 else float(2 * (a & b).sum() / s)


def _git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def analyze(scans, out_dir, predictor, *, clean=False, radiomics=True, extractor=None, radiomics_params=None,
            min_baseline_cm3=MIN_BASELINE_CM3):
    """Run every step on every scan and write the tables. Returns the run summary (also saved as JSON)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if radiomics:
        from bmts.radiomics.features import DEFAULT_PARAMS, extract_case, make_extractor
        radiomics_params = radiomics_params or DEFAULT_PARAMS
        extractor = extractor or make_extractor(radiomics_params)
    volumes, uncertainty, features, metrics, expert_volumes = [], [], [], [], []
    start = time.time()
    for scan in scans:
        if clean:
            scan = clean_scan(scan, out_dir)
        t0 = time.time()
        case_dir = out_dir / scan.scan_id
        seg, maps, props = T.predict_with_confidence(predictor, [scan.files[m] for m in MODALITIES], case_dir,
                                                     scan.scan_id)
        spacing = tuple(float(s) for s in props["sitk_stuff"]["spacing"])
        base = {"scan_id": scan.scan_id, "patient": scan.patient, "timepoint": scan.timepoint, "label": scan.label}
        vols = all_volumes(seg, spacing)
        volumes.append({**base, **{f"{r}_cm3": vols[r]["cm3"] for r in MEASURES},
                        **{f"{r}_voxels": vols[r]["voxels"] for r in MEASURES}})
        uncertainty.append({**base, **case_summary(seg, maps, voxel_ml=float(np.prod(spacing)) / 1000)})
        del maps
        if radiomics:
            for row in extract_case(scan.files, case_dir / f"{scan.scan_id}_seg.nii.gz", case_id=scan.scan_id,
                                    extractor=extractor, params_path=radiomics_params):
                features.append({"scan_id": scan.scan_id, **{k: v for k, v in row.items() if k != "patient_id"}})
        if scan.expert_seg:
            truth = np.rint(T.read_sitk(scan.expert_seg)).astype(np.int16)
            if truth.shape != seg.shape:
                raise ValueError(f"{scan.scan_id}: expert mask {truth.shape} and prediction {seg.shape} differ")
            truth_vols = all_volumes(truth, spacing)
            expert_volumes.append({**base, **{f"{r}_cm3": truth_vols[r]["cm3"] for r in MEASURES}})
            metrics.append({**base, **{f"dice_{r}": dice(np.isin(seg, l), np.isin(truth, l)) for r, l in REGIONS.items()}})
        volumes[-1]["seconds"] = round(time.time() - t0, 1)
        print(f"{scan.scan_id}: WT {vols['WT']['cm3']:.1f} cm3, TC {vols['TC']['cm3']:.1f}, ET {vols['ET']['cm3']:.1f} "
              f"| uncertainty near tumor {uncertainty[-1]['uncertainty_near_tumor']:.3f} | {volumes[-1]['seconds']} s",
              flush=True)

    written = {"volumes": "volumes.csv", "uncertainty": "uncertainty.csv"}
    pd.DataFrame(volumes).to_csv(out_dir / "volumes.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(out_dir / "uncertainty.csv", index=False)
    if features:
        pd.DataFrame(features).to_csv(out_dir / "radiomics.csv", index=False)
        written["radiomics"] = "radiomics.csv"
    if metrics:
        expert = pd.DataFrame(expert_volumes).rename(columns={f"{m}_cm3": f"{m}_cm3_expert" for m in MEASURES})
        pd.DataFrame(metrics).merge(expert, on=["scan_id", "patient", "timepoint", "label"]).to_csv(
            out_dir / "metrics.csv", index=False)
        written["metrics"] = "metrics.csv"

    changes, grid_notes = _changes(scans, volumes, expert_volumes, min_baseline_cm3)
    if changes:
        pd.DataFrame(changes).to_csv(out_dir / "changes.csv", index=False)
        written["changes"] = "changes.csv"

    summary = {
        "task": "M3-T05 analysis pipeline", "scans": [s.scan_id for s in scans], "cleaned_here": clean,
        "outputs": written, "maps": {s.scan_id: [f"{s.scan_id}/{s.scan_id}_{k}.nii.gz" for k in
                                                 ("seg", "uncertainty", "disagreement")] for s in scans},
        "model": json.loads((REPO / "models/registry/seg-model/v1/manifest.json").read_text())["nnunet"],
        "radiomics_settings": str(Path(radiomics_params).relative_to(REPO)) if radiomics and
        Path(radiomics_params).is_relative_to(REPO) else (str(radiomics_params) if radiomics else None),
        "min_baseline_cm3_for_percent": min_baseline_cm3, "same_grid_checks": grid_notes,
        "repo_commit": _git_commit(), "python": platform.python_version(), "seconds": round(time.time() - start, 1),
        "note": "Research prototype for decision support only; not a diagnosis.",
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    return summary


def _changes(scans, volumes, expert_volumes, min_baseline_cm3):
    """Change rows per patient with 2+ scans, from predictions and (when every scan has one) expert masks."""
    import nibabel as nib

    rows, notes = [], {}
    by_patient = {}
    for scan in scans:
        by_patient.setdefault(scan.patient, []).append(scan)
    for patient, group in by_patient.items():
        if len(group) < 2:
            continue
        group = sorted(group, key=lambda s: s.timepoint)
        ref = nib.load(group[0].files["t1ce"])
        notes[patient] = all(same_grid(ref, nib.load(s.files["t1ce"])) for s in group[1:])
        names = {s.timepoint: s.label or str(s.timepoint) for s in group}
        sources = [("prediction", volumes)] + ([("expert", expert_volumes)] if len(expert_volumes) == len(volumes) else [])
        for source, table in sources:
            vols = {r["timepoint"]: {m: r[f"{m}_cm3"] for m in CHANGE_MEASURES}
                    for r in table if r["patient"] == patient}
            for row in change_rows(vols, names=names, min_baseline_cm3=min_baseline_cm3, measures=CHANGE_MEASURES):
                rows.append({"patient": patient, "source": source, **row,
                             "same_grid": notes[patient]})
    return rows, notes
