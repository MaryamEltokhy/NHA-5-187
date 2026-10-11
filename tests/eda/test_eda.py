"""EDA tests: per-scan numbers on hand-made masks, and a tiny end-to-end run of scripts/eda.py."""
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from bmts.common.constants import MODALITIES
from bmts.eda.stats import best_axial_slice, intensity_stats, nonzero_percentiles, tumor_stats

SHAPE = (40, 36, 30)


def _seg(et=True, second_part=False):
    seg = np.zeros(SHAPE, np.uint8)
    seg[10:20, 10:20, 10:20] = 2            # 1000 voxels edema
    seg[13:17, 13:17, 13:17] = 1            # 64 voxels necrosis inside it
    if et:
        seg[14:16, 14:16, 14:16] = 3        # 8 voxels enhancing
    if second_part:
        seg[30:35, 28:33, 22:27] = 2        # 125 voxels, far away
    return seg


# ---------- per-scan numbers ----------

def test_tumor_volumes_and_composition():
    s = tumor_stats(_seg(), spacing=(1, 1, 1))
    assert s["vol_wt_ml"] == pytest.approx(1.0)                       # 1000 voxels of 1 mm³
    assert s["vol_tc_ml"] == pytest.approx(0.064)                     # NCR 56 + ET 8
    assert s["vol_et_ml"] == pytest.approx(0.008) and s["et_present"]
    assert s["share_ed_of_wt"] + s["share_ncr_of_wt"] + s["share_et_of_wt"] == pytest.approx(1.0)
    assert s["wt_components"] == 1 and s["wt_small_components"] == 0
    assert s["extent_x_mm"] == 10 and not s["touches_border"]
    assert s["centroid_x"] == pytest.approx(14.5 / (SHAPE[0] - 1))


def test_spacing_parts_and_missing_et():
    s = tumor_stats(_seg(et=False, second_part=True), spacing=(2, 1, 1))
    assert s["vol_wt_ml"] == pytest.approx(1125 * 2 / 1000)
    assert not s["et_present"] and s["vol_et_ml"] == 0
    assert s["wt_components"] == 2
    tiny = _seg()
    tiny[0, 0, 0] = 2                                                  # one stray voxel at the edge
    t = tumor_stats(tiny)
    assert t["wt_components"] == 1 and t["wt_small_components"] == 1 and t["touches_border"]


def test_empty_mask_gives_nan_position():
    s = tumor_stats(np.zeros(SHAPE, np.uint8))
    assert s["vol_wt_ml"] == 0 and s["wt_components"] == 0 and np.isnan(s["centroid_x"])


def test_intensity_per_tissue():
    seg = _seg()
    brain = np.zeros(SHAPE, bool)
    brain[5:35, 5:31, 4:26] = True
    images = {m: np.where(brain, 0.0, 0.0) + np.where(seg == 2, 3.0, 0) + np.where(seg == 3, 5.0, 0) for m in MODALITIES}
    images["t1"] = np.where(brain, -1.0, 0.0)                         # t1 alone defines the brain mask
    row, hist = intensity_stats(images, seg)
    assert row["brain_ml"] == pytest.approx(brain.sum() / 1000)
    assert row["wt_share_of_brain"] == pytest.approx(1000 / brain.sum())
    assert row["mean_flair_ed"] == pytest.approx(3.0) and row["mean_flair_et"] == pytest.approx(5.0)
    assert hist["flair"]["et"].sum() == 8 and hist["flair"]["healthy"].sum() == brain.sum() - 1000


def test_helpers():
    seg = _seg()
    assert best_axial_slice(seg > 0) in range(10, 20)
    assert best_axial_slice(np.zeros(SHAPE, bool)) == SHAPE[2] // 2
    img = np.zeros(SHAPE, np.float32)
    img[5:35, 5:31, 4:26] = np.linspace(100, 200, 30)[:, None, None]
    p = nonzero_percentiles(img)
    assert 100 <= p["p1"] < p["p50"] < p["p99"] <= 200


# ---------- tiny end-to-end run ----------

SPLIT = {"BraTS2021_00001": "train", "BraTS2021_00002": "train", "BraTS2021_00003": "train",
         "BraTS2021_00004": "val", "BraTS2021_00005": "val", "BraTS2021_00006": "test"}


def _write(folder, seg, rng, scale=1.0):
    folder.mkdir(parents=True)
    brain = np.zeros(SHAPE, bool)
    brain[5:35, 5:31, 4:26] = True
    for m in MODALITIES:
        img = np.where(brain, rng.normal(size=SHAPE) * scale + 4 * scale * (seg > 0), 0)
        nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), folder / f"{m}.nii.gz")
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "seg.nii.gz")


@pytest.fixture()
def eda_script(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "eda.py"
    spec = importlib.util.spec_from_file_location("eda_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "CANONICAL_SHAPE", SHAPE)       # small scans stand in for 240 x 240 x 155
    return module


def test_eda_end_to_end(tmp_path, eda_script, monkeypatch):
    rng = np.random.default_rng(0)
    raw_dir = tmp_path / "raw"
    for i, (subject, split) in enumerate(SPLIT.items()):
        seg = _seg(et=i != 1, second_part=i == 2)
        if split == "test":
            continue                                             # locked patient: no files, must never be asked for
        _write(tmp_path / "processed" / "brats2021" / subject / "0", seg, rng)
        _write(raw_dir / subject, seg, rng, scale=100.0 * (i + 1))
    with tarfile.open(tmp_path / "raw.tar", "w") as tar:          # raw archive layout: ./<subject>/<subject>_<mod>.nii.gz
        for subject_dir in raw_dir.iterdir():
            for m in MODALITIES:
                tar.add(subject_dir / f"{m}.nii.gz", arcname=f"./{subject_dir.name}/{subject_dir.name}_{m}.nii.gz")
    pd.DataFrame([{"dataset_source": "brats2021", "canonical_subject_id": f"brats2021__{s}", "split": sp,
                   "lockbox": str(sp == "test")} for s, sp in SPLIT.items()]
                 + [{"dataset_source": "africa", "canonical_subject_id": "africa__X", "split": "external", "lockbox": "True"}]
                 ).to_csv(tmp_path / "split.csv", index=False)
    pd.DataFrame([{"canonical_subject_id": f"brats2021__{s}", "dataset_source": "brats2021", "split": sp,
                   **{f"{m}_available": "1" for m in ("T1", "T1ce", "T2", "FLAIR")}, "segmentation_available": "1"}
                  for s, sp in SPLIT.items()]).to_csv(tmp_path / "index.csv", index=False)
    pd.DataFrame([{"case": s, "dice_WT": 0.9, "dice_TC": 0.8, "dice_ET": 0.7} for s, sp in SPLIT.items() if sp == "val"]
                 ).to_csv(tmp_path / "scores.csv", index=False)

    out = tmp_path / "eda"
    monkeypatch.setattr(sys, "argv", [
        "eda.py", "--out", str(out), "--split-csv", str(tmp_path / "split.csv"), "--index-csv", str(tmp_path / "index.csv"),
        "--processed-root", str(tmp_path / "processed"), "--datasets", "brats2021", "--intensity-sample", "3",
        "--raw-tar", str(tmp_path / "raw.tar"), "--raw-sample", "2", "--scores-csv", str(tmp_path / "scores.csv"),
        "--workers", "1"])
    eda_script.main()

    tumor = pd.read_csv(out / "tumor_per_scan.csv")
    assert len(tumor) == 5 and not tumor.id.str.contains("00006").any()     # the test patient was never read
    summary = json.loads((out / "summary.json").read_text())
    assert summary["patients"]["africa"]["external"] == 1                   # counted, not opened
    assert summary["tumor"]["brats2021"]["no_et"]["n"] == 1
    assert summary["tumor"]["brats2021"]["multifocal"]["n"] == 1
    assert summary["intensity"]["brats2021"]["sample_scans"] == 3
    assert set(summary["train_vs_val_ks_p"]) == {"vol_wt_ml", "vol_tc_ml", "vol_et_ml"}
    raw = pd.read_csv(out / "raw_vs_clean.csv")
    assert len(raw) == 2 * 4 * 2 and raw[raw.stage == "raw"].p99.max() > 50  # raw scale kept, cleaned is ~z-score
    figures = {p.stem for p in (out / "figures").glob("*.png")}
    assert {"f01_cohorts", "f03_examples", "f04_multiplanar", "f05_volumes", "f06_imbalance", "f08_location",
            "f10_intensity_by_tissue", "f11_raw_vs_clean", "f12_train_vs_val", "f13_volume_vs_dice",
            "f14_difficult_cases"} <= figures
