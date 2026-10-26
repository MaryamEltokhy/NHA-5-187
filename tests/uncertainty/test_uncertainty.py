"""Confidence-map tests (M3-T04): the math on hand-made probabilities, the flip handling with a fake nnU-Net
predictor, and a tiny end-to-end run of scripts/uncertainty.py (no GPU, no real weights needed)."""
import importlib.util
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from bmts.common.constants import MODALITIES
from bmts.uncertainty import nnunet_tta as T
from bmts.uncertainty.maps import (CalibrationCounter, ErrorDetectionCounter, TTAAccumulator, auroc_from_histograms,
                                   binary_entropy, case_summary, entropy, region_probabilities)


def onehot(labels, sure=0.97):
    """Class probabilities (4, ...) that put `sure` on the given label map."""
    probs = np.full((4, *labels.shape), (1 - sure) / 3, np.float32)
    for c in range(4):
        probs[c][labels == c] = sure
    return probs


# ---------- math ----------

def test_entropy_limits():
    assert entropy(np.full((4, 2), 0.25))[0] == pytest.approx(1.0)
    assert entropy(onehot(np.zeros(3, int), sure=1.0)).max() < 1e-6
    assert binary_entropy(np.array([0.5]))[0] == pytest.approx(1.0) and binary_entropy(np.array([1.0]))[0] < 1e-6
    sure = binary_entropy(np.array([0.0, 1.0], dtype=np.float32))    # model output is float32; must not give NaN
    assert np.isfinite(sure).all() and sure.max() < 1e-6


def test_region_probabilities_follow_regions():
    probs = np.array([0.1, 0.2, 0.3, 0.4]).reshape(4, 1)
    rp = region_probabilities(probs)
    assert rp["WT"][0] == pytest.approx(0.9) and rp["TC"][0] == pytest.approx(0.6) and rp["ET"][0] == pytest.approx(0.4)


def test_agreeing_passes_give_no_disagreement():
    labels = np.zeros((6, 6, 6), int)
    labels[2:4, 2:4, 2:4] = 3
    acc = TTAAccumulator()
    for _ in range(8):
        acc.add(onehot(labels))
    maps = acc.finalize(labels)
    assert maps["disagreement"].max() < 1e-5
    assert maps["flip_rate_ET"].max() == 0


def test_contradicting_passes_show_up():
    a, b = np.zeros((4, 4, 4), int), np.zeros((4, 4, 4), int)
    b[0, 0, 0] = 2                                   # half the passes call one voxel edema
    acc = TTAAccumulator()
    for k in range(8):
        acc.add(onehot(a if k % 2 else b))
    maps = acc.finalize(a)
    assert maps["disagreement"][0, 0, 0] > 0.2 and maps["disagreement"][1, 1, 1] < 1e-5
    assert maps["flip_rate_WT"][0, 0, 0] == pytest.approx(0.5)
    assert maps["uncertainty"][0, 0, 0] > maps["uncertainty"][1, 1, 1]
    row = case_summary(a, maps)
    assert row["vol_WT_ml"] == 0 and row["uncertain_WT_ml"] == pytest.approx(0.001)


def test_calibration_and_error_detection_counters():
    rng = np.random.default_rng(0)
    p_et = rng.uniform(0, 1, 200_000)
    truth = (rng.uniform(0, 1, p_et.size) < p_et).astype(int) * 3      # ET exactly as often as predicted
    probs = np.stack([1 - p_et, 0 * p_et, 0 * p_et, p_et]).astype(np.float32)
    mask = np.ones(p_et.size, bool)
    cal = CalibrationCounter()
    cal.add(probs, truth, mask)
    assert cal.ece("ET") < 0.01                                        # calibrated by construction
    over = probs.copy()
    over[3], over[0] = np.where(p_et > 0.5, 1.0, 0.0), np.where(p_et > 0.5, 0.0, 1.0)
    cal_over = CalibrationCounter()
    cal_over.add(over, truth, mask)
    assert cal_over.ece("ET") > 0.2                                    # always 0 or 1: overconfident

    det = ErrorDetectionCounter()
    seg = np.where(p_et > 0.5, 3, 0)
    det.add(probs, seg, truth, mask)
    assert det.auroc("ET") > 0.6                                       # mistakes sit where p is near 0.5
    assert auroc_from_histograms([0, 0, 5], [5, 0, 0]) == 1.0
    assert auroc_from_histograms([1, 1], [1, 1]) == pytest.approx(0.5)


# ---------- flips with a fake nnU-Net predictor ----------

class FakePredictor:
    """Stands in for nnUNetPredictor: logits from the image, optionally depending on position (not flip-proof)."""

    allowed_mirroring_axes = (0, 1, 2)

    def __init__(self, position_bias=0.0):
        self.position_bias, self.calls = position_bias, 0

    def predict_logits_from_preprocessed_data(self, x):
        self.calls += 1
        v = x[-1]                                                      # last channel (FLAIR) drives the classes
        ramp = torch.linspace(-1, 1, v.shape[-1]).expand_as(v)        # position-dependent part
        logits = torch.stack([-v, v - 1, v - 2 + self.position_bias * ramp, v - 3 + self.position_bias * ramp])
        return logits * 4


def softmax(logits):
    return torch.softmax(torch.as_tensor(logits), 0).numpy()


def test_flip_combinations_cover_nnunet_mirroring():
    combos = T.flip_combinations((0, 1, 2))
    assert len(combos) == 8 and combos[0] == () and (0, 1, 2) in combos
    assert T.flip_combinations(None) == [()]


def test_flips_are_undone_before_averaging():
    rng = np.random.default_rng(1)
    data = rng.uniform(0, 4, (4, 6, 7, 8)).astype(np.float32)
    fake = FakePredictor()                                             # pointwise: every pass must agree exactly
    seg, maps, n = T.predict_case_tta(fake, data, None, to_probabilities=softmax)
    assert n == 8 and fake.calls == 8
    direct = softmax(fake.predict_logits_from_preprocessed_data(torch.from_numpy(data))).argmax(0)
    assert np.array_equal(seg, direct)
    assert maps["disagreement"].max() < 1e-4

    biased = FakePredictor(position_bias=1.5)                          # passes now differ: the map must show it
    _, maps_b, _ = T.predict_case_tta(biased, data, None, to_probabilities=softmax)
    assert maps_b["disagreement"].max() > 0.01


# ---------- tiny end-to-end run of scripts/uncertainty.py ----------

SHAPE = (24, 22, 20)
SPLIT = {"BraTS2021_00001": "train", "BraTS2021_00002": "val", "BraTS2021_00003": "val", "BraTS2021_00004": "val",
         "BraTS2021_00005": "test"}


def _write_case(root, subject, rng, shift):
    folder = root / "brats2021" / subject / "0"
    folder.mkdir(parents=True)
    seg = np.zeros(SHAPE, np.uint8)
    seg[8:16, 8:16, 6:14] = 2
    seg[10:14, 10:14, 8:12] = 1
    seg[11:13, 11:13, 9:11] = 3
    flair = np.where(seg > 0, seg.astype(np.float32) + 0.5 + shift, 0.3) + rng.normal(0, 0.2, SHAPE)
    for m in MODALITIES:
        nib.save(nib.Nifti1Image((flair if m == "flair" else flair * 0.5).astype(np.float32), np.eye(4)),
                 folder / f"{m}.nii.gz")
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "seg.nii.gz")


@pytest.fixture()
def script(monkeypatch):
    path = Path(__file__).resolve().parents[2] / "scripts" / "uncertainty.py"
    spec = importlib.util.spec_from_file_location("uncertainty_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fake_preprocess(predictor, files):                             # nnU-Net preprocessing stand-in
        import SimpleITK as sitk
        img = sitk.ReadImage(str(files[0]))
        data = np.stack([T.read_sitk(f, np.float32) for f in files])
        return data, {"sitk_stuff": {"spacing": img.GetSpacing(), "origin": img.GetOrigin(),
                                     "direction": img.GetDirection()}}

    monkeypatch.setattr(T, "load_predictor", lambda *a, **k: FakePredictor(position_bias=0.8))
    monkeypatch.setattr(T, "preprocess", fake_preprocess)
    monkeypatch.setattr(T, "nnunet_probabilities", lambda predictor, logits, props: softmax(logits))
    return module


def test_script_end_to_end_and_resume(tmp_path, script, monkeypatch):
    rng = np.random.default_rng(2)
    for i, (subject, split) in enumerate(SPLIT.items()):
        if split != "test":                                            # locked patient has no files at all
            _write_case(tmp_path / "processed", subject, rng, shift=0.3 * i)
    pd.DataFrame([{"dataset_source": "brats2021", "canonical_subject_id": f"brats2021__{s}", "split": sp,
                   "lockbox": str(sp == "test")} for s, sp in SPLIT.items()]).to_csv(tmp_path / "split.csv", index=False)
    out = tmp_path / "out"
    argv = ["uncertainty.py", "--split-csv", str(tmp_path / "split.csv"), "--processed-root", str(tmp_path / "processed"),
            "--out", str(out), "--device", "cpu", "--save-maps", "1"]

    monkeypatch.setattr(sys, "argv", argv + ["--cases", "2"])          # first session: only 2 of the 3 patients
    script.main()
    assert len(pd.read_csv(out / "per_case.csv")) == 2
    assert json.loads((out / "summary.json").read_text())["cases"] == 2

    monkeypatch.setattr(sys, "argv", argv)                             # rerun continues with the third
    script.main()
    table = pd.read_csv(out / "per_case.csv")
    assert sorted(table.case) == ["BraTS2021_00002", "BraTS2021_00003", "BraTS2021_00004"]
    assert (table.passes == 8).all() and table.dice_WT.between(0, 1).all()
    summary = json.loads((out / "summary.json").read_text())
    assert summary["cases"] == 3 and set(summary["ece"]) == {"TC", "WT", "ET"}
    assert {p.stem for p in (out / "figures").glob("*.png")} == {"f01_examples", "f02_reliability",
                                                                  "f03_uncertainty_vs_dice", "f04_error_detection"}
    maps = sorted(p.name for p in (out / "maps").iterdir())
    assert maps == ["BraTS2021_00002_disagreement.nii.gz", "BraTS2021_00002_seg.nii.gz",
                    "BraTS2021_00002_uncertainty.nii.gz"]
    written = nib.load(out / "maps" / "BraTS2021_00002_uncertainty.nii.gz")
    assert written.shape == SHAPE and 0 <= written.get_fdata().min() and written.get_fdata().max() <= 1
