"""Pipeline tests (M3-T05): a 3-scan patient goes through every step with a stand-in nnU-Net and radiomics
extractor; locked patients are refused; the command line and the raw-scan cleaning path work."""
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
from bmts.data.splits import LockboxError, check_subject
from bmts.pipeline.analyze import Scan, analyze
from bmts.uncertainty import nnunet_tta as T

SHAPE = (24, 22, 20)


class FakePredictor:
    """Stand-in for nnUNetPredictor: the class whose FLAIR intensity band (see write_scan) is closest."""
    allowed_mirroring_axes = (0, 1, 2)
    centres = torch.tensor([0.3, 1.9, 2.9, 3.9])        # background/brain, NCR, ED, ET

    def predict_logits_from_preprocessed_data(self, x):
        v = x[-1]
        return -4 * (v.unsqueeze(0) - self.centres.view(4, 1, 1, 1)) ** 2


def softmax(logits):
    return torch.softmax(torch.as_tensor(logits), 0).numpy()


class FakeExtractor:
    def execute(self, image, mask):
        import SimpleITK as sitk
        m = sitk.GetArrayFromImage(mask).astype(bool)
        return {"original_shape_VoxelVolume": float(m.sum()), "original_firstorder_Range": 8.0}


def write_scan(folder, wt_size, rng):
    folder.mkdir(parents=True)
    seg = np.zeros(SHAPE, np.uint8)
    lo, hi = 10 - wt_size // 2, 10 + wt_size // 2
    seg[lo:hi, lo:hi, 6:14] = 2
    seg[lo + 1:hi - 1, lo + 1:hi - 1, 8:12] = 1
    seg[10:12, 10:12, 9:11] = 3
    brain = np.zeros(SHAPE, bool)
    brain[3:21, 3:19, 2:18] = True
    flair = np.where(brain, 0.3 + rng.normal(0, 0.1, SHAPE), 0) + np.where(seg > 0, seg + 0.6, 0)
    for m in MODALITIES:
        nib.save(nib.Nifti1Image(flair.astype(np.float32), np.eye(4)), folder / f"{m}.nii.gz")
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "seg.nii.gz")


@pytest.fixture()
def fake_nnunet(monkeypatch):
    def fake_preprocess(predictor, files):
        import SimpleITK as sitk
        img = sitk.ReadImage(str(files[0]))
        data = np.stack([T.read_sitk(f, np.float32) for f in files])
        return data, {"sitk_stuff": {"spacing": img.GetSpacing(), "origin": img.GetOrigin(),
                                     "direction": img.GetDirection()}}
    monkeypatch.setattr(T, "load_predictor", lambda *a, **k: FakePredictor())
    monkeypatch.setattr(T, "preprocess", fake_preprocess)
    monkeypatch.setattr(T, "nnunet_probabilities", lambda predictor, logits, props: softmax(logits))


@pytest.fixture()
def data(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "processed"
    for tp, size in ((0, 12), (1, 6), (2, 10)):
        write_scan(root / "rhuh" / "RHUH-0002" / f"{tp}.0", size, rng)
    write_scan(root / "rhuh" / "RHUH-0009" / "0.0", 8, rng)
    rows = [{"dataset_source": "rhuh", "canonical_subject_id": "rhuh__RHUH-0002", "split": "train", "lockbox": "False"}
            for _ in range(3)]
    rows += [{"dataset_source": "rhuh", "canonical_subject_id": "rhuh__RHUH-0009", "split": "test", "lockbox": "True"},
             {"dataset_source": "africa", "canonical_subject_id": "africa__X", "split": "external", "lockbox": "True"}]
    pd.DataFrame(rows).to_csv(tmp_path / "split.csv", index=False)
    return root, tmp_path / "split.csv"


def test_split_guard(data):
    _, split_csv = data
    assert check_subject(split_csv, "rhuh", "RHUH-0002") == "train"
    for dataset, subject in (("rhuh", "RHUH-0009"), ("africa", "X"), ("rhuh", "RHUH-9999")):
        with pytest.raises(LockboxError):
            check_subject(split_csv, dataset, subject)


def test_three_scans_give_every_output(data, tmp_path, fake_nnunet):
    root, _ = data
    names = {0: "pre-op", 1: "early post-op", 2: "recurrence"}
    scans = [Scan.from_folder(root / "rhuh/RHUH-0002" / f"{tp}.0", scan_id=f"RHUH-0002_tp{tp}", timepoint=tp,
                              label=names[tp], patient="RHUH-0002") for tp in (0, 1, 2)]
    out = tmp_path / "out"
    summary = analyze(scans, out, FakePredictor(), extractor=FakeExtractor())
    assert set(summary["outputs"]) == {"volumes", "uncertainty", "radiomics", "metrics", "changes"}
    vols = pd.read_csv(out / "volumes.csv")
    assert len(vols) == 3 and (vols.ET_cm3 <= vols.TC_cm3).all() and (vols.TC_cm3 <= vols.WT_cm3).all()
    changes = pd.read_csv(out / "changes.csv")
    assert len(changes) == 3 * 3 * 2 and set(changes.source) == {"prediction", "expert"} and changes.same_grid.all()
    expert_wt = changes[(changes.source == "expert") & (changes.region == "WT") & (changes.tp_a == 0) & (changes.tp_b == 1)]
    assert expert_wt.pair.item() == "pre-op -> early post-op" and expert_wt.change_cm3.item() < 0
    metrics = pd.read_csv(out / "metrics.csv")
    assert metrics[["dice_WT", "dice_TC", "dice_ET"]].gt(0.5).all().all()     # fake model sees the bright tumor
    rad = pd.read_csv(out / "radiomics.csv")
    assert len(rad) == 9 and (rad.voxels == rad.shape_VoxelVolume).all()
    for tp in (0, 1, 2):
        for kind in ("seg", "uncertainty", "disagreement"):
            assert (out / f"RHUH-0002_tp{tp}" / f"RHUH-0002_tp{tp}_{kind}.nii.gz").is_file()
    saved = json.loads((out / "run_summary.json").read_text())
    assert saved["model"]["checkpoint"] == "checkpoint_best.pth" and saved["same_grid_checks"] == {"RHUH-0002": True}
    assert "decision support" in saved["note"]


@pytest.fixture()
def script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "analyze.py"
    spec = importlib.util.spec_from_file_location("analyze_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_command_line_refuses_locked_patient_and_runs_an_allowed_one(data, tmp_path, script, fake_nnunet, monkeypatch):
    root, split_csv = data
    base = ["analyze.py", "--dataset", "rhuh", "--processed-root", str(root), "--split-csv", str(split_csv),
            "--device", "cpu", "--no-radiomics"]
    monkeypatch.setattr(sys, "argv", base + ["--subject", "RHUH-0009", "--out", str(tmp_path / "locked")])
    with pytest.raises(LockboxError):
        script.main()
    assert not (tmp_path / "locked").exists()
    monkeypatch.setattr(sys, "argv", base + ["--subject", "RHUH-0002", "--out", str(tmp_path / "ok")])
    script.main()
    changes = pd.read_csv(tmp_path / "ok" / "changes.csv")
    assert set(changes.pair) == {"pre-op -> early post-op", "early post-op -> recurrence", "pre-op -> recurrence"}


def test_raw_scans_are_cleaned_first(tmp_path, script, fake_nnunet, monkeypatch):
    rng = np.random.default_rng(1)
    raw = tmp_path / "raw_exam"
    write_scan(raw, 10, rng)
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--scan", str(raw), "--raw", "--device", "cpu", "--no-radiomics",
                                      "--out", str(tmp_path / "out")])
    script.main()
    cleaned = nib.load(tmp_path / "out" / "cleaned" / "raw_exam" / "flair.nii.gz")
    assert cleaned.shape == (240, 240, 155)                          # M1-T04 grid
    assert len(pd.read_csv(tmp_path / "out" / "volumes.csv")) == 1
    assert not (tmp_path / "out" / "changes.csv").exists()          # one scan: nothing to compare


def test_raw_brats_masks_get_their_et_label_remapped(tmp_path, script, fake_nnunet, monkeypatch):
    rng = np.random.default_rng(2)
    raw = tmp_path / "raw_brats"
    write_scan(raw, 10, rng)
    seg_img = nib.load(raw / "seg.nii.gz")
    seg = np.asarray(seg_img.dataobj).copy()
    seg[seg == 3] = 4                                               # raw BraTS 2021 scheme: ET = 4
    nib.save(nib.Nifti1Image(seg, np.eye(4)), raw / "seg.nii.gz")
    monkeypatch.setattr(sys, "argv", ["analyze.py", "--scan", str(raw), "--raw", "--dataset", "brats2021", "--device",
                                      "cpu", "--no-radiomics", "--out", str(tmp_path / "out")])
    script.main()
    cleaned = np.asarray(nib.load(tmp_path / "out" / "cleaned" / "raw_brats" / "seg.nii.gz").dataobj)
    assert 4 not in np.unique(cleaned) and (cleaned == 3).sum() == (seg == 4).sum()
    assert pd.read_csv(tmp_path / "out" / "metrics.csv").ET_cm3.item() > 0     # expert ET counted as ET
