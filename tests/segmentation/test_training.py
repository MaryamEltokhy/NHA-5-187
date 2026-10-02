"""Training tests (M2-T03): scoring conventions, and a tiny end-to-end run with MLflow, resume and pause."""
import math

import mlflow
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from mlflow.tracking import MlflowClient

from bmts.common.constants import MODALITIES
from bmts.segmentation.training.metrics import region_scores, summarize
from bmts.segmentation.training.trainer import load_experiment, train

SHAPE = (40, 36, 30)
SPLIT = {"BraTS2021_00001": "train", "BraTS2021_00002": "train", "BraTS2021_00003": "train",
         "BraTS2021_00004": "val", "BraTS2021_00005": "test"}


# ---------- scoring ----------

def _masks(*boxes):
    out = torch.zeros(3, 10, 10, 10)
    for c, box in enumerate(boxes):
        if box:
            out[c][box] = 1
    return out


def test_scores_follow_the_team_conventions():
    full = (slice(2, 6), slice(2, 6), slice(2, 6))
    half = (slice(2, 4), slice(2, 6), slice(2, 6))
    label = _masks(full, full, None)          # TC, WT present; ET empty in the ground truth
    pred = _masks(half, full, None)           # TC half found, WT perfect, ET correctly empty
    s = region_scores(pred, label)
    assert s["TC"]["dice"] == pytest.approx(2 * 32 / (32 + 64))
    assert s["TC"]["iou"] == pytest.approx(32 / 64)
    assert s["TC"]["sensitivity"] == pytest.approx(0.5) and s["TC"]["precision"] == pytest.approx(1.0)
    assert s["WT"]["dice"] == 1.0 and s["WT"]["hd95"] == 0.0
    assert s["ET"]["dice"] == 1.0 and s["ET"]["hd95"] == 0.0          # both empty: perfect
    assert math.isnan(s["ET"]["sensitivity"]) and math.isnan(s["ET"]["precision"])

    missed = region_scores(_masks(None, full, None), label)          # TC missed completely
    assert missed["TC"]["dice"] == 0.0 and math.isnan(missed["TC"]["hd95"])


def test_summary_skips_nan_and_counts_it():
    full = (slice(2, 6), slice(2, 6), slice(2, 6))
    a = region_scores(_masks(full, full, full), _masks(full, full, full))
    b = region_scores(_masks(None, full, full), _masks(full, full, full))
    summary = summarize([a, b])
    assert summary["TC"]["dice"] == pytest.approx(0.5)
    assert summary["TC"]["hd95"] == 0.0 and summary["TC"]["hd95_nan"] == 1
    assert summary["mean_dice"] == pytest.approx((0.5 + 1 + 1) / 3)


# ---------- tiny end-to-end training ----------

def _write_case(root, subject, rng):
    folder = root / "brats2021" / subject / "0"
    folder.mkdir(parents=True)
    brain = np.zeros(SHAPE, bool)
    brain[5:35, 5:31, 4:26] = True
    seg = np.zeros(SHAPE, np.uint8)
    seg[12:24, 12:24, 10:20] = 2
    seg[15:21, 15:21, 12:18] = 1
    seg[17:19, 17:19, 14:16] = 3
    for k in MODALITIES:  # tumor brighter than brain, so a tiny network can learn something
        img = np.where(brain, rng.normal(size=SHAPE), 0) + 2.0 * (seg > 0)
        nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), folder / f"{k}.nii.gz")
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "seg.nii.gz")


@pytest.fixture()
def setup(tmp_path):
    rng = np.random.default_rng(0)
    for subject in SPLIT:
        _write_case(tmp_path / "processed", subject, rng)
    pd.DataFrame([{"dataset_source": "brats2021", "canonical_subject_id": f"brats2021__{s}", "split": sp,
                   "lockbox": str(sp == "test"), "split_version": "v1_test"} for s, sp in SPLIT.items()]
                 ).to_csv(tmp_path / "split_v1.csv", index=False)
    yield tmp_path
    if mlflow.active_run():
        mlflow.end_run()


def _config(tmp, epochs=2):
    cfg = load_experiment("configs/training/EXP-0001.yaml", processed_root=str(tmp / "processed"),
                          split_csv=str(tmp / "split_v1.csv"), output_dir=str(tmp / "runs" / "EXP-test"),
                          tracking_uri=f"sqlite:///{(tmp / 'mlflow.db').as_posix()}", epochs=epochs,
                          iters_per_epoch=2, val_every=1, val_cases_during_training=1, num_workers=0,
                          smart_cache_num=2, model={"channels": [4, 8], "strides": [2], "num_res_units": 1})
    cfg["data"]["roi_size"] = [16, 16, 16]
    cfg["exp"]["mlflow"]["experiment"] = "test_experiment"
    cfg["exp"]["training"]["warmup_epochs"] = 1
    return cfg


def test_training_logs_everything_and_resumes(setup):
    result = train(_config(setup, epochs=2), log=lambda *_: None)
    assert result["finished"] and result["val_cases"] == 1
    out = setup / "runs" / "EXP-test"
    for name in ("last.pt", "best.pt", "final_val_metrics.json", "final_val_per_case.csv", "config_resolved.yaml"):
        assert (out / name).exists(), name
    per_case = pd.read_csv(out / "final_val_per_case.csv")
    assert per_case.case.tolist() == ["brats2021__BraTS2021_00004/0"]          # the locked test patient is never read

    client = MlflowClient(tracking_uri=f"sqlite:///{(setup / 'mlflow.db').as_posix()}")
    run = client.search_runs([client.get_experiment_by_name("test_experiment").experiment_id])[0]
    assert run.data.tags["state"] == "finished" and run.data.tags["task"] == "M2-T03"
    assert run.data.params["train_cases"] == "3" and run.data.tags["split_version"] == "v1_test"
    for key in ("train_loss", "val_dice_TC", "val_dice_WT", "val_dice_ET", "final_val_dice_WT", "final_val_hd95_WT"):
        assert key in run.data.metrics, key
    assert len(client.get_metric_history(run.info.run_id, "train_loss")) == 2

    resumed = train(_config(setup, epochs=3), log=lambda *_: None)        # more epochs: continues from epoch 2
    assert resumed["finished"]
    history = client.get_metric_history(run.info.run_id, "train_loss")
    assert [m.step for m in history] == [1, 2, 3]                          # same MLflow run, no epoch repeated
    assert len(client.search_runs([run.info.experiment_id])) == 1


def test_time_budget_pauses_and_a_rerun_continues(setup):
    paused = train(_config(setup, epochs=3), max_hours=1e-9, log=lambda *_: None)
    assert not paused["finished"] and paused["epoch"] == 1
    client = MlflowClient(tracking_uri=f"sqlite:///{(setup / 'mlflow.db').as_posix()}")
    run = client.search_runs([client.get_experiment_by_name("test_experiment").experiment_id])[0]
    assert run.info.status == "KILLED" and run.data.tags["state"].startswith("paused")
    done = train(_config(setup, epochs=3), log=lambda *_: None)
    assert done["finished"]
    assert client.get_run(run.info.run_id).data.tags["state"] == "finished"
