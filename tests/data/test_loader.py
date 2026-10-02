"""Data loader tests (M1-T05) on tiny synthetic cases in the cleaned layout of M1-T04."""
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from bmts.common.constants import MODALITIES
from bmts.data.augmentation.transforms import SegToRegionsd
from bmts.data.datasets.brats import LockboxError, build_loaders, list_cases

SHAPE = (40, 36, 30)
ROI = [16, 16, 16]
SPLIT = {"BraTS2021_00001": "train", "BraTS2021_00002": "train", "BraTS2021_00003": "train",
         "BraTS2021_00004": "val", "BraTS2021_00005": "test"}


def write_case(root, subject, rng):
    folder = root / "brats2021" / subject / "0"
    folder.mkdir(parents=True)
    brain = np.zeros(SHAPE, bool)
    brain[5:35, 5:31, 4:26] = True
    for k in MODALITIES:  # z-scored brain, background exactly 0, like the cleaning step writes
        nib.save(nib.Nifti1Image(np.where(brain, rng.normal(size=SHAPE), 0).astype(np.float32), np.eye(4)),
                 folder / f"{k}.nii.gz")
    seg = np.zeros(SHAPE, np.uint8)
    seg[12:24, 12:24, 10:20] = 2   # edema
    seg[15:21, 15:21, 12:18] = 1   # necrosis
    seg[17:19, 17:19, 14:16] = 3   # enhancing tumor
    nib.save(nib.Nifti1Image(seg, np.eye(4)), folder / "seg.nii.gz")


@pytest.fixture()
def data(tmp_path):
    rng = np.random.default_rng(0)
    for subject in SPLIT:
        write_case(tmp_path / "processed", subject, rng)
    rows = [{"dataset_source": "brats2021", "canonical_subject_id": f"brats2021__{s}", "split": sp,
             "lockbox": str(sp == "test")} for s, sp in SPLIT.items()]
    split_csv = tmp_path / "split_v1.csv"
    pd.DataFrame(rows).to_csv(split_csv, index=False)
    return tmp_path / "processed", split_csv


def config(processed, split_csv):
    return {"processed_root": str(processed), "split_csv": str(split_csv), "datasets": ["brats2021"],
            "train_split": "train", "val_split": "val", "on_missing": "error", "normalize": False,
            "roi_size": ROI, "batch_size": 2, "num_workers": 0, "cache_rate": 0.0,
            "sampling": {"pos": 1, "neg": 1},
            "augment": {"flip_prob": 0.5, "affine_prob": 0.5, "rotate_deg": 15, "scale": 0.1,
                        "intensity_prob": 0.5, "gamma_prob": 0.5, "elastic_prob": 0.0}}


def test_lists_only_the_requested_split(data):
    cases = list_cases(data[1], data[0], splits=("train",))
    assert [c["id"] for c in cases] == [f"brats2021__BraTS2021_0000{i}/0" for i in (1, 2, 3)]
    assert set(cases[0]) == {"id", *MODALITIES, "seg"}


def test_locked_test_patients_are_refused(data):
    with pytest.raises(LockboxError):
        list_cases(data[1], data[0], splits=("test",))
    assert len(list_cases(data[1], data[0], splits=("test",), allow_lockbox=True)) == 1


def test_lockbox_flag_is_enforced_even_outside_the_test_split(data, tmp_path):
    table = pd.read_csv(data[1], dtype=str)
    table.loc[0, "lockbox"] = "True"   # a training patient marked as locked
    table.to_csv(tmp_path / "bad_split.csv", index=False)
    with pytest.raises(LockboxError):
        list_cases(tmp_path / "bad_split.csv", data[0], splits=("train",))


def test_missing_files_stop_or_are_skipped(data):
    (data[0] / "brats2021" / "BraTS2021_00002" / "0" / "flair.nii.gz").unlink()
    with pytest.raises(FileNotFoundError, match="missing flair"):
        list_cases(data[1], data[0], splits=("train",))
    with pytest.warns(UserWarning):
        cases = list_cases(data[1], data[0], splits=("train",), on_missing="skip")
    assert len(cases) == 2


def test_limit_keeps_the_first_patients(data):
    assert len(list_cases(data[1], data[0], splits=("train",), limit=2)) == 2


def test_regions_follow_the_canonical_labels():
    seg = torch.tensor([0, 1, 2, 3, 4]).reshape(1, 5, 1, 1).float()
    out = SegToRegionsd()({"seg": seg})
    tc, wt, et = out["label"][:, :, 0, 0]
    assert tc.tolist() == [0, 1, 0, 1, 0]   # necrosis + enhancing
    assert wt.tolist() == [0, 1, 1, 1, 0]   # + edema; resection cavity (4) excluded
    assert et.tolist() == [0, 0, 0, 1, 0]
    assert out["fg"][0, :, 0, 0].tolist() == wt.tolist() and "seg" not in out


def test_loaders_give_the_model_what_it_expects(data):
    train_loader, val_loader = build_loaders(config(*data))
    for _ in range(2):  # two epochs, fresh random patches each time
        for batch in train_loader:
            assert batch["image"].shape[1:] == (4, *ROI) and batch["image"].dtype == torch.float32
            assert batch["label"].shape[1:] == (3, *ROI)
            assert set(batch["label"].unique().tolist()) <= {0.0, 1.0}
            assert torch.isfinite(batch["image"]).all()
    val = next(iter(val_loader))
    assert val["image"].shape == (1, 4, *SHAPE) and val["label"].shape == (1, 3, *SHAPE)
    assert val["label"][0, 1].sum() == 12 * 12 * 10   # whole tumor = the edema box
    assert len(val_loader.dataset) == 1                 # the test patient never appears


def test_broken_intensities_are_sanitized(data):
    """A scan with NaN and absurd values (e.g. a near-constant modality divided by a tiny std) must not reach the model."""
    path = data[0] / "brats2021" / "BraTS2021_00004" / "0" / "t1.nii.gz"
    img = nib.load(path)
    arr = np.asarray(img.dataobj).copy()
    arr[10, 10, 10], arr[11, 11, 11], arr[12, 12, 12] = np.nan, 1e9, -1e9
    nib.save(nib.Nifti1Image(arr, img.affine), path)
    _, val_loader = build_loaders(config(*data))
    image = next(iter(val_loader))["image"]
    assert torch.isfinite(image).all() and image.abs().max() <= 20
