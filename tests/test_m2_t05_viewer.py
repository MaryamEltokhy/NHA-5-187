from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("streamlit")  # the viewer needs environment/requirements-viewer.txt

from apps.m2_t05_viewer import Case, discover_cases, normalize_for_display, render_slice


def _make_case(tmp_path: Path) -> Case:
    case_dir = tmp_path / "brats2021" / "P001" / "tp00"
    case_dir.mkdir(parents=True)
    # Discovery checks file presence only; actual NIfTI parsing is tested separately in the real environment.
    for name in ("t1", "t1ce", "t2", "flair", "seg"):
        (case_dir / f"{name}.nii.gz").write_bytes(b"placeholder")
    return Case("brats2021", "P001", "tp00", case_dir)


def test_discover_cases_finds_complete_case(tmp_path):
    case = _make_case(tmp_path)
    assert discover_cases(tmp_path) == [case]


def test_discover_cases_skips_incomplete_case(tmp_path):
    case_dir = tmp_path / "brats2021" / "P001" / "tp00"
    case_dir.mkdir(parents=True)
    for name in ("t1", "t1ce", "t2", "flair"):
        (case_dir / f"{name}.nii.gz").write_bytes(b"placeholder")
    assert discover_cases(tmp_path) == []


def test_normalize_for_display_returns_unit_range():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    vol[1:, 1:, 1:] = np.arange(27, dtype=np.float32).reshape(3, 3, 3) + 1
    out = normalize_for_display(vol)
    assert out.dtype == np.float32
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


def test_render_slice_accepts_aligned_image_and_mask():
    image = np.random.default_rng(0).normal(size=(16, 16, 8)).astype(np.float32)
    mask = np.zeros((16, 16, 8), dtype=np.int16)
    mask[4:8, 4:8, 3] = 3
    fig = render_slice(image, mask, 3)
    try:
        assert fig.axes
        assert len(fig.axes[0].images) == 2
    finally:
        fig.clear()
