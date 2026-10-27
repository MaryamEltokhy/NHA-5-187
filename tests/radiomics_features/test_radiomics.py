"""Radiomics tests (M3-T03): settings, region masks, column names and quality notes with a stand-in extractor.
A last test runs real PyRadiomics when it is installed (environment/requirements-radiomics.txt)."""
import nibabel as nib
import numpy as np
import pytest
import yaml

from bmts.radiomics.features import DEFAULT_PARAMS, extract_case, load_params, quality_note, to_float


def write_case(tmp_path, with_et=True):
    seg = np.zeros((20, 20, 20), np.uint8)
    seg[5:15, 5:15, 5:15] = 2
    seg[8:12, 8:12, 8:12] = 1
    if with_et:
        seg[9:11, 9:11, 9:11] = 3
    rng = np.random.default_rng(0)
    files = {}
    for m in ("t1ce", "flair", "t2"):
        files[m] = tmp_path / f"{m}.nii.gz"
        nib.save(nib.Nifti1Image(rng.normal(size=seg.shape).astype(np.float32), np.eye(4)), files[m])
    nib.save(nib.Nifti1Image(seg, np.eye(4)), tmp_path / "seg.nii.gz")
    return files, tmp_path / "seg.nii.gz", seg


class FakeExtractor:
    """Returns the voxel count of the mask as a 'shape' feature and the image mean inside it as 'first-order'."""

    def execute(self, image, mask):
        import SimpleITK as sitk
        m = sitk.GetArrayFromImage(mask).astype(bool)
        img = sitk.GetArrayFromImage(image)
        return {"diagnostics_Versions": "x", "original_shape_VoxelVolume": float(m.sum()),
                "original_firstorder_Mean": float(img[m].mean()), "original_firstorder_Range": 2.0}


def test_settings_file_is_the_team_one():
    params = load_params()
    assert params["setting"]["binWidth"] == 0.25 and params["setting"]["normalize"] is False
    assert DEFAULT_PARAMS.name == "pyradiomics_params.yaml"


def test_missing_or_incomplete_settings_are_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_params(tmp_path / "nope.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"setting": {"label": 1}}))
    with pytest.raises(ValueError, match="binWidth"):
        load_params(bad)


def test_rows_per_region_and_column_names(tmp_path):
    files, seg_file, seg = write_case(tmp_path)
    rows = extract_case(files, seg_file, case_id="p1", extractor=FakeExtractor())
    by_region = {r["region"]: r for r in rows}
    assert set(by_region) == {"TC", "WT", "ET"}
    for region, labels in {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}.items():
        n = int(np.isin(seg, labels).sum())
        assert by_region[region]["voxels"] == n == by_region[region]["shape_VoxelVolume"]
    wt = by_region["WT"]
    assert {"t1ce_firstorder_Mean", "flair_firstorder_Mean", "t2_firstorder_Mean"} <= set(wt)
    assert not any(k.startswith(("diagnostics", "original_", "flair_shape")) for k in wt)
    assert by_region["ET"]["voxels"] <= by_region["TC"]["voxels"] <= by_region["WT"]["voxels"]
    assert "small region" in by_region["ET"]["quality_note"]           # 8 voxels
    assert "few gray-level bins" in wt["quality_note"]                  # range 2 / bin width 0.25 = 8 bins


def test_empty_region_is_skipped_with_a_reason(tmp_path):
    files, seg_file, _ = write_case(tmp_path, with_et=False)
    et = next(r for r in extract_case(files, seg_file, extractor=FakeExtractor()) if r["region"] == "ET")
    assert et["voxels"] == 0 and et["skip_reason"] == "empty_region"


def test_quality_note_and_to_float():
    assert quality_note({"voxels": 5000, "t2_firstorder_Range": 10.0}, 0.25) == ""
    assert "t2" in quality_note({"voxels": 5000, "t2_firstorder_Range": 2.0}, 0.25)
    assert to_float(np.float64(2.5)) == 2.5 and to_float(complex(3, 0)) == 3.0 and np.isnan(to_float(complex(3, 1)))


def test_real_pyradiomics(tmp_path):
    pytest.importorskip("radiomics.featureextractor")
    files, seg_file, seg = write_case(tmp_path)
    rows = extract_case(files, seg_file, case_id="p1")
    wt = next(r for r in rows if r["region"] == "WT")
    assert wt["skip_reason"] == "" and wt["shape_VoxelVolume"] == pytest.approx(np.isin(seg, (1, 2, 3)).sum())
    assert any(k.startswith("t2_glcm_") for k in wt)
