"""Tumor volume tests (M3-T01): counts checked by hand on tiny synthetic label maps."""
import numpy as np
import nibabel as nib

from bmts.quantitative.volumes import region_volumes, voxel_volume_mm3, volumes_from_nifti, volumes_table


def make_seg():
    seg = np.zeros((10, 10, 10), dtype=np.uint8)
    seg[0:2, 0:2, 0:2] = 3      # 8 voxels of ET
    seg[3:6, 0, 0] = 1          # 3 voxels of NCR
    seg[0, 5:10, 0] = 2         # 5 voxels of edema
    return seg


def test_voxel_volume():
    assert voxel_volume_mm3((1, 1, 1)) == 1.0
    assert voxel_volume_mm3((2, 2, 2)) == 8.0
    assert voxel_volume_mm3((1.0, 1.0, 1.5)) == 1.5


def test_counts_and_nesting_one_mm_voxels():
    v = region_volumes(make_seg(), (1.0, 1.0, 1.0))
    assert v["ET"]["voxels"] == 8
    assert v["TC"]["voxels"] == 8 + 3          # labels 1 and 3
    assert v["WT"]["voxels"] == 8 + 3 + 5      # labels 1, 2 and 3
    assert v["ET"]["cm3"] == 0.008 and v["WT"]["cm3"] == 0.016


def test_voxel_size_changes_the_volume():
    v = region_volumes(make_seg(), (2.0, 2.0, 2.0))   # 8 mm3 per voxel
    assert v["ET"]["mm3"] == 64.0 and v["ET"]["cm3"] == 0.064


def test_empty_mask_is_zero():
    v = region_volumes(np.zeros((5, 5, 5), dtype=np.uint8), (1, 1, 1))
    assert all(r["voxels"] == 0 and r["cm3"] == 0.0 for r in v.values())


def test_reads_voxel_size_from_the_file_header(tmp_path):
    img = nib.Nifti1Image(make_seg(), np.diag([2.0, 2.0, 2.0, 1.0]))
    path = tmp_path / "case1.nii.gz"
    nib.save(img, path)
    vols, spacing = volumes_from_nifti(path)
    assert spacing == (2.0, 2.0, 2.0) and vols["ET"]["cm3"] == 0.064
    table = volumes_table([path])
    assert table.loc[0, "case"] == "case1" and table.loc[0, "ET_cm3"] == 0.064


def test_resection_cavity_is_not_tumor():
    seg = make_seg()
    seg[7:9, 7:9, 7:9] = 4                     # post-operative cavity (MU-Glioma-Post)
    v = region_volumes(seg, (1, 1, 1))
    assert v["WT"]["voxels"] == 8 + 3 + 5 and v["TC"]["voxels"] == 11 and v["ET"]["voxels"] == 8


def test_unknown_labels_are_refused():
    import pytest
    seg = make_seg()
    seg[9, 9, 9] = 5
    with pytest.raises(ValueError, match="unexpected label"):
        region_volumes(seg, (1, 1, 1))
