"""Tumor volume per region from a segmentation map (M3-T01).

Volume = number of voxels x volume of one voxel. The voxel size is read from the NIfTI header of the
file itself, never assumed. Volumes are kept in mm3 and reported in cm3 (1 cm3 = 1000 mm3 = 1 mL).
Regions are the team's REGIONS (TC = labels 1+3, WT = labels 1+2+3, ET = label 3), so they overlap.
"""
import numpy as np
import nibabel as nib
import pandas as pd

from bmts.common.constants import REGIONS

MM3_PER_CM3 = 1000.0


def voxel_volume_mm3(spacing) -> float:
    """Volume of one voxel in mm3 from its size (dx, dy, dz) in mm."""
    return float(np.prod([float(s) for s in spacing]))


def region_volumes(seg: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """Voxel count, mm3 and cm3 for TC, WT and ET of one label map."""
    one_voxel = voxel_volume_mm3(spacing)
    out = {}
    for name, labels in REGIONS.items():
        voxels = int(np.isin(seg, labels).sum())
        mm3 = voxels * one_voxel
        out[name] = {"voxels": voxels, "mm3": mm3, "cm3": mm3 / MM3_PER_CM3}
    return out


def volumes_from_nifti(path):
    """Read a segmentation file; returns (region volumes, voxel size in mm)."""
    img = nib.load(str(path))
    seg = np.rint(np.asarray(img.dataobj)).astype(np.int16)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])
    return region_volumes(seg, spacing), spacing


def volumes_table(paths) -> pd.DataFrame:
    """One row per file: case id, voxel size, and voxels / cm3 for TC, WT, ET."""
    rows = []
    for p in paths:
        vols, spacing = volumes_from_nifti(p)
        name = str(p).split("/")[-1].split("\\")[-1]
        row = {"case": name.replace(".nii.gz", "").replace(".nii", ""),
               "voxel_mm3": voxel_volume_mm3(spacing)}
        for region, v in vols.items():
            row[f"{region}_voxels"] = v["voxels"]
            row[f"{region}_cm3"] = round(v["cm3"], 4)
        rows.append(row)
    return pd.DataFrame(rows)
