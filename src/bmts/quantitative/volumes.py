"""Tumor volume per region from a segmentation map (M3-T01).

Volume = number of voxels x volume of one voxel. The voxel size is read from the NIfTI header of the
file itself, never assumed. Volumes are kept in mm3 and reported in cm3 (1 cm3 = 1000 mm3 = 1 mL).
Regions are the team's REGIONS (TC = labels 1+3, WT = labels 1+2+3, ET = label 3), so they overlap.
Label 4 (resection cavity, post-operative cohorts such as MU-Glioma-Post) belongs to no region and is not counted.
Masks must already use the canonical labels 0-4 (the cleaning step M1-T04 maps BraTS 2021's ET 4 to 3): any other
value raises an error instead of being silently left out.
"""
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

from bmts.common.constants import ED, ET, NCR, RC, REGIONS

MM3_PER_CM3 = 1000.0
VALID_LABELS = (0, 1, 2, 3, 4)
LABELS = {"NCR": NCR, "ED": ED, "ET": ET, "RC": RC}   # one entry per label; RC only in post-op cohorts


def voxel_volume_mm3(spacing) -> float:
    """Volume of one voxel in mm3 from its size (dx, dy, dz) in mm."""
    return float(np.prod([float(s) for s in spacing]))


def check_labels(seg: np.ndarray) -> None:
    """Raise if the label map has values outside the canonical labels 0-4."""
    unexpected = sorted(set(np.unique(seg).tolist()) - set(VALID_LABELS))
    if unexpected:
        raise ValueError(f"unexpected label values {unexpected}; expected canonical labels {VALID_LABELS}")


def region_volumes(seg: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """Voxel count, mm3 and cm3 for TC, WT and ET of one label map (label 4 is not part of any region)."""
    check_labels(seg)
    one_voxel = voxel_volume_mm3(spacing)
    out = {}
    for name, labels in REGIONS.items():
        voxels = int(np.isin(seg, labels).sum())
        mm3 = voxels * one_voxel
        out[name] = {"voxels": voxels, "mm3": mm3, "cm3": mm3 / MM3_PER_CM3}
    return out


def label_volumes(seg: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """Voxel count, mm3 and cm3 of each label on its own: NCR (1), ED (2), ET (3) and RC (4, resection cavity).

    Useful to compare with sources that report single labels, e.g. a necrosis or edema volume. ET is the same as the
    ET region; TC = NCR + ET and WT = NCR + ED + ET.
    """
    check_labels(seg)
    one_voxel = voxel_volume_mm3(spacing)
    out = {}
    for name, label in LABELS.items():
        voxels = int(np.count_nonzero(seg == label))
        mm3 = voxels * one_voxel
        out[name] = {"voxels": voxels, "mm3": mm3, "cm3": mm3 / MM3_PER_CM3}
    return out


def all_volumes(seg: np.ndarray, spacing=(1.0, 1.0, 1.0)) -> dict:
    """Regions (TC, WT, ET) and single labels (NCR, ED, RC) in one dict, keyed by name."""
    out = region_volumes(seg, spacing)
    out.update({k: v for k, v in label_volumes(seg, spacing).items() if k != "ET"})
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
        name = Path(p).name
        row = {"case": name.replace(".nii.gz", "").replace(".nii", ""),
               "voxel_mm3": voxel_volume_mm3(spacing)}
        for region, v in vols.items():
            row[f"{region}_voxels"] = v["voxels"]
            row[f"{region}_cm3"] = round(v["cm3"], 4)
        rows.append(row)
    return pd.DataFrame(rows)
