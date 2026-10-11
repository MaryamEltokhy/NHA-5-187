"""Per-scan numbers for the exploratory data analysis (EDA report, Milestone 1).

Works on cleaned scans (M1-T04): canonical labels 0 background, 1 NCR, 2 ED, 3 ET, 4 RC, and every MRI type
z-scored inside the brain with the background written as 0. Only numpy and scipy, so it is easy to test.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from bmts.common.constants import ED, ET, MODALITIES, NCR, RC

LABELS = {"ncr": NCR, "ed": ED, "et": ET, "rc": RC}
TISSUES = ("healthy", "ed", "ncr", "et")       # brain without tumor, then the three tumor labels
HIST_EDGES = np.linspace(-4.0, 8.0, 121)        # z-score bins for the intensity histograms
_CONNECTIVITY = ndimage.generate_binary_structure(3, 3)  # 26-neighbour connectivity


def tumor_stats(seg, spacing=(1.0, 1.0, 1.0), min_component_ml=0.1):
    """Volumes (mL), composition, number of separate tumor parts and position of one mask.

    Components smaller than min_component_ml are counted separately (wt_small_components) so a few stray voxels
    do not make a tumor look multifocal.
    """
    seg = np.asarray(seg)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    counts = {name: int(np.count_nonzero(seg == label)) for name, label in LABELS.items()}
    wt = np.isin(seg, (NCR, ED, ET))
    n_wt = counts["ncr"] + counts["ed"] + counts["et"]
    out = {f"vol_{name}_ml": n * voxel_ml for name, n in counts.items()}
    out["vol_tc_ml"] = (counts["ncr"] + counts["et"]) * voxel_ml
    out["vol_wt_ml"] = n_wt * voxel_ml
    out["et_present"] = counts["et"] > 0
    out["wt_share_of_volume"] = n_wt / seg.size
    for name in ("ncr", "ed", "et"):
        out[f"share_{name}_of_wt"] = counts[name] / n_wt if n_wt else np.nan

    labelled, n = ndimage.label(wt, structure=_CONNECTIVITY)
    sizes_ml = np.bincount(labelled.ravel())[1:] * voxel_ml
    big = sizes_ml >= min_component_ml
    out["wt_components"] = int(big.sum())
    out["wt_small_components"] = int(n - big.sum())
    out["largest_component_share"] = float(sizes_ml.max() / sizes_ml.sum()) if n else np.nan

    if n_wt:
        centre = ndimage.center_of_mass(wt)
        idx = np.nonzero(wt)
        for axis, name in enumerate("xyz"):
            out[f"centroid_{name}"] = centre[axis] / (seg.shape[axis] - 1)       # 0..1 across the volume
            out[f"extent_{name}_mm"] = (idx[axis].max() - idx[axis].min() + 1) * spacing[axis]
        out["touches_border"] = any(idx[a].min() == 0 or idx[a].max() == seg.shape[a] - 1 for a in range(3))
    else:
        for name in "xyz":
            out[f"centroid_{name}"] = out[f"extent_{name}_mm"] = np.nan
        out["touches_border"] = False
    return out


def tissue_masks(images, seg):
    """Boolean masks of healthy brain and each tumor label. Brain = voxels that are non-zero in any MRI type."""
    seg = np.asarray(seg)
    brain = np.zeros(seg.shape, bool)
    for name in MODALITIES:
        brain |= np.asarray(images[name]) != 0
    return {"brain": brain, "healthy": brain & (seg == 0), "ed": seg == ED, "ncr": seg == NCR, "et": seg == ET}


def intensity_stats(images, seg, spacing=(1.0, 1.0, 1.0)):
    """Brain volume, tumor share of the brain, mean z-score of each MRI type in each tissue, and histograms.

    Returns (row, hist): row is a flat dict for a table; hist[modality][tissue] holds counts over HIST_EDGES.
    """
    masks = tissue_masks(images, seg)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    n_brain = int(masks["brain"].sum())
    n_tumor = int(sum(masks[t].sum() for t in ("ed", "ncr", "et")))
    row = {"brain_ml": n_brain * voxel_ml, "wt_share_of_brain": n_tumor / n_brain if n_brain else np.nan}
    hist = {}
    for name in MODALITIES:
        img = np.asarray(images[name], dtype=np.float32)
        hist[name] = {}
        for tissue in TISSUES:
            values = img[masks[tissue]]
            row[f"mean_{name}_{tissue}"] = float(values.mean()) if values.size else np.nan
            hist[name][tissue] = np.histogram(np.clip(values, HIST_EDGES[0], HIST_EDGES[-1]), HIST_EDGES)[0]
    return row, hist


def nonzero_percentiles(image, q=(1, 50, 99)):
    """Percentiles of the non-background voxels (background = the most common value, as in the cleaning step)."""
    image = np.asarray(image, dtype=np.float32)
    values, counts = np.unique(image[::4, ::4, ::4], return_counts=True)   # subsample: background estimate only
    background = values[counts.argmax()]
    inside = image[image != background]
    return dict(zip((f"p{v}" for v in q), np.percentile(inside, q))) if inside.size else {f"p{v}": np.nan for v in q}


def best_axial_slice(mask):
    """Index of the axial slice (last axis) with the most voxels of mask; the middle slice when mask is empty."""
    area = np.asarray(mask).reshape(-1, mask.shape[-1]).sum(axis=0)
    return int(area.argmax()) if area.any() else mask.shape[-1] // 2
