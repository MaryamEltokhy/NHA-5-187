"""Radiomics features per tumor region (M3-T03), the code behind results/M3-T03 as a reusable function.

For each region (TC, WT, ET from REGIONS) of one label map, PyRadiomics computes shape features (once, from the mask)
and first-order + texture features on T1ce, FLAIR and T2, with the settings in configs/radiomics/pyradiomics_params.yaml.
Column names match results/M3-T03/radiomics_features_val.csv: shape_<Feature>, <mod>_<class>_<Feature>.

The images must be the cleaned scans (M1-T04: 1 mm, z-scored inside the brain); that is why the settings use a bin
width of 0.25 and no normalization or resampling. Features describe the given mask: on predicted masks they carry the
model's errors. They are research measurements, not clinical findings.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml

from bmts.common.constants import REGIONS

RADIOMICS_MODALITIES = ("t1ce", "flair", "t2")    # T1 adds little contrast for the tumor parts (EDA report §2)
DEFAULT_PARAMS = Path(__file__).resolve().parents[3] / "configs" / "radiomics" / "pyradiomics_params.yaml"
SMALL_REGION_VOXELS = 100      # texture features of regions this small are unreliable
FEW_BINS = 16                  # fewer gray-level bins than this make texture features coarse


def load_params(path=DEFAULT_PARAMS):
    """Read the settings file and fail loudly when it is missing (PyRadiomics would silently use its defaults)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"radiomics settings not found: {path}")
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "setting" not in params or "binWidth" not in params["setting"]:
        raise ValueError(f"{path} has no setting.binWidth; refusing to run with PyRadiomics defaults")
    return params


def make_extractor(params_path=DEFAULT_PARAMS):
    """PyRadiomics extractor configured from the settings file."""
    from radiomics import featureextractor

    load_params(params_path)                        # validates before PyRadiomics reads it
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    return featureextractor.RadiomicsFeatureExtractor(str(params_path))


def to_float(value):
    """PyRadiomics values as float; complex shape values keep their real part (NaN if the imaginary part is not ~0)."""
    v = np.asarray(value).item()
    if isinstance(v, complex):
        return v.real if abs(v.imag) < 1e-9 else float("nan")
    return float(v)


def quality_note(row, bin_width):
    """Short warning when texture features of this region are unreliable ('' when fine)."""
    notes = []
    if 0 < row.get("voxels", 0) < SMALL_REGION_VOXELS:
        notes.append(f"small region (<{SMALL_REGION_VOXELS} voxels)")
    coarse = [m for m in RADIOMICS_MODALITIES
              if f"{m}_firstorder_Range" in row and row[f"{m}_firstorder_Range"] / bin_width < FEW_BINS]
    if coarse:
        notes.append(f"few gray-level bins (<{FEW_BINS}) in {', '.join(coarse)}")
    return "; ".join(notes)


def extract_case(image_files, seg_file, case_id="case", extractor=None, params_path=DEFAULT_PARAMS,
                 regions=REGIONS, modalities=RADIOMICS_MODALITIES):
    """Feature rows (one per region) for one scan.

    image_files: {modality: path} with at least the modalities used; seg_file: label map in the same grid.
    """
    import SimpleITK as sitk

    if extractor is None:
        extractor = make_extractor(params_path)
    bin_width = float(load_params(params_path)["setting"]["binWidth"])
    seg = sitk.ReadImage(str(seg_file))
    seg_arr = sitk.GetArrayFromImage(seg)
    images = {m: sitk.ReadImage(str(image_files[m])) for m in modalities}
    rows = []
    for region, labels in regions.items():
        region_arr = np.isin(seg_arr, labels)
        row = {"patient_id": case_id, "region": region, "voxels": int(region_arr.sum()), "skip_reason": ""}
        if row["voxels"] == 0:
            row["skip_reason"] = "empty_region"
            rows.append(row)
            continue
        mask = sitk.GetImageFromArray(region_arr.astype(np.uint8))
        mask.CopyInformation(seg)
        for mod in modalities:
            try:
                result = extractor.execute(images[mod], mask)
            except Exception as e:  # noqa: BLE001 - keep the other regions and image types
                row["skip_reason"] += f"{mod}:{type(e).__name__}:{str(e)[:80]} "
                continue
            for key, value in result.items():
                if key.startswith("diagnostics") or (key.startswith("original_shape") and mod != modalities[0]):
                    continue                                 # shape features once, from the mask
                name = key.replace("original_", "")
                row[name if name.startswith("shape") else f"{mod}_{name}"] = to_float(value)
        row["skip_reason"] = row["skip_reason"].strip()
        row["quality_note"] = quality_note(row, bin_width)
        rows.append(row)
    return rows
