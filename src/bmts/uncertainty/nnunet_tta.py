"""Run seg-model v1 (nnU-Net v2) once per mirror flip and turn the passes into confidence maps (M3-T04).

nnU-Net's own prediction already averages the 8 mirror flips (mirroring test-time augmentation). Here the 8 passes are
run one by one instead, so they can be compared. The final mask is the argmax of the averaged logits, exactly what
`nnUNetv2_predict` with mirroring returns, so the segmentation of seg-model v1 does not change.

Arrays come back in nnU-Net's (SimpleITK) axis order (z, y, x); read the expert mask with read_sitk() so it matches.
"""
from __future__ import annotations

import itertools

import numpy as np

from bmts.uncertainty.maps import TTAAccumulator


def flip_combinations(axes):
    """No flip, then every combination of the mirror axes: (), (0,), (1,), (2,), (0, 1), ... (8 for 3 axes)."""
    axes = tuple(axes or ())
    return [()] + [c for k in range(1, len(axes) + 1) for c in itertools.combinations(axes, k)]


def load_predictor(model_dir, fold=0, checkpoint="checkpoint_best.pth", device="cuda"):
    """nnU-Net predictor with its own mirroring switched OFF (the flips are done in predict_case_tta)."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                                perform_everything_on_device=device == "cuda", device=torch.device(device),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint)
    if predictor.label_manager.has_regions:
        raise ValueError("seg-model v1 was trained on labels; a region-based model needs other probability handling")
    return predictor


def preprocess(predictor, image_files):
    """nnU-Net preprocessing of one case. image_files in channel order T1, T1ce, T2, FLAIR."""
    pre = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _, properties = pre.run_case([str(f) for f in image_files], None, predictor.plans_manager,
                                       predictor.configuration_manager, predictor.dataset_json)
    return data, properties


def nnunet_probabilities(predictor, logits, properties):
    """Logits in network space -> class probabilities in the original image grid (C, z, y, x)."""
    from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape

    _, probs = convert_predicted_logits_to_segmentation_with_correct_shape(
        logits, predictor.plans_manager, predictor.configuration_manager, predictor.label_manager, properties,
        return_probabilities=True)
    return np.asarray(probs, dtype=np.float32)


def predict_case_tta(predictor, data, properties, to_probabilities=None):
    """One pass per mirror flip. Returns (segmentation, maps, number of passes).

    to_probabilities(logits) -> (C, ...) probabilities in the output grid; defaults to nnU-Net's own conversion
    (tests pass a simple softmax instead).
    """
    import torch

    if to_probabilities is None:
        def to_probabilities(logits):
            return nnunet_probabilities(predictor, logits, properties)

    x = torch.from_numpy(np.asarray(data, dtype=np.float32))
    acc, sum_logits = TTAAccumulator(), None
    combos = flip_combinations(predictor.allowed_mirroring_axes)
    for combo in combos:
        dims = [a + 1 for a in combo]                        # axis 0 of data is the channel
        logits = predictor.predict_logits_from_preprocessed_data(torch.flip(x, dims) if dims else x)
        logits = (torch.flip(logits, dims) if dims else logits).float()
        sum_logits = logits.clone() if sum_logits is None else sum_logits + logits
        acc.add(to_probabilities(logits))
    segmentation = to_probabilities(sum_logits / len(combos)).argmax(0).astype(np.uint8)
    return segmentation, acc.finalize(segmentation), len(combos)


def predict_with_confidence(predictor, image_files, out_dir=None, case_id="case"):
    """Scan in -> mask + confidence maps out (the entry point for the M3-T05 pipeline).

    Returns (segmentation, maps, properties). With out_dir, also writes <case_id>_seg.nii.gz,
    <case_id>_uncertainty.nii.gz and <case_id>_disagreement.nii.gz next to each other, in the scan's geometry.
    """
    data, properties = preprocess(predictor, image_files)
    segmentation, maps, _ = predict_case_tta(predictor, data, properties)
    if out_dir is not None:
        from pathlib import Path
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_like(segmentation, out / f"{case_id}_seg.nii.gz", properties)
        write_like(maps["uncertainty"], out / f"{case_id}_uncertainty.nii.gz", properties)
        write_like(maps["disagreement"], out / f"{case_id}_disagreement.nii.gz", properties)
    return segmentation, maps, properties


def read_sitk(path, dtype=None):
    """Image as a numpy array in SimpleITK order (z, y, x), the order nnU-Net works in."""
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return arr.astype(dtype) if dtype is not None else arr


def write_like(array, path, properties):
    """Save an array (z, y, x) as NIfTI with the geometry of the input scan (nnU-Net's sitk_stuff)."""
    import SimpleITK as sitk

    img = sitk.GetImageFromArray(array)
    geometry = properties["sitk_stuff"]
    img.SetSpacing(geometry["spacing"])
    img.SetOrigin(geometry["origin"])
    img.SetDirection(geometry["direction"])
    sitk.WriteImage(img, str(path), True)
