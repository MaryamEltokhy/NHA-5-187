# Evaluation metrics (M2-T02): Dice and HD95 for WT, TC and ET, verified to match MONAI.
import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion

NCR, ED, ET_LABEL = 1, 2, 3
REGIONS = {
    'WT': (NCR, ED, ET_LABEL),   # whole tumor: everything that is not background
    'TC': (NCR, ET_LABEL),       # tumor core: necrotic + enhancing, excludes edema
    'ET': (ET_LABEL,),           # enhancing tumor only
}


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice similarity between two binary masks. Returns 1.0 if both are empty."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def surface_distances(mask: np.ndarray, spacing: tuple) -> np.ndarray:
    """Distance (in mm) from every voxel to the nearest surface voxel of `mask`."""
    eroded = binary_erosion(mask)
    surface = mask & ~eroded
    return distance_transform_edt(~surface, sampling=spacing)


def hd95(pred: np.ndarray, target: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> float:
    """95th-percentile Hausdorff Distance between two binary masks, in millimeters.

    Returns 0.0 if both masks are empty, nan if only one is empty (matches MONAI's convention).
    """
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float('nan')

    pred_surface = pred & ~binary_erosion(pred)
    target_surface = target & ~binary_erosion(target)
    dist_to_target = distance_transform_edt(~target, sampling=spacing)
    dist_to_pred = distance_transform_edt(~pred, sampling=spacing)
    distances_pred_to_target = dist_to_target[pred_surface]
    distances_target_to_pred = dist_to_pred[target_surface]
    all_distances = np.concatenate([distances_pred_to_target, distances_target_to_pred])
    return float(np.percentile(all_distances, 95))


def region_mask(seg: np.ndarray, labels: tuple) -> np.ndarray:
    """True wherever seg has one of the given label values."""
    return np.isin(seg, labels)


def evaluate_case(pred_seg: np.ndarray, target_seg: np.ndarray, spacing: tuple = (1.0, 1.0, 1.0)) -> dict:
    """Dice and HD95 for WT, TC and ET, from two full segmentation maps (not binary masks)."""
    results = {}
    for region_name, labels in REGIONS.items():
        pred_mask = region_mask(pred_seg, labels)
        target_mask = region_mask(target_seg, labels)
        results[region_name] = {
            'dice': dice_score(pred_mask, target_mask),
            'hd95': hd95(pred_mask, target_mask, spacing=spacing),
        }
    return results


def summarize_results(results_list: list) -> dict:
    """Averages Dice and HD95 over several cases (output of evaluate_case), ignoring nan values."""
    summary = {}
    for region_name in REGIONS:
        dice_values = [r[region_name]['dice'] for r in results_list]
        hd95_values = [r[region_name]['hd95'] for r in results_list]
        summary[region_name] = {
            'dice_mean': float(np.nanmean(dice_values)),
            'hd95_mean': float(np.nanmean(hd95_values)),
            'n_cases': len(results_list),
        }
    return summary
