"""Per-region scores for one case (M2-T03 training and final validation).

Same conventions as the team's metrics (M2-T02, pipeline/metrics.py, checked against MONAI):
Dice and IoU are 1 when prediction and ground truth are both empty; HD95 is 0 when both are empty and NaN when
only one is; sensitivity is NaN without ground truth, precision is NaN without prediction. NaN values are left out
of averages and counted separately.
"""
from __future__ import annotations

import math

import torch
from monai.metrics import compute_hausdorff_distance

from bmts.common.constants import REGIONS

METRICS = ("dice", "iou", "sensitivity", "precision", "hd95")


def region_scores(pred, label, spacing=(1.0, 1.0, 1.0), with_hd95=True):
    """pred, label: (C, H, W, D) binary tensors, channels in REGIONS order. Returns {region: {metric: value}}."""
    scores = {}
    for i, region in enumerate(REGIONS):
        p, g = pred[i].bool(), label[i].bool()
        tp, fp, fn = (p & g).sum().item(), (p & ~g).sum().item(), (~p & g).sum().item()
        empty = tp + fp + fn == 0
        s = {
            "dice": 1.0 if empty else 2 * tp / (2 * tp + fp + fn),
            "iou": 1.0 if empty else tp / (tp + fp + fn),
            "sensitivity": tp / (tp + fn) if tp + fn else math.nan,
            "precision": tp / (tp + fp) if tp + fp else math.nan,
        }
        if with_hd95:
            if empty:
                s["hd95"] = 0.0
            elif not p.any() or not g.any():
                s["hd95"] = math.nan
            else:
                s["hd95"] = float(compute_hausdorff_distance(p[None, None].float(), g[None, None].float(),
                                                             include_background=True, percentile=95,
                                                             spacing=list(spacing))[0, 0])
        scores[region] = s
    return scores


def summarize(per_case):
    """Average a list of region_scores outputs: {region: {metric: mean, metric_n: cases used, metric_nan: skipped}}."""
    summary = {}
    for region in REGIONS:
        summary[region] = {}
        for metric in METRICS:
            values = [case[region][metric] for case in per_case if metric in case[region]]
            if not values:
                continue
            valid = [v for v in values if not math.isnan(v)]
            summary[region][metric] = sum(valid) / len(valid) if valid else math.nan
            summary[region][f"{metric}_n"] = len(valid)
            summary[region][f"{metric}_nan"] = len(values) - len(valid)
    dice = [summary[r]["dice"] for r in REGIONS if "dice" in summary[r]]
    summary["mean_dice"] = sum(dice) / len(dice) if dice else math.nan
    return summary


def binarize(logits, threshold=0.5):
    """Sigmoid outputs -> binary region masks."""
    return (torch.sigmoid(logits.float()) > threshold).float()
