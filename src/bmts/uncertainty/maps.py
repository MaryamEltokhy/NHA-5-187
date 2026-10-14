"""Confidence (uncertainty) maps from repeated predictions of one scan (M3-T04).

The selected model (seg-model v1, nnU-Net) has no dropout, so "run the model several times with small changes" is done
with test-time augmentation: the same scan is predicted once per mirror flip (8 passes) and the passes are compared.
Every pass gives class probabilities (background, NCR, ED, ET). From them, per voxel:

- uncertainty (normalized predictive entropy, 0..1): how unsure the averaged prediction is. 0 = all passes agree
  with full confidence, 1 = all four classes equally likely. This is the confidence map shown to the user.
- disagreement (mutual information between passes, normalized 0..1): the part of that uncertainty that comes from
  the passes contradicting each other.
- per region (WT, TC, ET): averaged probability, and the share of passes whose region mask differs from the final one.

Only numpy here; running nnU-Net is in nnunet_tta.py.
"""
from __future__ import annotations

import numpy as np

from bmts.common.constants import REGIONS

EPS = 1e-8
N_CLASSES = 4                      # background, NCR, ED, ET (canonical labels 0..3)


def region_probabilities(probs):
    """(4, ...) class probabilities -> {region: probability} using REGIONS (WT = 1+2+3, TC = 1+3, ET = 3)."""
    return {name: np.sum([probs[label] for label in labels], axis=0) for name, labels in REGIONS.items()}


def entropy(probs, axis=0):
    """Entropy over the class axis, divided by log(number of classes) so it lies in 0..1."""
    p = np.clip(probs, EPS, 1.0)
    return -(p * np.log(p)).sum(axis=axis) / np.log(probs.shape[axis])


def binary_entropy(p):
    """Entropy of a yes/no probability in bits (0..1): 1 at p = 0.5, 0 at p = 0 or 1."""
    # float64: in float32, 1 - 1e-8 rounds to 1.0, so p = 1 gave 0 * log(0) = NaN
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


class TTAAccumulator:
    """Collects the class probabilities of the passes one at a time, so only running sums are kept in memory."""

    def __init__(self):
        self.n = 0
        self.sum_probs = self.sum_entropy = None
        self.region_votes = {}

    def add(self, probs):
        probs = np.asarray(probs, dtype=np.float32)
        ent = entropy(probs)
        if self.n == 0:
            self.sum_probs, self.sum_entropy = probs.copy(), ent
        else:
            self.sum_probs += probs
            self.sum_entropy += ent
        labels = probs.argmax(0)
        for name, region_labels in REGIONS.items():
            vote = np.isin(labels, region_labels).astype(np.uint8)
            self.region_votes[name] = vote if self.n == 0 else self.region_votes[name] + vote
        self.n += 1

    def finalize(self, segmentation):
        """Maps for the final segmentation (label map of the averaged prediction). Returns a dict of arrays."""
        if self.n == 0:
            raise ValueError("no pass added")
        mean_probs = self.sum_probs / self.n
        uncertainty = entropy(mean_probs)
        disagreement = np.clip(uncertainty - self.sum_entropy / self.n, 0, 1)
        out = {"mean_probs": mean_probs, "uncertainty": uncertainty.astype(np.float32),
               "disagreement": disagreement.astype(np.float32)}
        for name, region_labels in REGIONS.items():
            final = np.isin(segmentation, region_labels)
            votes = self.region_votes[name] / self.n          # share of passes that put the voxel in the region
            out[f"flip_rate_{name}"] = np.where(final, 1 - votes, votes).astype(np.float32)
        return out


def case_summary(segmentation, maps, voxel_ml=0.001, threshold=0.5):
    """Numbers for one case: region volumes, how much of the tumor is uncertain, and mean uncertainty near it.

    "Near the tumor" = voxels where the averaged probability of any tumor label is at least 5 %, so the uncertain
    rim around the predicted tumor is included. Uncertain volume = voxels with region binary entropy > threshold.
    """
    rp = region_probabilities(maps["mean_probs"])
    near = rp["WT"] >= 0.05
    row = {"uncertainty_near_tumor": float(maps["uncertainty"][near].mean()) if near.any() else 0.0,
           "disagreement_near_tumor": float(maps["disagreement"][near].mean()) if near.any() else 0.0,
           "max_uncertainty": float(maps["uncertainty"].max())}
    for name, labels in REGIONS.items():
        pred = np.isin(segmentation, labels)
        be = binary_entropy(rp[name])
        zone = rp[name] >= 0.05
        row[f"vol_{name}_ml"] = float(pred.sum() * voxel_ml)
        row[f"uncertain_{name}_ml"] = float((be > threshold).sum() * voxel_ml)
        row[f"uncertainty_{name}"] = float(be[zone].mean()) if zone.any() else 0.0
        row[f"flip_rate_{name}"] = float(maps[f"flip_rate_{name}"][zone].mean()) if zone.any() else 0.0
    return row


# ---------------------------------------------------------------- checks against the expert mask (validation only)

class CalibrationCounter:
    """Reliability-diagram counts per region: for each probability bin, how many voxels and how many truly in the region.

    A well calibrated model has, in the bin around 0.7, about 70 % of voxels really in the region.
    """

    def __init__(self, bins=20):
        self.edges = np.linspace(0, 1, bins + 1)
        self.count = {r: np.zeros(bins, np.int64) for r in REGIONS}
        self.positive = {r: np.zeros(bins, np.int64) for r in REGIONS}
        self.sum_prob = {r: np.zeros(bins) for r in REGIONS}

    def add(self, mean_probs, ground_truth, mask):
        rp = region_probabilities(mean_probs)
        for name, labels in REGIONS.items():
            p = rp[name][mask]
            truth = np.isin(ground_truth[mask], labels)
            idx = np.clip(np.digitize(p, self.edges) - 1, 0, len(self.edges) - 2)
            self.count[name] += np.bincount(idx, minlength=len(self.count[name]))
            self.positive[name] += np.bincount(idx, weights=truth, minlength=len(self.count[name])).astype(np.int64)
            self.sum_prob[name] += np.bincount(idx, weights=p, minlength=len(self.count[name]))

    def ece(self, name):
        """Expected calibration error: average gap between predicted probability and observed frequency."""
        n = self.count[name]
        used = n > 0
        conf = self.sum_prob[name][used] / n[used]
        freq = self.positive[name][used] / n[used]
        return float((n[used] * np.abs(conf - freq)).sum() / n.sum())

    def curve(self, name):
        n = self.count[name]
        used = n > 0
        return self.sum_prob[name][used] / n[used], self.positive[name][used] / n[used], n[used]


class ErrorDetectionCounter:
    """Does the map point at the mistakes? Histograms of region uncertainty for wrong and for right voxels."""

    def __init__(self, bins=100):
        self.edges = np.linspace(0, 1, bins + 1)
        self.wrong = {r: np.zeros(bins, np.int64) for r in REGIONS}
        self.right = {r: np.zeros(bins, np.int64) for r in REGIONS}

    def add(self, mean_probs, segmentation, ground_truth, mask):
        rp = region_probabilities(mean_probs)
        for name, labels in REGIONS.items():
            u = binary_entropy(rp[name][mask])
            err = np.isin(segmentation[mask], labels) != np.isin(ground_truth[mask], labels)
            idx = np.clip(np.digitize(u, self.edges) - 1, 0, len(self.edges) - 2)
            self.wrong[name] += np.bincount(idx[err], minlength=len(self.edges) - 1)
            self.right[name] += np.bincount(idx[~err], minlength=len(self.edges) - 1)

    def auroc(self, name):
        """Chance that a random wrong voxel is more uncertain than a random right one (0.5 = no better than chance)."""
        return auroc_from_histograms(self.wrong[name], self.right[name])

    def share_of_errors_flagged(self, name, threshold=0.5):
        """Share of wrong voxels, and of all voxels, with uncertainty above threshold."""
        cut = np.searchsorted(self.edges, threshold)          # first bin starting at threshold
        w, r = self.wrong[name], self.right[name]
        flagged_errors = w[cut:].sum() / max(w.sum(), 1)
        flagged_all = (w[cut:].sum() + r[cut:].sum()) / max(w.sum() + r.sum(), 1)
        return float(flagged_errors), float(flagged_all)


def auroc_from_histograms(positive, negative):
    """AUROC from two histograms over the same ascending bins (ties count half)."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    if positive.sum() == 0 or negative.sum() == 0:
        return float("nan")
    neg_below = np.cumsum(negative) - negative                  # negatives in strictly lower bins
    wins = (positive * neg_below).sum() + 0.5 * (positive * negative).sum()
    return float(wins / (positive.sum() * negative.sum()))
