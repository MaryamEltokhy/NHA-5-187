"""Tumor volume change between scans of the same patient (M3-T02).

Change per region between an earlier scan A and a later scan B:
    change_cm3 = V_B - V_A
    change_pct = (V_B - V_A) / V_A x 100, only when V_A >= min_baseline_cm3 (default 1 cm3)
Below the minimum the percentage is left empty (NaN) with a reason: a few voxels of difference on a tiny region would
look like a change of thousands of percent. The change in cm3 is always given.

"Lining up" the scans: the cleaned scans (M1-T04) of a patient are all in the same 240 x 240 x 155, 1 mm grid, and a
volume does not depend on where the tumor is, so no registration is needed for volume change. same_grid() checks
that two scans really share the grid before they are compared; a voxel-by-voxel change map would need registration.

These are measurements of labelled volumes, for decision support only; they are not a response assessment.
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from bmts.common.constants import REGIONS

MIN_BASELINE_CM3 = 1.0


def percent_change(before_cm3, after_cm3, min_baseline_cm3=MIN_BASELINE_CM3):
    """(after - before) / before x 100, or NaN when the earlier volume is below min_baseline_cm3."""
    if before_cm3 is None or after_cm3 is None or before_cm3 < min_baseline_cm3:
        return math.nan
    return (after_cm3 - before_cm3) / before_cm3 * 100.0


def change_rows(volumes, pairs=None, names=None, min_baseline_cm3=MIN_BASELINE_CM3, days=None):
    """One row per (pair of scans, region).

    volumes: {timepoint: {region: cm3}} for one patient, timepoints sortable in time order (e.g. 0, 1, 2).
    pairs: list of (earlier, later); default every pair in time order.
    names: optional {timepoint: readable name} (e.g. {0: "pre-op"}).
    days: optional {(earlier, later): days between the scans}.
    """
    order = sorted(volumes)
    pairs = pairs if pairs is not None else list(itertools.combinations(order, 2))
    names, days = names or {}, days or {}
    rows = []
    for a, b in pairs:
        if order.index(a) >= order.index(b):
            raise ValueError(f"pair ({a}, {b}) is not in time order")
        for region in REGIONS:
            va, vb = volumes[a][region], volumes[b][region]
            pct = percent_change(va, vb, min_baseline_cm3)
            rows.append({
                "tp_a": a, "tp_b": b,
                "pair": f"{names.get(a, a)} -> {names.get(b, b)}",
                "region": region,
                "vol_a_cm3": va, "vol_b_cm3": vb,
                "change_cm3": vb - va,
                "change_pct": pct,
                "pct_note": "" if not math.isnan(pct) else f"earlier volume below {min_baseline_cm3:g} cm3",
                "days": days.get((a, b), math.nan),
            })
    return rows


def same_grid(img_a, img_b, atol=1e-3):
    """True when two nibabel images have the same shape and affine (same voxel grid)."""
    return img_a.shape[:3] == img_b.shape[:3] and np.allclose(img_a.affine, img_b.affine, atol=atol)
