"""Volume change tests (M3-T02): percentages, small baselines, pair order and the same-grid check."""
import math

import nibabel as nib
import numpy as np
import pytest

from bmts.quantitative.longitudinal import change_rows, percent_change, same_grid


def test_percent_change():
    assert percent_change(10.0, 15.0) == pytest.approx(50.0)
    assert percent_change(10.0, 0.0) == pytest.approx(-100.0)
    assert math.isnan(percent_change(0.0, 4.0))               # no baseline: no percentage
    assert math.isnan(percent_change(0.2, 4.0))               # tiny baseline (default minimum 1 cm3)
    assert percent_change(0.2, 0.4, min_baseline_cm3=0.1) == pytest.approx(100.0)


def test_change_rows_cover_every_pair_and_region():
    vols = {0: {"TC": 2.0, "WT": 30.0, "ET": 1.5},
            1: {"TC": 0.0, "WT": 6.0, "ET": 0.0},
            2: {"TC": 4.0, "WT": 12.0, "ET": 3.0}}
    rows = change_rows(vols, names={0: "pre-op", 1: "early post-op", 2: "recurrence"}, days={(0, 1): 9})
    assert len(rows) == 3 * 3
    first = next(r for r in rows if r["tp_a"] == 0 and r["tp_b"] == 1 and r["region"] == "WT")
    assert first["pair"] == "pre-op -> early post-op" and first["days"] == 9
    assert first["change_cm3"] == pytest.approx(-24.0) and first["change_pct"] == pytest.approx(-80.0)
    after_surgery = next(r for r in rows if r["tp_a"] == 1 and r["region"] == "TC")
    assert math.isnan(after_surgery["change_pct"]) and after_surgery["change_cm3"] == pytest.approx(4.0)
    assert "below 1 cm3" in after_surgery["pct_note"]


def test_pairs_must_be_in_time_order():
    vols = {0: {"TC": 1, "WT": 1, "ET": 1}, 1: {"TC": 1, "WT": 1, "ET": 1}}
    with pytest.raises(ValueError):
        change_rows(vols, pairs=[(1, 0)])


def test_same_grid():
    a = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4))
    b = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.eye(4))
    shifted = nib.Nifti1Image(np.zeros((4, 4, 4), np.uint8), np.diag([1.0, 1.0, 2.0, 1.0]))
    assert same_grid(a, b) and not same_grid(a, shifted)


def test_single_labels_can_be_compared_too():
    vols = {0: {"NCR": 2.0, "ED": 20.0}, 1: {"NCR": 1.0, "ED": 30.0}}
    rows = change_rows(vols, measures=("NCR", "ED"))
    assert [r["region"] for r in rows] == ["NCR", "ED"]
    assert rows[1]["change_pct"] == pytest.approx(50.0)
