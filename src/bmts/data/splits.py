"""Which scans may be used during development (lightweight: pandas only, no torch).

The split file (M1-T04, qc/split_v1.csv) has one row per patient for BraTS 2021 and one row per scan for the
longitudinal cohorts (MU-Glioma-Post, RHUH-GBM). Columns used: dataset_source, canonical_subject_id, split, lockbox.
Locked rows (split "test", split "external" or lockbox True) may only be read by the final evaluation (M5-T02).
"""
from __future__ import annotations

import pandas as pd

LOCKED_SPLITS = ("test", "external")


class LockboxError(RuntimeError):
    """Raised when code asks for locked test patients outside the final evaluation."""


def subject_id(canonical_subject_id: str) -> str:
    """'rhuh__RHUH-0002' -> 'RHUH-0002'."""
    return canonical_subject_id.split("__", 1)[-1]


def _locked(rows: pd.DataFrame) -> pd.Series:
    locked = rows.split.isin(LOCKED_SPLITS)
    if "lockbox" in rows:
        locked |= rows.lockbox.astype(str).str.lower().isin(["true", "1", "yes"])
    return locked


def check_subject(split_csv, dataset: str, subject: str, allow_lockbox: bool = False) -> str:
    """Split of one patient ('train' or 'val'); raises LockboxError if any of its rows is locked or it is unknown."""
    table = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
    rows = table[(table.dataset_source == dataset) & (table.canonical_subject_id.map(subject_id) == subject)]
    if rows.empty:
        raise LockboxError(f"{dataset}/{subject} is not in {split_csv}; unknown scans are not used during development")
    if _locked(rows).any() and not allow_lockbox:
        raise LockboxError(f"{dataset}/{subject} is in the locked test or external set; it may only be read by "
                           "the final evaluation (M5-T02)")
    splits = sorted(set(rows.split))
    return splits[0] if len(splits) == 1 else "+".join(splits)
