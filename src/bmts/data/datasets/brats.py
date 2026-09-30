"""Find the cleaned cases of a split and build the training / validation data loaders (M1-T05).

Cleaned layout (M1-T04, Yasmin's pipeline/standardize.py):
    <processed_root>/<dataset>/<subject>/<timepoint>/{t1,t1ce,t2,flair,seg}.nii.gz
Split file (M1-T04, qc/split_v1.csv): one row per patient with dataset_source, canonical_subject_id, split, lockbox.

Locked patients (split "test" or lockbox True) are refused unless allow_lockbox=True, which only the frozen
final evaluation (M5-T02) may set.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from monai.data import CacheDataset, DataLoader, Dataset

from bmts.common.constants import MODALITIES, SEG
from bmts.data.augmentation.transforms import train_transforms, val_transforms

REQUIRED_COLUMNS = {"dataset_source", "canonical_subject_id", "split"}


class LockboxError(RuntimeError):
    """Raised when code asks for locked test patients outside the final evaluation."""


def load_config(path, **overrides):
    """Read a loader YAML config; keyword overrides replace top-level keys (None values are ignored)."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def list_cases(split_csv, processed_root, splits=("train",), datasets=("brats2021",), limit=None,
               allow_lockbox=False, on_missing="error"):
    """Return one dict per (patient, timepoint): {"id", "t1", "t1ce", "t2", "flair", "seg"} file paths.

    limit keeps the first N patients (sorted by ID) before the files are checked, for quick smoke tests.
    on_missing: "error" stops on the first incomplete case list; "skip" drops those cases with a warning.
    """
    table = pd.read_csv(split_csv, dtype=str, keep_default_na=False)
    missing_columns = REQUIRED_COLUMNS - set(table.columns)
    if missing_columns:
        raise ValueError(f"{split_csv} lacks columns {sorted(missing_columns)}")
    rows = table[table.split.isin(splits) & table.dataset_source.isin(datasets)]
    locked = rows.split.eq("test")
    if "lockbox" in rows:
        locked |= rows.lockbox.str.lower().isin(["true", "1", "yes"])
    if locked.any() and not allow_lockbox:
        raise LockboxError(f"{int(locked.sum())} requested patients are in the locked test set; "
                           "they may only be loaded by the final evaluation (M5-T02).")
    rows = rows.sort_values("canonical_subject_id")
    if limit is not None:
        rows = rows.head(limit)

    cases, problems = [], []
    for row in rows.itertuples():
        subject = row.canonical_subject_id.split("__", 1)[-1]
        subject_dir = Path(processed_root) / row.dataset_source / subject
        timepoints = sorted(p for p in subject_dir.iterdir() if p.is_dir()) if subject_dir.is_dir() else []
        if not timepoints:
            problems.append(f"{row.canonical_subject_id}: no cleaned scans in {subject_dir}")
            continue
        for tp in timepoints:
            files = {k: tp / f"{k}.nii.gz" for k in (*MODALITIES, SEG)}
            absent = [k for k, p in files.items() if not p.is_file()]
            if absent:
                problems.append(f"{row.canonical_subject_id}/{tp.name}: missing {', '.join(absent)}")
                continue
            cases.append({"id": f"{row.canonical_subject_id}/{tp.name}", **{k: str(p) for k, p in files.items()}})

    if problems:
        message = f"{len(problems)} of {len(rows)} patients are incomplete, e.g.\n  " + "\n  ".join(problems[:5])
        if on_missing == "error":
            raise FileNotFoundError(message + "\nRun the cleaning step for them, or set on_missing: skip.")
        warnings.warn(message, stacklevel=2)
    return cases


def build_loaders(cfg, train_limit=None, val_limit=None):
    """Training loader (random augmented patches) and validation loader (whole volumes, batch size 1)."""
    common = dict(split_csv=cfg["split_csv"], processed_root=cfg["processed_root"], datasets=cfg["datasets"],
                  on_missing=cfg.get("on_missing", "error"))
    train_cases = list_cases(splits=(cfg["train_split"],), limit=train_limit, **common)
    val_cases = list_cases(splits=(cfg["val_split"],), limit=val_limit, **common)
    if not train_cases:
        raise RuntimeError("No training cases found: check split_csv, processed_root and datasets in the config.")

    cache_rate = cfg.get("cache_rate", 0.0)
    workers = cfg.get("num_workers", 0)
    if cache_rate > 0:
        train_ds = CacheDataset(train_cases, train_transforms(cfg), cache_rate=cache_rate, num_workers=workers)
    else:
        train_ds = Dataset(train_cases, train_transforms(cfg))
    val_ds = Dataset(val_cases, val_transforms(cfg))
    extra = dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=False, **extra)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **extra)
    return train_loader, val_loader
