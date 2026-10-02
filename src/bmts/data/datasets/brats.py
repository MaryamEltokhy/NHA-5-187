"""Find the cleaned cases of a split and build the training / validation data loaders (M1-T05).

Cleaned layout (M1-T04, Yasmin's pipeline/standardize.py):
    <processed_root>/<dataset>/<subject>/<timepoint>/{t1,t1ce,t2,flair,seg}.nii.gz
Split file (M1-T04, qc/split_v1.csv): one row per patient with dataset_source, canonical_subject_id, split, lockbox.

Locked patients (split "test" or lockbox True) are refused unless allow_lockbox=True, which only the frozen
final evaluation (M5-T02) may set.
"""
from __future__ import annotations

import multiprocessing
import warnings
from pathlib import Path

import pandas as pd
import torch
import yaml
from monai.data import CacheDataset, DataLoader, Dataset, SmartCacheDataset

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


def _cases(cfg, split, limit):
    return list_cases(split_csv=cfg["split_csv"], processed_root=cfg["processed_root"], datasets=cfg["datasets"],
                      splits=(split,), limit=limit, on_missing=cfg.get("on_missing", "error"))


def _loader_options(cfg):
    workers = cfg.get("num_workers", 0)
    return dict(num_workers=workers, pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def build_train_loader(cfg, limit=None):
    """Random augmented patches. Cases can be kept in RAM between epochs:

    - smart_cache_num: N   keeps N patients in RAM and swaps smart_cache_replace_rate of them after every epoch
                           (the trainer calls start / update_cache / shutdown); best for the full training set.
    - cache_rate: r        keeps a fixed share r of the patients in RAM.
    """
    cases = _cases(cfg, cfg["train_split"], limit)
    if not cases:
        raise RuntimeError("No training cases found: check split_csv, processed_root and datasets in the config.")
    transforms, workers = train_transforms(cfg), cfg.get("num_workers", 0)
    smart = cfg.get("smart_cache_num", 0)
    if smart and smart < len(cases):
        dataset = SmartCacheDataset(cases, transforms, cache_num=smart, replace_rate=cfg.get("smart_cache_replace_rate", 0.25),
                                    num_init_workers=max(workers, 1), num_replace_workers=max(workers, 1))
    elif smart or cfg.get("cache_rate", 0.0) > 0:
        dataset = CacheDataset(cases, transforms, cache_rate=1.0 if smart else cfg["cache_rate"], num_workers=max(workers, 1))
    else:
        dataset = Dataset(cases, transforms)
    options = _loader_options(cfg)
    if isinstance(dataset, SmartCacheDataset):
        # it swaps cases between epochs, so worker processes must be recreated every epoch (not persistent),
        # and they must start by "fork" (Linux, Colab): with "spawn" (Windows, macOS) MONAI cannot hand the cache over
        options["persistent_workers"] = False
        if workers > 0 and multiprocessing.get_start_method() != "fork":
            warnings.warn("smart_cache_num needs fork-started workers; loading in the main process instead", stacklevel=2)
            options["num_workers"] = 0
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, drop_last=False, **options)


def build_val_loader(cfg, limit=None):
    """Whole validation volumes, batch size 1 (never the locked test patients)."""
    return DataLoader(Dataset(_cases(cfg, cfg["val_split"], limit), val_transforms(cfg)), batch_size=1,
                      shuffle=False, **_loader_options(cfg))


def build_loaders(cfg, train_limit=None, val_limit=None):
    """Training loader (random augmented patches) and validation loader (whole volumes, batch size 1)."""
    return build_train_loader(cfg, train_limit), build_val_loader(cfg, val_limit)
