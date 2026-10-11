"""Figures of the EDA report. Each function takes the tables made by scripts/eda.py and saves one PNG.

Colors: one fixed color per cohort, per split and per tissue, the same in every figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from bmts.common.constants import MODALITIES  # noqa: E402
from bmts.eda.stats import HIST_EDGES, TISSUES  # noqa: E402

COHORT_COLORS = {"brats2021": "#2a78d6", "mu": "#eb6834", "rhuh": "#1baf7a", "africa": "#eda100", "utsw": "#e87ba4"}
COHORT_NAMES = {"brats2021": "BraTS 2021", "mu": "MU-Glioma-Post", "rhuh": "RHUH-GBM", "africa": "BraTS-Africa",
                "utsw": "UTSW-Glioma"}
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a", "external": "#eda100"}
TISSUE_COLORS = {"healthy": "#8a8985", "ed": "#1baf7a", "ncr": "#eb6834", "et": "#e87ba4"}
TISSUE_NAMES = {"healthy": "healthy brain", "ed": "edema (ED)", "ncr": "necrotic core (NCR)", "et": "enhancing tumor (ET)"}
MOD_NAMES = {"t1": "T1", "t1ce": "T1ce", "t2": "T2", "flair": "FLAIR"}
# mask overlay: 0 transparent, 1 NCR, 2 ED, 3 ET, 4 RC
MASK_CMAP = ListedColormap([(0, 0, 0, 0), TISSUE_COLORS["ncr"], TISSUE_COLORS["ed"], TISSUE_COLORS["et"], "#eda100"])
HEAT_CMAP = LinearSegmentedColormap.from_list("heat", [(1, 0.82, 0.70, 0), (0.92, 0.41, 0.20, 0.75), (0.55, 0.15, 0.02, 1)])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8985", "axes.labelcolor": "#52514e",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, out, name):
    path = Path(out) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _cohort_order(values):
    return [c for c in COHORT_COLORS if c in set(values)]


def cohorts(patients, scans, out):
    """Stacked bars: patients and scans per cohort, by split (counts only, nothing read from locked patients)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, table, title in ((axes[0], patients, "Patients per cohort"), (axes[1], scans, "Scans (timepoints) per cohort")):
        order = _cohort_order(table.index)[::-1]
        left = np.zeros(len(order))
        for split in [s for s in SPLIT_COLORS if s in table.columns]:
            values = table.loc[order, split].to_numpy()
            ax.barh([COHORT_NAMES[c] for c in order], values, left=left, color=SPLIT_COLORS[split], label=split,
                    edgecolor="white", linewidth=1)
            left += values
        for y, total in enumerate(left):
            ax.text(total, y, f" {int(total):,}", va="center", color="#0b0b0b")
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, left.max() * 1.15)
    present = [s for s in SPLIT_COLORS if s in set(patients.columns) | set(scans.columns)]
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=SPLIT_COLORS[s]) for s in present], present, title="split",
               loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    return _save(fig, out, "f01_cohorts")


def timepoints(per_patient, out):
    """How many scans (timepoints) each patient has, for the cohorts that follow patients over time."""
    cohorts_ = [c for c in _cohort_order(per_patient.dataset_source) if per_patient[per_patient.dataset_source == c].n.max() > 1]
    if not cohorts_:
        return None
    fig, axes = plt.subplots(1, len(cohorts_), figsize=(4.2 * len(cohorts_), 3), squeeze=False)
    for ax, c in zip(axes[0], cohorts_):
        counts = per_patient[per_patient.dataset_source == c].n.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=COHORT_COLORS[c], width=0.7)
        ax.set(title=COHORT_NAMES[c], xlabel="scans per patient", ylabel="patients")
        ax.set_xticks(counts.index)
        ax.grid(axis="x", visible=False)
    fig.suptitle("Longitudinal cohorts: scans per patient", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "f02_timepoints")


def _overlay(ax, image, mask, title=None, alpha=0.55):
    ax.imshow(np.rot90(image), cmap="gray", vmin=np.percentile(image[image != 0], 1) if (image != 0).any() else None,
              vmax=np.percentile(image[image != 0], 99.5) if (image != 0).any() else None)
    if mask is not None:
        m = np.rot90(np.asarray(mask, dtype=np.int16))
        ax.imshow(np.ma.masked_equal(m, 0), cmap=MASK_CMAP, vmin=0, vmax=4, alpha=alpha, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)


def _mask_legend(fig):
    handles = [plt.Rectangle((0, 0), 1, 1, color=TISSUE_COLORS[t]) for t in ("ncr", "ed", "et")]
    fig.legend(handles, [TISSUE_NAMES[t] for t in ("ncr", "ed", "et")], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))


def examples(cases, out):
    """One row per example patient: the four MRI types and FLAIR with the expert mask, on the slice with most tumor.

    cases: list of (label, {"t1", "t1ce", "t2", "flair", "seg"} arrays, slice index).
    """
    fig, axes = plt.subplots(len(cases), 5, figsize=(11, 2.4 * len(cases)))
    axes = np.atleast_2d(axes)
    for row, (label, vols, z) in zip(axes, cases):
        for ax, m in zip(row[:4], MODALITIES):
            _overlay(ax, vols[m][:, :, z], None, MOD_NAMES[m])
        _overlay(row[4], vols["flair"][:, :, z], vols["seg"][:, :, z], "FLAIR + expert mask")
        row[0].text(-0.08, 0.5, label, transform=row[0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    _mask_legend(fig)
    fig.suptitle("The four MRI types of three training patients (small, typical and large tumor)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    return _save(fig, out, "f03_examples")


def multiplanar(vols, centre, label, out):
    """Axial, coronal and sagittal FLAIR and T1ce through the tumor centre, without and with the mask."""
    x, y, z = (int(round(c)) for c in centre)
    cuts = {"axial": (slice(None), slice(None), z), "coronal": (slice(None), y, slice(None)),
            "sagittal": (x, slice(None), slice(None))}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.6))
    for col, (plane, cut) in enumerate(cuts.items()):
        _overlay(axes[0, col], vols["t1ce"][cut], None, f"{plane} — T1ce")
        _overlay(axes[1, col], vols["flair"][cut], vols["seg"][cut], f"{plane} — FLAIR + mask")
    _mask_legend(fig)
    fig.suptitle(f"Three views through the tumor centre ({label})", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    return _save(fig, out, "f04_multiplanar")


def volumes(df, out):
    """Distribution of tumor volumes per region and cohort (log scale: sizes span three orders of magnitude)."""
    regions = (("vol_wt_ml", "Whole tumor (WT)"), ("vol_tc_ml", "Tumor core (TC)"), ("vol_et_ml", "Enhancing tumor (ET)"))
    bins = np.logspace(-1, np.log10(max(df.vol_wt_ml.max(), 1) * 1.1), 40)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=False)
    for ax, (col, title) in zip(axes, regions):
        for c in _cohort_order(df.dataset):
            v = df.loc[df.dataset == c, col]
            v = v[v > 0]
            ax.hist(v, bins=bins, histtype="step", linewidth=2, color=COHORT_COLORS[c],
                    label=f"{COHORT_NAMES[c]} (median {v.median():.0f} mL)", density=True)
        ax.set_xscale("log")
        ax.set(title=title, xlabel="volume (mL, log scale)", ylabel="share of scans (density)")
        ax.legend(fontsize=7.5, loc="upper left")
    fig.tight_layout()
    return _save(fig, out, "f05_volumes")


def imbalance(voxel_shares, brain_share, out):
    """Class imbalance: share of brain voxels per tissue, and per-scan tumor share of the brain."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    names = [TISSUE_NAMES[t] for t in TISSUES]
    shares = [voxel_shares[t] * 100 for t in TISSUES]
    bars = axes[0].barh(names[::-1], shares[::-1], color=[TISSUE_COLORS[t] for t in TISSUES][::-1])
    axes[0].set_xscale("log")
    axes[0].bar_label(bars, labels=[f" {s:.2f} %" for s in shares[::-1]], color="#0b0b0b")
    axes[0].set(title="Share of all brain voxels (BraTS 2021, log scale)", xlabel="% of brain voxels")
    axes[0].set_xlim(min(shares) / 2, 400)
    axes[0].grid(axis="y", visible=False)
    axes[1].hist(brain_share * 100, bins=30, color=COHORT_COLORS["brats2021"], edgecolor="white")
    axes[1].axvline(np.median(brain_share * 100), color="#0b0b0b", linestyle="--", linewidth=1)
    axes[1].text(np.median(brain_share * 100), axes[1].get_ylim()[1] * 0.92, f"  median {np.median(brain_share) * 100:.1f} %")
    axes[1].set(title="Tumor share of each patient's brain", xlabel="whole tumor, % of brain volume", ylabel="patients")
    fig.tight_layout()
    return _save(fig, out, "f06_imbalance")


def composition(df, out):
    """What the whole tumor is made of, per scan: share of NCR, ED and ET."""
    cohorts_ = _cohort_order(df.dataset)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, t in zip(axes, ("ncr", "ed", "et")):
        data = [df.loc[df.dataset == c, f"share_{t}_of_wt"].dropna() * 100 for c in cohorts_]
        parts = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False, medianprops={"color": "#0b0b0b"})
        for patch, c in zip(parts["boxes"], cohorts_):
            patch.set(facecolor=COHORT_COLORS[c], alpha=0.8, edgecolor="white")
        ax.set_xticks(range(1, len(cohorts_) + 1), [COHORT_NAMES[c] for c in cohorts_], fontsize=8)
        ax.set_title(TISSUE_NAMES[t], color="#0b0b0b")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of the whole tumor")
    fig.suptitle("Tumor composition per scan", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "f07_composition")


def location(freq, mean_flair, n, out):
    """Where tumors are: share of BraTS 2021 patients with tumor at each point, averaged through each view."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    views = (("axial (top-down)", 2), ("coronal (front)", 1), ("sagittal (side)", 0))
    for ax, (title, axis) in zip(axes, views):
        heat = freq.max(axis=axis) / n * 100
        background = mean_flair.mean(axis=axis) if mean_flair is not None else np.zeros_like(heat)
        ax.imshow(np.rot90(background), cmap="gray")
        im = ax.imshow(np.rot90(heat), cmap=HEAT_CMAP, vmin=0, vmax=max(heat.max(), 1e-6))
        ax.set_title(title)
        ax.set_axis_off()
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cbar.set_label("% of patients with tumor (max along the view)")
    fig.suptitle(f"Tumor location frequency, BraTS 2021 training + validation ({n} patients)", x=0.01, ha="left",
                 fontweight="bold")
    return _save(fig, out, "f08_location")


def components(df, out):
    """Number of separate tumor parts per scan, and how many scans have no enhancing tumor."""
    cohorts_ = _cohort_order(df.dataset)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    cats = ["0", "1", "2", "3", "4+"]
    width = 0.8 / len(cohorts_)
    for i, c in enumerate(cohorts_):
        k = df.loc[df.dataset == c, "wt_components"].clip(upper=4)
        share = [(k == j).mean() * 100 for j in range(5)]
        axes[0].bar(np.arange(5) + (i - (len(cohorts_) - 1) / 2) * width, share, width * 0.92, color=COHORT_COLORS[c],
                    label=COHORT_NAMES[c])
    axes[0].set_xticks(range(5), cats)
    axes[0].set(title="Separate tumor parts per scan (≥ 0.1 mL)", xlabel="parts", ylabel="% of scans")
    axes[0].legend()
    axes[0].grid(axis="x", visible=False)
    no_et = [(~df.loc[df.dataset == c, "et_present"].astype(bool)).mean() * 100 for c in cohorts_]
    bars = axes[1].bar([COHORT_NAMES[c] for c in cohorts_], no_et, color=[COHORT_COLORS[c] for c in cohorts_], width=0.55)
    axes[1].bar_label(bars, labels=[f"{v:.1f} %" for v in no_et])
    axes[1].set(title="Scans with no enhancing tumor in the expert mask", ylabel="% of scans")
    axes[1].grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, out, "f09_components_et")


def intensity_by_tissue(hist, out, dataset="brats2021"):
    """Cleaned (z-scored) intensity of each tissue in each MRI type: which MRI type shows which tumor part."""
    centres = (HIST_EDGES[:-1] + HIST_EDGES[1:]) / 2
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    for ax, m in zip(axes, MODALITIES):
        for t in TISSUES:
            counts = hist[m][t].astype(float)
            if counts.sum():
                ax.plot(centres, counts / counts.sum() / np.diff(HIST_EDGES), color=TISSUE_COLORS[t], linewidth=2,
                        label=TISSUE_NAMES[t])
        ax.set(title=MOD_NAMES[m], xlabel="z-score (cleaned scan)")
        ax.set_xlim(-3, 7)
    axes[0].set_ylabel("density")
    axes[-1].legend(fontsize=7.5)
    fig.suptitle(f"Intensity of each tissue per MRI type ({COHORT_NAMES[dataset]} sample)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "f10_intensity_by_tissue")


def raw_vs_clean(raw, clean, out):
    """Intensity range (1st-99th percentile, dot = median) of each sample scan before and after cleaning."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 5))
    for col, m in enumerate(MODALITIES):
        for row, (table, title) in enumerate(((raw, "raw scan"), (clean, "after cleaning (z-score)"))):
            ax = axes[row, col]
            sub = table[table.modality == m].reset_index(drop=True)
            ax.vlines(sub.index, sub.p1, sub.p99, color=COHORT_COLORS["brats2021"], linewidth=3)
            ax.plot(sub.index, sub.p50, "o", color="#0b0b0b", markersize=4)
            ax.set_title(f"{MOD_NAMES[m]} — {title}", fontsize=9)
            ax.set_xticks([])
            ax.grid(axis="x", visible=False)
        axes[1, col].set_xlabel("sample patients")
    axes[0, 0].set_ylabel("intensity")
    axes[1, 0].set_ylabel("z-score")
    fig.suptitle("Raw intensities differ from scan to scan; cleaning puts every scan on the same scale",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "f11_raw_vs_clean")


def train_vs_val(df, ks, out):
    """Cumulative distribution of whole-tumor volume in the training and validation patients."""
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for split in ("train", "val"):
        v = np.sort(df.loc[df.split == split, "vol_wt_ml"])
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=SPLIT_COLORS[split], linewidth=2, label=f"{split} ({len(v)})")
    ax.set_xscale("log")
    ax.set(title="Whole-tumor volume: training vs validation", xlabel="volume (mL, log scale)",
           ylabel="share of patients ≤ volume")
    ax.text(0.98, 0.05, f"KS test p = {ks:.2f}", transform=ax.transAxes, ha="right")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, out, "f12_train_vs_val")


def volume_vs_dice(merged, out):
    """Selected model (nnU-Net) validation Dice against the size of each region in the expert mask."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, (region, col) in zip(axes, (("WT", "vol_wt_ml"), ("TC", "vol_tc_ml"), ("ET", "vol_et_ml"))):
        d = merged[[col, f"dice_{region}"]].dropna()
        empty = d[col] == 0
        ax.scatter(d.loc[~empty, col], d.loc[~empty, f"dice_{region}"], s=12, color=COHORT_COLORS["brats2021"],
                   alpha=0.7, edgecolor="white", linewidth=0.5)
        if empty.any():
            ax.scatter(np.full(empty.sum(), 0.15), d.loc[empty, f"dice_{region}"], s=22, color="#e34948", marker="x",
                       label=f"no {region} in expert mask ({empty.sum()})")
            ax.legend(loc="lower right", fontsize=7.5)
        ax.set_xscale("log")
        ax.set(title=region, xlabel="region volume in expert mask (mL)")
    axes[0].set_ylabel("Dice")
    fig.suptitle("Smaller regions are harder: nnU-Net validation Dice by tumor size", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "f13_volume_vs_dice")


def difficult(cases, out):
    """Gallery of hard cases: (label, reason, flair slice, mask slice)."""
    fig, axes = plt.subplots(1, len(cases), figsize=(3 * len(cases), 3.5))
    for ax, (label, reason, flair, mask) in zip(np.atleast_1d(axes), cases):
        _overlay(ax, flair, mask)
        ax.set_title(f"{label}\n{reason}", fontsize=8)
    _mask_legend(fig)
    fig.suptitle("Difficult cases", x=0.01, y=1.0, ha="left", va="bottom", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(fig, out, "f14_difficult_cases")


def ed_vs_core(df, rho, out):
    """Edema volume against tumor-core volume per scan."""
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for c in _cohort_order(df.dataset):
        d = df[(df.dataset == c) & (df.vol_tc_ml > 0) & (df.vol_ed_ml > 0)]
        ax.scatter(d.vol_tc_ml, d.vol_ed_ml, s=10, alpha=0.6, color=COHORT_COLORS[c], label=COHORT_NAMES[c],
                   edgecolor="white", linewidth=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(title="Edema vs tumor core", xlabel="tumor core (mL)", ylabel="edema (mL)")
    ax.text(0.98, 0.05, f"BraTS 2021 Spearman ρ = {rho:.2f}", transform=ax.transAxes, ha="right")
    ax.legend(loc="upper left", fontsize=7.5)
    fig.tight_layout()
    return _save(fig, out, "f15_edema_vs_core")
