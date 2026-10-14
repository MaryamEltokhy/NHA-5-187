"""Figures for the confidence-map check (M3-T04). Colors follow the EDA figures (src/bmts/eda/figures.py)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from bmts.common.constants import REGIONS  # noqa: E402

REGION_COLORS = {"WT": "#2a78d6", "TC": "#eb6834", "ET": "#e87ba4"}
MASK_CMAP = ListedColormap([(0, 0, 0, 0), "#eb6834", "#1baf7a", "#e87ba4"])        # NCR, ED, ET as in the EDA
UNC_CMAP = LinearSegmentedColormap.from_list("unc", [(1, 0.95, 0.4, 0), (1, 0.75, 0.1, 0.8), (0.85, 0.1, 0.1, 1)])
ERR_CMAP = ListedColormap([(0, 0, 0, 0), "#e34948", "#2a78d6"])                    # 1 missed, 2 extra

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
                     "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "axes.axisbelow": True, "legend.frameon": False})


def _save(fig, out, name):
    path = Path(out) / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _gray(ax, image):
    nz = image[image != 0]
    lo, hi = (np.percentile(nz, (1, 99.5)) if nz.size else (None, None))
    ax.imshow(np.flipud(image), cmap="gray", vmin=lo, vmax=hi)
    ax.set_axis_off()


def overlay_row(axes, sl, title=None):
    """Four panels for one axial slice: FLAIR, predicted mask, confidence map, mistakes against the expert mask.

    sl: dict with flair, seg, gt, uncertainty 2-D arrays (z, y, x order already sliced at one z).
    """
    _gray(axes[0], sl["flair"])
    axes[0].set_title("FLAIR" if title is None else f"{title}\nFLAIR", fontsize=8.5)
    _gray(axes[1], sl["flair"])
    axes[1].imshow(np.flipud(np.ma.masked_equal(sl["seg"], 0)), cmap=MASK_CMAP, vmin=0, vmax=3, alpha=0.6,
                   interpolation="nearest")
    axes[1].set_title("predicted mask", fontsize=8.5)
    _gray(axes[2], sl["flair"])
    im = axes[2].imshow(np.flipud(np.ma.masked_less(sl["uncertainty"], 0.02)), cmap=UNC_CMAP, vmin=0, vmax=1,
                        interpolation="nearest")
    axes[2].set_title("confidence map (uncertainty)", fontsize=8.5)
    _gray(axes[3], sl["flair"])
    wt_pred, wt_gt = sl["seg"] > 0, sl["gt"] > 0
    err = np.zeros(sl["seg"].shape, np.uint8)
    err[wt_gt & ~wt_pred] = 1
    err[wt_pred & ~wt_gt] = 2
    axes[3].imshow(np.flipud(np.ma.masked_equal(err, 0)), cmap=ERR_CMAP, vmin=0, vmax=2, interpolation="nearest")
    axes[3].set_title("whole-tumor mistakes", fontsize=8.5)
    return im


def gallery(rows, out, name="f01_examples"):
    """rows: list of (label, slice dict). One row per case."""
    fig, axes = plt.subplots(len(rows), 4, figsize=(11, 2.9 * len(rows)))
    axes = np.atleast_2d(axes)
    im = None
    for row_axes, (label, sl) in zip(axes, rows):
        im = overlay_row(row_axes, sl, label)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), location="right", shrink=0.5, pad=0.02)
    cbar.set_label("uncertainty (0 = sure, 1 = no idea)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#eb6834", "#1baf7a", "#e87ba4", "#e34948", "#2a78d6")]
    fig.legend(handles, ["necrotic core", "edema", "enhancing tumor", "missed tumor", "extra (false) tumor"],
               loc="lower center", ncol=5, bbox_to_anchor=(0.45, -0.01))
    fig.suptitle("Predicted mask, confidence map and mistakes (validation patients)", x=0.01, ha="left",
                 fontweight="bold")
    return _save(fig, out, name)


def reliability(calibration, out, name="f02_reliability"):
    """Predicted probability vs how often the voxel really is in the region (diagonal = perfectly calibrated)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, region in zip(axes, REGIONS):
        conf, freq, n = calibration.curve(region)
        ax.plot([0, 1], [0, 1], color="#8a8985", linestyle="--", linewidth=1, label="perfect")
        ax.plot(conf, freq, marker="o", markersize=4, linewidth=2, color=REGION_COLORS[region], label=region)
        ax.set(title=f"{region} (ECE {calibration.ece(region):.4f})", xlabel="predicted probability",
               xlim=(0, 1), ylim=(0, 1))
    axes[0].set_ylabel("share of voxels really in the region")
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibration: does a probability of 0.7 mean 70 %?", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


def uncertainty_vs_dice(df, rho, out, name="f03_uncertainty_vs_dice"):
    """Case-level check: patients with a less certain map should be the ones with lower Dice."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, region in zip(axes, REGIONS):
        ax.scatter(df[f"uncertainty_{region}"], df[f"dice_{region}"], s=14, alpha=0.7, color=REGION_COLORS[region],
                   edgecolor="white", linewidth=0.5)
        ax.set(title=f"{region} (Spearman ρ = {rho[region]:.2f})", xlabel="mean uncertainty near the region")
    axes[0].set_ylabel("Dice against the expert mask")
    fig.suptitle("Case uncertainty vs Dice", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)


def error_detection(detector, out, name="f04_error_detection"):
    """Uncertainty of the voxels the model got wrong vs right."""
    centres = (detector.edges[:-1] + detector.edges[1:]) / 2
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, region in zip(axes, REGIONS):
        for counts, label, color in ((detector.right[region], "right", "#8a8985"),
                                     (detector.wrong[region], "wrong", REGION_COLORS[region])):
            share = counts / max(counts.sum(), 1) * 100
            ax.plot(centres, share, color=color, linewidth=2, label=f"{label} voxels")
        ax.set_yscale("log")
        ax.set(title=f"{region} (AUROC {detector.auroc(region):.3f})", xlabel="uncertainty")
    axes[0].set_ylabel("% of voxels (log scale)")
    axes[0].legend()
    fig.suptitle("Uncertainty of the voxels the model got wrong vs right", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, name)
