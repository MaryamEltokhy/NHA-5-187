"""MONAI transform chains for training and validation (M1-T05, made cache-friendly in M2-T03).

Input: one cleaned case = {t1, t1ce, t2, flair, seg} file paths (output of M1-T04, already RAS,
240x240x155, z-scored per modality, masks in the canonical label scheme).
Output: "image" (4, H, W, D) float32 in MODALITIES order, "label" (3, H, W, D) float32 region masks (TC, WT, ET).

Training chain: the deterministic part (load, crop to the brain, store the image as float16 and the mask as uint8)
comes first, so a CacheDataset keeps about 30 MB per patient in RAM instead of about 110 MB; the random part
(patch sampling, augmentation, region channels) runs on every draw.
"""
from __future__ import annotations

import math

import torch
from monai import transforms as T
from monai.data import MetaTensor

from bmts.common.constants import MODALITIES, REGIONS, SEG

FOREGROUND = "fg"  # 1-channel whole-tumor mask written next to the region channels


Z_LIMIT = 20.0  # cleaned scans are z-scored: real tissue stays far inside +-20 standard deviations


def sanitize(x):
    """Replace NaN/inf and cap intensities at +-Z_LIMIT; a scan with almost no contrast (divided by a tiny standard
    deviation when it was z-scored) would otherwise feed huge values into the network and overflow half precision."""
    return torch.nan_to_num(x, nan=0.0, posinf=Z_LIMIT, neginf=-Z_LIMIT).clamp(-Z_LIMIT, Z_LIMIT)


def nonzero(x):
    """Brain voxels: the cleaned scans are z-scored, so the brain has negative values too; background is exactly 0."""
    return x != 0


class SegToRegionsd(T.MapTransform):
    """Canonical label map (1 channel) -> one binary channel per region in REGIONS (TC, WT, ET).

    Also writes a 1-channel whole-tumor mask under `foreground_key`.
    Labels outside every region (background, resection cavity 4) end up 0 in every channel.
    """

    def __init__(self, keys=SEG, label_key="label", foreground_key=FOREGROUND, regions=REGIONS):
        super().__init__(keys)
        self.label_key, self.foreground_key = label_key, foreground_key
        self.regions = {name: torch.tensor(labels) for name, labels in regions.items()}

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            seg = torch.as_tensor(d[key])[0].float().round().long()
            channels = torch.stack([torch.isin(seg, labels) for labels in self.regions.values()]).float()
            whole = channels[list(self.regions).index("WT")][None] if "WT" in self.regions else channels.amax(0, keepdim=True)
            if isinstance(d[key], MetaTensor):
                channels = MetaTensor(channels, meta=d[key].meta)
                whole = MetaTensor(whole, meta=d[key].meta)
            d[self.label_key], d[self.foreground_key] = channels, whole
            del d[key]
        return d


def load_transforms(normalize=False):
    """Read the 5 files of a case and stack the 4 MRI types as channels ("image"); the mask stays in "seg"."""
    keys = [*MODALITIES, SEG]
    chain = [
        T.LoadImaged(keys, image_only=True),
        T.EnsureChannelFirstd(keys),
        T.ConcatItemsd(list(MODALITIES), "image"),
        T.DeleteItemsd(list(MODALITIES)),
        T.Orientationd(["image", SEG], axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        T.EnsureTyped("image", dtype=torch.float32),
        T.Lambdad("image", func=sanitize),
    ]
    if normalize:  # only for scans that were not z-scored by the cleaning step
        chain.append(T.NormalizeIntensityd("image", nonzero=True, channel_wise=True))
    return chain


def train_transforms(cfg):
    """Loading + random tumor-centred patches + augmentation (flips, small rotation/scaling, intensity, gamma)."""
    roi = cfg["roi_size"]
    aug, sampling = cfg["augment"], cfg["sampling"]
    chain = load_transforms(cfg.get("normalize", False)) + [
        # deterministic: this is what a cache keeps
        T.CropForegroundd(["image", SEG], source_key="image", select_fn=nonzero, allow_smaller=True),
        T.SpatialPadd(["image", SEG], spatial_size=roi),
        T.CastToTyped(["image", SEG], dtype=(torch.float16, torch.uint8)),
        # random: runs on every draw; two thirds of the patches (pos:neg = 2:1) are centred on tumor
        T.RandCropByPosNegLabeld(["image", SEG], label_key=SEG, spatial_size=roi, pos=sampling["pos"],
                                 neg=sampling["neg"], num_samples=sampling.get("samples_per_case", 1)),
        T.CastToTyped("image", dtype=torch.float32),
        SegToRegionsd(SEG),
        T.DeleteItemsd(FOREGROUND),
    ]
    for axis in range(3):
        chain.append(T.RandFlipd(["image", "label"], prob=aug["flip_prob"], spatial_axis=axis))
    if aug["affine_prob"] > 0:
        angle = math.radians(aug["rotate_deg"])
        chain.append(T.RandAffined(["image", "label"], prob=aug["affine_prob"], rotate_range=(angle,) * 3,
                                   scale_range=(aug["scale"],) * 3, mode=("bilinear", "nearest"),
                                   padding_mode="zeros"))
    if aug.get("elastic_prob", 0) > 0:
        chain.append(T.Rand3DElasticd(["image", "label"], sigma_range=(5, 8), magnitude_range=(100, 200),
                                      prob=aug["elastic_prob"], mode=("bilinear", "nearest")))
    chain += [
        T.RandScaleIntensityd("image", factors=0.1, prob=aug["intensity_prob"]),
        T.RandShiftIntensityd("image", offsets=0.1, prob=aug["intensity_prob"]),
        T.RandAdjustContrastd("image", gamma=(0.7, 1.5), prob=aug["gamma_prob"]),
        T.EnsureTyped(["image", "label"], dtype=torch.float32, track_meta=False),
    ]
    return T.Compose(chain)


def val_transforms(cfg):
    """Loading only: the whole 240x240x155 volume with region channels, for sliding-window evaluation."""
    return T.Compose(load_transforms(cfg.get("normalize", False)) + [
        SegToRegionsd(SEG),
        T.DeleteItemsd(FOREGROUND),
        T.EnsureTyped(["image", "label"], dtype=torch.float32, track_meta=False),
    ])
