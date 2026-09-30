"""MONAI transform chains for training and validation (M1-T05).

Input: one cleaned case = {t1, t1ce, t2, flair, seg} file paths (output of M1-T04, already RAS,
240x240x155, z-scored per modality, masks in the canonical label scheme).
Output: "image" (4, H, W, D) float32 in MODALITIES order, "label" (3, H, W, D) float32 region masks (TC, WT, ET).
"""
from __future__ import annotations

import math

import torch
from monai import transforms as T
from monai.data import MetaTensor

from bmts.common.constants import MODALITIES, REGIONS, SEG

FOREGROUND = "fg"  # temporary whole-tumor mask used only to pick training patches


def nonzero(x):
    """Brain voxels: the cleaned scans are z-scored, so the brain has negative values too; background is exactly 0."""
    return x != 0


class SegToRegionsd(T.MapTransform):
    """Canonical label map (1 channel) -> one binary channel per region in REGIONS (TC, WT, ET).

    Also writes a 1-channel whole-tumor mask under `foreground_key` for tumor-centred patch sampling.
    Labels outside every region (background, resection cavity 4) end up 0 in every channel.
    """

    def __init__(self, keys=SEG, label_key="label", foreground_key=FOREGROUND, regions=REGIONS):
        super().__init__(keys)
        self.label_key, self.foreground_key = label_key, foreground_key
        self.regions = {name: torch.tensor(labels) for name, labels in regions.items()}

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            seg = torch.as_tensor(d[key])[0].round().long()
            channels = torch.stack([torch.isin(seg, labels) for labels in self.regions.values()]).float()
            whole = channels[list(self.regions).index("WT")][None] if "WT" in self.regions else channels.amax(0, keepdim=True)
            if isinstance(d[key], MetaTensor):
                channels = MetaTensor(channels, meta=d[key].meta)
                whole = MetaTensor(whole, meta=d[key].meta)
            d[self.label_key], d[self.foreground_key] = channels, whole
            del d[key]
        return d


def load_transforms(normalize=False):
    """Read the 5 files of a case, stack the 4 MRI types as channels, turn the mask into region channels."""
    keys = [*MODALITIES, SEG]
    chain = [
        T.LoadImaged(keys, image_only=True),
        T.EnsureChannelFirstd(keys),
        T.ConcatItemsd(list(MODALITIES), "image"),
        T.DeleteItemsd(list(MODALITIES)),
        T.Orientationd(["image", SEG], axcodes="RAS", labels=(("L", "R"), ("P", "A"), ("I", "S"))),
        T.EnsureTyped("image", dtype=torch.float32),
    ]
    if normalize:  # only for scans that were not z-scored by the cleaning step
        chain.append(T.NormalizeIntensityd("image", nonzero=True, channel_wise=True))
    chain.append(SegToRegionsd(SEG))
    return chain


def train_transforms(cfg):
    """Loading + random tumor-centred patch + augmentation (flips, small rotation/scaling, intensity, gamma)."""
    roi = cfg["roi_size"]
    aug, sampling = cfg["augment"], cfg["sampling"]
    keys = ["image", "label", FOREGROUND]
    chain = load_transforms(cfg.get("normalize", False)) + [
        T.CropForegroundd(keys, source_key="image", select_fn=nonzero, allow_smaller=True),
        T.SpatialPadd(keys, spatial_size=roi),
        T.RandCropByPosNegLabeld(["image", "label"], label_key=FOREGROUND, spatial_size=roi,
                                 pos=sampling["pos"], neg=sampling["neg"], num_samples=1),
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
    """Loading only: the whole 240x240x155 volume, for sliding-window evaluation."""
    return T.Compose(load_transforms(cfg.get("normalize", False)) + [
        T.DeleteItemsd(FOREGROUND),
        T.EnsureTyped(["image", "label"], dtype=torch.float32, track_meta=False),
    ])
