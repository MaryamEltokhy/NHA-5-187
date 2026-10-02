"""3D U-Net baseline (M2-T03): MONAI's residual UNet, 4 MRI types in, one sigmoid channel per region (TC, WT, ET) out."""
from monai.networks.nets import UNet

from bmts.common.constants import MODALITIES, REGIONS


def build_unet3d(cfg):
    """Build the network from a model config (configs/models/unet3d_baseline.yaml)."""
    return UNet(
        spatial_dims=3,
        in_channels=len(MODALITIES),
        out_channels=len(REGIONS),
        channels=tuple(cfg["channels"]),
        strides=tuple(cfg["strides"]),
        num_res_units=cfg.get("num_res_units", 2),
        norm=cfg.get("norm", "instance"),
        dropout=cfg.get("dropout", 0.0),  # also enables Monte Carlo dropout for the confidence map (M3-T04)
    )
