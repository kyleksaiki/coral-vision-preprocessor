"""The model: a U-Net with a ResNet-34 encoder pretrained on ImageNet.

Identical to the card model -- coral is also binary segmentation (coral vs not).
The model architecture isn't the hard part for coral; the labeling convention and
the per-card crops are. To try alternatives, swap smp.Unet for smp.FPN or
smp.DeepLabV3Plus (same arguments)."""

import segmentation_models_pytorch as smp
import torch


def build_model(cfg):
    return smp.Unet(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=3,   # RGB
        classes=1,       # one output channel: coral vs not-coral
    )


def load_model(ckpt_path, device="cpu"):
    """Rebuild the model from a saved checkpoint (which also stores its config)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg
