"""The model: a U-Net with a ResNet-34 encoder pretrained on ImageNet.

Why this choice (the interview answer):
- U-Net is the standard, reliable workhorse for binary segmentation with limited data.
- A pretrained ResNet-34 encoder means the network already knows edges/shapes, so you
  need far fewer labeled images to get good results.
- It's small and fast — no prompt needed at inference, unlike SAM.
To try alternatives, swap smp.Unet for smp.FPN or smp.DeepLabV3Plus (same arguments)."""

import segmentation_models_pytorch as smp
import torch


def build_model(cfg):
    return smp.Unet(
        encoder_name=cfg["model"]["encoder"],
        encoder_weights=cfg["model"]["encoder_weights"],
        in_channels=3,   # RGB
        classes=1,       # one output channel: card vs not-card
    )


def load_model(ckpt_path, device="cpu"):
    """Rebuild the model from a saved checkpoint (which also stores its config)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg
