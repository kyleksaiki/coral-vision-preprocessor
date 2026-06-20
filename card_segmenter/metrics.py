"""IoU and Dice for binary masks. Written out longhand so it's obvious what's
measured — these are the metrics you report, never pixel accuracy (which lies when
most of the image is 'not card')."""

import torch


@torch.no_grad()
def binarize(logits, threshold=0.5):
    """Model outputs raw 'logits'. Sigmoid maps them to 0..1, then we threshold
    to a hard 0/1 mask."""
    return (torch.sigmoid(logits) > threshold).float()


def iou_score(pred, target, eps=1e-7):
    """Intersection-over-Union. pred and target are {0,1} tensors shaped (N,1,H,W).
    1.0 = perfect overlap, 0.0 = no overlap."""
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def dice_score(pred, target, eps=1e-7):
    """Dice (a.k.a. F1 for pixels). Similar to IoU but rewards overlap a bit more."""
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()
