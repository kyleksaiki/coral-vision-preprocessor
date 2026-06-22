"""IoU and Dice for binary masks. These are what you report for coral -- never pixel
accuracy, which lies when most of a crop is 'not coral'."""

import torch


@torch.no_grad()
def binarize(logits, threshold=0.5):
    """sigmoid(logits) > threshold -> hard 0/1 mask."""
    return (torch.sigmoid(logits) > threshold).float()


def iou_score(pred, target, eps=1e-7):
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = (pred + target - pred * target).sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


def dice_score(pred, target, eps=1e-7):
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (denom + eps)).mean().item()
