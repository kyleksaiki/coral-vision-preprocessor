"""Evaluate the trained coral model on the held-out TEST trays and save overlays.

Run:  python evaluate.py --config config.yaml --ckpt checkpoints/best.pt"""

import argparse
import os

import cv2
import numpy as np
import torch
import yaml

from dataset import eval_tf, make_splits
from metrics import binarize, dice_score, iou_score
from model import load_model
from viz import overlay_instances


def _load_for_model(img_path, h, w):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    aug = eval_tf(h, w)(image=np.ascontiguousarray(img), mask=np.zeros((h, w), np.uint8))
    return aug["image"].unsqueeze(0), img


def _gt_mask(mask_path, h, w):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((m > 127).astype(np.float32))[None, None]


def run(cfg, ckpt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_cfg = load_model(ckpt, device)
    cfg = cfg or ckpt_cfg
    h, w = cfg["train"]["img_height"], cfg["train"]["img_width"]

    _, _, test_items, info = make_splits(cfg)
    print("Test trays:", info["test_trays"])
    if not test_items:
        raise RuntimeError("No test crops. Set test_dives in config.yaml.")

    out_dir = cfg["output"]["pred_dir"]
    os.makedirs(out_dir, exist_ok=True)
    n_qual = cfg["output"]["num_qualitative"]

    ious, dices = [], []
    for k, (img_path, mask_path) in enumerate(test_items):
        x, orig_rgb = _load_for_model(img_path, h, w)
        gt = _gt_mask(mask_path, h, w)
        with torch.no_grad():
            pred = binarize(model(x.to(device))).cpu()
        ious.append(iou_score(pred, gt))
        dices.append(dice_score(pred, gt))

        if k < n_qual:
            pred_mask = (pred[0, 0].numpy() * 255).astype(np.uint8)
            pred_full = cv2.resize(pred_mask, (orig_rgb.shape[1], orig_rgb.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            overlay = overlay_instances(orig_rgb, pred_full)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"\n=== CORAL TEST RESULTS  ({len(test_items)} crops) ===")
    print(f"IoU {np.mean(ious):.3f}   Dice {np.mean(dices):.3f}")
    print(f"Overlays saved to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="defaults to the config in the checkpoint")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    args = ap.parse_args()
    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    run(cfg, args.ckpt)
