"""Evaluate the trained model on the held-out TEST dives, save qualitative overlays,
and (optionally) compare against the classical Java baseline.

Run:  python evaluate.py --config config.yaml --ckpt checkpoints/best.pt
Baseline comparison (optional):
      python evaluate.py --ckpt checkpoints/best.pt --baseline_dir path/to/java/masks
where that folder holds the Java pipeline's 9_card_mask.png files, renamed to match
each test image's stem (e.g. dive07_frame02.png)."""

import argparse
import os

import cv2
import numpy as np
import torch
import yaml

from dataset import IMAGENET_MEAN, IMAGENET_STD, eval_tf, make_splits
from metrics import binarize, dice_score, iou_score
from model import load_model
from viz import overlay_instances


def _load_for_model(img_path, h, w):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    aug = eval_tf(h, w)(image=np.ascontiguousarray(img), mask=np.zeros((h, w), np.uint8))
    return aug["image"].unsqueeze(0), img  # tensor (1,3,h,w), original RGB


def _gt_mask(mask_path, h, w):
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((m > 127).astype(np.float32))[None, None]  # (1,1,h,w)


def run(cfg, ckpt, baseline_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt_cfg = load_model(ckpt, device)
    cfg = cfg or ckpt_cfg
    h, w = cfg["train"]["img_height"], cfg["train"]["img_width"]

    _, _, test_items, info = make_splits(cfg)
    print("Test dives:", info["test_dives"])
    if not test_items:
        raise RuntimeError("No test images. Set test_dives in config.yaml.")

    out_dir = cfg["output"]["pred_dir"]
    os.makedirs(out_dir, exist_ok=True)
    n_qual = cfg["output"]["num_qualitative"]

    model_ious, model_dices, base_ious = [], [], []
    for k, (img_path, mask_path) in enumerate(test_items):
        x, orig_rgb = _load_for_model(img_path, h, w)
        gt = _gt_mask(mask_path, h, w)
        with torch.no_grad():
            pred = binarize(model(x.to(device))).cpu()
        model_ious.append(iou_score(pred, gt))
        model_dices.append(dice_score(pred, gt))

        # Optional classical baseline on the same image, same resolution.
        if baseline_dir:
            stem = os.path.splitext(os.path.basename(img_path))[0]
            bpath = os.path.join(baseline_dir, stem + ".png")
            if os.path.exists(bpath):
                b = cv2.resize(cv2.imread(bpath, cv2.IMREAD_GRAYSCALE), (w, h),
                               interpolation=cv2.INTER_NEAREST)
                b = torch.from_numpy((b > 127).astype(np.float32))[None, None]
                base_ious.append(iou_score(b, gt))

        # Save a few side-by-side overlays for the README / interview.
        if k < n_qual:
            pred_mask = (pred[0, 0].numpy() * 255).astype(np.uint8)
            pred_full = cv2.resize(pred_mask, (orig_rgb.shape[1], orig_rgb.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            overlay = overlay_instances(orig_rgb, pred_full)
            stem = os.path.splitext(os.path.basename(img_path))[0]
            cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"),
                        cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"\n=== TEST RESULTS  ({len(test_items)} images) ===")
    print(f"Learned model : IoU {np.mean(model_ious):.3f}   Dice {np.mean(model_dices):.3f}")
    if base_ious:
        print(f"Classical base: IoU {np.mean(base_ious):.3f}   "
              f"(over {len(base_ious)} images with a baseline mask)")
        print(f"Improvement   : {np.mean(model_ious) - np.mean(base_ious):+.3f} IoU")
    print(f"Overlays saved to: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--baseline_dir", default=None)
    args = ap.parse_args()
    cfg = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
    run(cfg, args.ckpt, args.baseline_dir)
