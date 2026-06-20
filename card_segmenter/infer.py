"""Run the trained model on new photos. THIS is the replacement for CardFinder.java.

For each input image it writes, at full resolution:
  <name>_mask.png     binary card mask (255 = card)  -> same role as Java 9_card_mask.png
  <name>_overlay.png  colored instance overlay for humans

Run on one image:   python infer.py --ckpt checkpoints/best.pt --input photo.jpg --out out/
Run on a folder:    python infer.py --ckpt checkpoints/best.pt --input some_dir/ --out out/"""

import argparse
import glob
import os

import cv2
import numpy as np
import torch

from dataset import eval_tf
from model import load_model
from viz import overlay_instances


def predict_mask(model, device, img_rgb, h, w):
    """Returns a full-resolution binary mask (uint8, 0/255) for one RGB image."""
    aug = eval_tf(h, w)(image=np.ascontiguousarray(img_rgb), mask=np.zeros((h, w), np.uint8))
    x = aug["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    small = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(small, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)


def gather_inputs(path):
    if os.path.isdir(path):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            files += glob.glob(os.path.join(path, ext))
            files += glob.glob(os.path.join(path, ext.upper()))
        return sorted(files)
    return [path]


def run(ckpt, input_path, out_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(ckpt, device)
    h, w = cfg["train"]["img_height"], cfg["train"]["img_width"]
    os.makedirs(out_dir, exist_ok=True)

    for img_path in gather_inputs(input_path):
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip (unreadable): {img_path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = predict_mask(model, device, rgb, h, w)
        overlay = overlay_instances(rgb, mask)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(out_dir, f"{stem}_mask.png"), mask)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_overlay.png"),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        n_cards = cv2.connectedComponents((mask > 127).astype(np.uint8))[0] - 1
        print(f"{stem}: found {n_cards} card region(s)")

    print(f"Done. Results in: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--input", required=True, help="an image file or a folder of images")
    ap.add_argument("--out", default="outputs/inference")
    args = ap.parse_args()
    run(args.ckpt, args.input, args.out)
