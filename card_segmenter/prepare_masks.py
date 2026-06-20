"""Convert a Roboflow *Semantic Segmentation Masks* export into the simple layout
this project expects:

    data/images/<name>.jpg     the photo
    data/masks/<name>.png      binary mask, 255 = card, 0 = everything else

Roboflow encodes the 'card' class as some color/value (sometimes a near-black index
like 1) and splits files into train/valid/test folders with hashed names. This script
flattens all of that. To be robust to however Roboflow encoded the mask, it treats the
MOST COMMON pixel value as background and everything else as card. It also reports how
much of each mask is card, so you can immediately spot empty (all-black) masks.

Usage:
    python prepare_masks.py --export path/to/UNZIPPED_roboflow_export --out data
"""

import argparse
import glob
import os
import re
import shutil

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg")


def mask_key(stem):
    return stem[:-5] if stem.endswith("_mask") else stem


def tray_of(stem):
    m = re.match(r"([a-zA-Z]+\d+)", stem)
    return m.group(1) if m else stem.split("_")[0]


def run(export_dir, out_dir):
    images, masks = {}, {}
    for path in glob.glob(os.path.join(export_dir, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(path)[1].lower()
        stem = os.path.splitext(os.path.basename(path))[0]
        if ext in IMG_EXT:
            images[stem] = path
        elif ext == ".png":
            masks[mask_key(stem)] = path

    img_dir = os.path.join(out_dir, "images")
    msk_dir = os.path.join(out_dir, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(msk_dir, exist_ok=True)

    paired, missing, coverages, empty = 0, [], [], []
    for key, img_path in sorted(images.items()):
        mpath = masks.get(key)
        if mpath is None:
            missing.append(os.path.basename(img_path))
            continue
        g = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if g is None:
            missing.append(os.path.basename(img_path))
            continue

        # Background = the most common pixel value. Everything else = card.
        background = np.bincount(g.reshape(-1)).argmax()
        fg = (g != background).astype(np.uint8) * 255

        cov = float((fg > 0).mean()) * 100.0
        coverages.append(cov)
        if cov < 1.0:
            empty.append(key)

        shutil.copy(img_path, os.path.join(img_dir, key + ".jpg"))
        cv2.imwrite(os.path.join(msk_dir, key + ".png"), fg)
        paired += 1

    print(f"\nPaired {paired} image/mask pairs -> {out_dir}/")
    if coverages:
        print(f"Card coverage per mask: avg {np.mean(coverages):.1f}%  "
              f"(min {np.min(coverages):.1f}%, max {np.max(coverages):.1f}%)")
        print("  -> healthy masks are usually ~20-45% card. ~0% means the mask is empty.")
    if empty:
        print(f"WARNING: {len(empty)} mask(s) are essentially empty (no card found): {empty}")
    if missing:
        print(f"WARNING: {len(missing)} image(s) had no matching mask: {missing[:20]}")

    trays = sorted({tray_of(k) for k in images if masks.get(k)})
    print(f"Distinct trays: {len(trays)} -> {trays}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True, help="path to the UNZIPPED Roboflow export folder")
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    run(args.export, args.out)