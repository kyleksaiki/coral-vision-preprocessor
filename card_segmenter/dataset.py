"""Dataset, augmentation, and the tray-grouped train/val/test split.

The split is the part interviewers care about most: photos from the same tray are
near-duplicates, so a random split would let the model 'memorize' a tray and score
high without really generalizing. We split by tray instead."""

import glob
import os
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dive_id(path):
    """Group photos by physical tray. Handles both 'tray01a.jpg' and 'tray01_a.jpg'
    (and Roboflow's hashed 'tray01a_JPG.rf.xxxx.jpg'). Returns e.g. 'tray01'.
    Falls back to the text before the first underscore if the pattern doesn't match."""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.match(r"([a-zA-Z]+\d+)", stem)   # letters + digits at the start: 'tray01'
    return m.group(1) if m else stem.split("_")[0]


def discover_items(images_dir, masks_dir, image_glob):
    """Find (image, mask) pairs. A mask must be <same stem>.png in masks_dir."""
    items = []
    for img in sorted(glob.glob(os.path.join(images_dir, image_glob))):
        stem = os.path.splitext(os.path.basename(img))[0]
        mask = os.path.join(masks_dir, stem + ".png")
        if os.path.exists(mask):
            items.append((img, mask))
    return items


def make_splits(cfg):
    """Return (train_items, val_items, test_items, info_dict), grouped by tray."""
    d = cfg["data"]
    items = discover_items(d["images_dir"], d["masks_dir"], d["image_glob"])
    if not items:
        raise RuntimeError(
            "No (image, mask) pairs found. Put raw photos in "
            f"'{d['images_dir']}' and matching <name>.png masks in '{d['masks_dir']}'."
        )

    by_dive = {}
    for it in items:
        by_dive.setdefault(dive_id(it[0]), []).append(it)

    dives = sorted(by_dive)
    random.Random(cfg["train"]["seed"]).shuffle(dives)

    val_dives = set(d.get("val_dives") or [])
    test_dives = set(d.get("test_dives") or [])
    if not val_dives and not test_dives:
        n = len(dives)
        n_test = max(1, round(n * d["test_frac"]))
        n_val = max(1, round(n * d["val_frac"]))
        test_dives = set(dives[:n_test])
        val_dives = set(dives[n_test:n_test + n_val])

    train, val, test = [], [], []
    for dv in dives:
        bucket = test if dv in test_dives else (val if dv in val_dives else train)
        bucket.extend(by_dive[dv])

    info = {
        "n_trays": len(dives),
        "train_trays": sorted(d for d in dives if d not in val_dives and d not in test_dives),
        "val_trays": sorted(val_dives),
        "test_trays": sorted(test_dives),
    }
    return train, val, test, info


def train_tf(h, w):
    """Augmentation tuned for underwater variation: flips for the warped grid,
    brightness/hue jitter for the blue-green color cast and uneven lighting.

    NOTE: no RandomRotate90 here. A 90-degree turn swaps width/height, and since our
    images aren't square that produces mixed sizes in a batch and crashes the loader.
    HorizontalFlip + VerticalFlip + the +/-15 degree Affine give plenty of variety."""
    return A.Compose([
        A.Resize(h, w),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(0.0, 0.05), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_tf(h, w):
    """No randomness for validation/test/inference -- just resize + normalize."""
    return A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _to_landscape(img, mask):
    """Force every photo to landscape before resizing so portrait shots aren't
    stretched oddly and everything is consistent."""
    if img.shape[0] > img.shape[1]:  # height > width => portrait
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    return img, mask


class CardDataset(Dataset):
    def __init__(self, items, tf):
        self.items = items
        self.tf = tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img_path, mask_path = self.items[i]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mask_path}")

        img, mask = _to_landscape(img, mask)

        aug = self.tf(image=np.ascontiguousarray(img), mask=mask)
        image = aug["image"]
        m = aug["mask"]
        if not torch.is_tensor(m):
            m = torch.from_numpy(m)
        m = (m > 127).float().unsqueeze(0)  # (1, H, W), values 0.0 / 1.0
        return image, m