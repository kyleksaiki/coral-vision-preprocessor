"""Generate the CLAHE card crops you will label coral on.

WHY: the Java app runs the coral model per-card on CLAHE crops (CoralFinderOnnx.java),
so the coral model must be TRAINED on the same thing. This script reproduces the Java
pipeline exactly so the crops you label match what the model sees at inference:

  1. CLAHE (Lab L, clip 3.0, tile 8x8) + gray-world white balance on the full RAW photo
     -- matches TrayLightingNormalizer.java (no Gaussian pre-blur).
  2. Run the CARD model on the RAW photo to get the card mask -- matches CardFinderOnnx.java.
  3. For each card blob, crop the CLAHE-full image at its padded bounding box
     -- matches TrayCleaner.buildCoralMask (CROP_PAD_FRAC=0.03, CARD_AREA_FRAC_MIN=0.001).

Each crop is saved as <rawstem>_cardNN.png, keeping the tray id at the front so the
training split can group by tray. Upload these crops to Roboflow and label coral.

Usage:
  python make_card_crops.py --card_onnx ../coral_preprocessor/card_seg.onnx \
      --input path/to/raw_photos/ --out crops/

(point --card_onnx at wherever your trained CARD model lives.)
"""

import argparse
import glob
import os

import cv2
import numpy as np
import onnxruntime as ort

# --- must match CardFinderOnnx.java ---
CARD_IN_W, CARD_IN_H = 1024, 768
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)
# --- must match TrayCleaner.java ---
CROP_PAD_FRAC = 0.03
CARD_AREA_FRAC_MIN = 0.001


def clahe_white_balance(bgr):
    """CLAHE on the Lab L channel, then gray-world white balance. Matches
    TrayLightingNormalizer.claheLabThenWhiteBalance (no pre-blur)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    f = out.astype(np.float32)
    means = f.reshape(-1, 3).mean(axis=0)          # B, G, R means
    m = float(means.mean())
    for c in range(3):
        if means[c] > 0:
            f[..., c] *= (m / means[c])
    return np.clip(f, 0, 255).astype(np.uint8)


def card_mask(session, in_name, bgr):
    """Run the card model on a RAW BGR image; return a full-res 0/255 mask. Matches
    CardFinderOnnx.findCards (resize 1024x768, BGR->RGB, ImageNet norm, logit>0)."""
    h0, w0 = bgr.shape[:2]
    resized = cv2.resize(bgr, (CARD_IN_W, CARD_IN_H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    chw = np.transpose(rgb, (2, 0, 1))[None].astype(np.float32)   # 1,3,H,W
    logits = session.run(None, {in_name: chw})[0]                 # 1,1,H,W
    small = (logits[0, 0] > 0).astype(np.uint8) * 255
    return cv2.resize(small, (w0, h0), interpolation=cv2.INTER_NEAREST)


def gather_inputs(path):
    if os.path.isdir(path):
        files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
            files += glob.glob(os.path.join(path, ext))
            files += glob.glob(os.path.join(path, ext.upper()))
        return sorted(set(files))
    return [path]


def run(card_onnx, input_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    sess = ort.InferenceSession(card_onnx, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    total = 0
    for fp in gather_inputs(input_path):
        bgr = cv2.imread(fp, cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skip (unreadable): {fp}")
            continue
        H, W = bgr.shape[:2]
        clahe = clahe_white_balance(bgr)
        mask = card_mask(sess, in_name, bgr)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            (mask > 127).astype(np.uint8), connectivity=8)
        min_area = CARD_AREA_FRAC_MIN * H * W
        stem = os.path.splitext(os.path.basename(fp))[0]

        idx = 0
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            pad = int(round(CROP_PAD_FRAC * max(w, h)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(W, x + w + pad)
            y1 = min(H, y + h + pad)
            crop = clahe[y0:y1, x0:x1]
            idx += 1
            cv2.imwrite(os.path.join(out_dir, f"{stem}_card{idx:02d}.png"), crop)
            total += 1
        print(f"{stem}: {idx} card crop(s)")

    print(f"\nDone. {total} crop(s) -> {out_dir}")
    print("Next: upload these to Roboflow, label coral (semantic mask, 255 = coral), export.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--card_onnx", required=True, help="path to your trained card_seg.onnx")
    ap.add_argument("--input", required=True, help="a raw photo or a folder of raw photos")
    ap.add_argument("--out", default="crops")
    args = ap.parse_args()
    run(args.card_onnx, args.input, args.out)
