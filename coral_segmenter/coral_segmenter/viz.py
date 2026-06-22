"""Turns a binary coral mask into a human-friendly overlay: each separate coral blob
gets its own translucent color, a white outline, and an index number. Mirrors the look
of the Java coral overlay so the two are visually comparable."""

import cv2
import numpy as np


def _color_for(idx):
    """Distinct RGB color per blob, spread around the hue wheel by the golden angle."""
    hue = ((idx * 0.61803398875) % 1.0) * 180.0  # OpenCV hue is 0..179
    hsv = np.uint8([[[int(hue), 217, 230]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def overlay_instances(image_rgb, mask, alpha=0.45, dim_non_coral=0.75):
    """image_rgb: HxWx3 uint8. mask: HxW, nonzero = coral. Returns HxWx3 uint8 RGB."""
    h, w = mask.shape[:2]
    binary = (mask > 127).astype(np.uint8)

    out = image_rgb.astype(np.float32)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    out[labels == 0] *= dim_non_coral  # dim non-coral so coral pops

    for i in range(1, n):
        color = np.array(_color_for(i), dtype=np.float32)
        sel = labels == i
        out[sel] = alpha * color + (1.0 - alpha) * out[sel]

    out = np.clip(out, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thickness = max(2, round(3 * (max(h, w) / 1500.0)))
    cv2.drawContours(bgr, contours, -1, (255, 255, 255), thickness)

    font_scale = max(0.8, 1.6 * (max(h, w) / 1500.0))
    for i in range(1, n):
        cx, cy = centroids[i]
        cv2.putText(bgr, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(bgr, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
