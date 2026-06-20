# Coral-Nursery Card Segmentation (PyTorch)

Find the white ID cards in underwater coral-nursery tray photos and output a
**card / not-card segmentation mask**. This is the learned replacement for a hand-tuned
classical OpenCV finder (`CardFinder.java`), kept as a baseline to compare against.

**Problem → Data → Method → Metrics → Failure analysis → Next steps** is the structure below.

---

## Problem

Each ~12 MP photo shows a flexible black mesh grid holding ~20 white plastic ID cards in a
rough 4×5 layout. The grid warps (cards tilt), cards touch, coral colonies occlude and bite
into the white, and the scene has an underwater blue-green color cast plus distractors (PVC
pipe, paper labels, an orange ruler, background sand). The classical pipeline thresholds on
brightness, but "bright" means three different things — card, mesh, and sand — so it needs
per-image retuning and breaks when cards connect to the border through thin mesh strands.
A learned model handles this variation far more robustly.

## Data

- Input: **raw tray photos** (`.jpg`). Train on the originals, *not* the CLAHE/white-balanced
  output — the model and augmentation handle lighting, and pre-processing only adds a failure
  point. (If your Java pipeline's `2_clahe_white_balance.png` ever comes out all-black, that's
  the bug this sidesteps.)
- Labels: one **binary PNG mask per photo** (255 = card, 0 = everything else), same filename
  stem as the image.
- **Split by dive, not by random frame.** Name files `<diveId>_<anything>.jpg` (e.g.
  `dive03_frame12.jpg`). Frames from one dive are near-duplicates; grouping them into the same
  split prevents the model from "memorizing" a dive and faking a high score.

```
data/
  images/   dive01_001.jpg, dive01_002.jpg, dive02_001.jpg, ...
  masks/    dive01_001.png, dive01_002.png, dive02_001.png, ...   (255 = card)
```

## Method

- **U-Net** (`segmentation-models-pytorch`) with a **ResNet-34 encoder pretrained on ImageNet**.
  Standard, reliable for binary segmentation with limited data; the pretrained encoder means
  you need far fewer labels. Small and fast — no prompt at inference (unlike SAM).
- **Image size:** downscale to **1024×768** (don't tile). Cards are large relative to the frame,
  so full 12 MP resolution isn't needed, and downscaling keeps the project simple to run.
- **Augmentation** (`albumentations`): flips, 90° rotations, mild affine (for the warped grid),
  brightness/contrast and hue/saturation jitter (for the underwater color cast).
- **Loss:** BCE + Dice. **Metrics:** IoU and Dice (never pixel accuracy).

## Labeling (do this first — the model is only as good as these masks)

You need ~30–80 labeled trays to start. Recommended path for a first project:

1. Sign up for **Roboflow** (free tier) → create an *Instance/Semantic Segmentation* project.
2. Upload your raw photos. Use the **Smart Polygon / SAM-assisted** tool: click a card and it
   proposes a mask (this is the "SAM pre-labeling / weak supervision" step — a big general model
   does ~80% of the work, you fix the last 20%). Label every card; fill across coral so the whole
   card footprint is one region.
3. Export as **semantic masks (PNG)**. Put the PNGs in `data/masks/` and the photos in
   `data/images/`, matching stems, with the `<diveId>_...` naming above.

CVAT is the free, local, open-source alternative if you'd rather not use a hosted tool; it also
has a SAM-assisted mode. (A fully local SAM script is possible but heavier to set up — start
hosted, automate later.)

## Setup

```bash
# 1) Install PyTorch for YOUR machine (skip on Colab — it's preinstalled):
#    https://pytorch.org/get-started/locally/
pip install -r requirements.txt
```

You can train on **Google Colab's free GPU** or any single consumer GPU. CPU-only works but is
slow. A run on this data size is minutes-per-epoch on a GPU.

## Run, in order

```bash
python train.py        --config config.yaml                 # trains, saves checkpoints/best.pt
python evaluate.py     --ckpt checkpoints/best.pt            # test IoU/Dice + overlay images
python infer.py        --ckpt checkpoints/best.pt --input photo.jpg --out out/   # new photos
python export_onnx.py  --ckpt checkpoints/best.pt --out card_seg.onnx            # portable model
```

Compare against the classical baseline (point it at a folder of the Java pipeline's
`9_card_mask.png` files, renamed to match each test image's stem):

```bash
python evaluate.py --ckpt checkpoints/best.pt --baseline_dir path/to/java_masks/
```

This prints `Learned model IoU` vs `Classical baseline IoU` on the **same held-out test set** —
that one line is the core of the resume story.

## How it replaces CardFinder.java

`infer.py` writes `<name>_mask.png` — a full-resolution binary card mask, the exact same role as
the Java `9_card_mask.png`. So you can either (a) use the Python overlay directly, or (b) feed the
predicted mask back into your existing Java pipeline in place of `CardFinder.findCards`. The
`_overlay.png` it also writes mirrors `CardFinder.renderCardOverlay` so the two are visually
comparable.

## Metrics (fill in after training)

| Method | Test IoU | Test Dice | Latency |
|---|---|---|---|
| Classical baseline (Java) | … | … | … |
| Learned U-Net (this repo) | … | … | … |

Report on the **held-out test dives only**, plus a few qualitative overlays and honest failures.

## Failure analysis (be honest — interviewers reward this)

- **Warped grid / tilt:** affine augmentation helps but extreme tilt can still confuse edges.
- **Coral biting card edges:** the model may cut a card short where coral eats the white.
- **Color cast across dives:** if a new dive's lighting is far from training, IoU drops — the
  dive-based split is what surfaces this honestly instead of hiding it.
- **Distractors** (pipe, sand, labels) are the main source of false positives.

## What most improves results

**More and cleaner labels — full stop.** Not the architecture, loss, or learning rate. If IoU is
weak, label 20 more trays (especially from dives that fail) before tuning anything else. Second
lever: make sure your masks fill *across* coral so the model learns the full card footprint.

## Next steps if it's not good enough

1. Label more trays, prioritizing failure dives.
2. Try `smp.FPN` or `smp.DeepLabV3Plus` (same arguments as `smp.Unet`).
3. Add Tversky/Focal loss if cards are under-segmented.
4. Move to **instance** segmentation (per-card IDs) with Mask R-CNN or SAM-2-based pipelines.
5. Add tiling at full resolution if small/thin cards are being lost at 1024×768.

## Resume / interview talking points

- Framed an ambiguous real-world task (warped, occluded, color-cast underwater imagery) as
  binary segmentation with a clear deliverable and metric.
- Built a **classical baseline and a learned model** and measured both with IoU/Dice on a
  **leakage-free, dive-grouped** test split.
- Used **SAM-assisted weak supervision** to label efficiently, then trained a compact, prompt-free
  U-Net that's deployable (ONNX export, latency noted).
- Did honest failure analysis and identified the highest-leverage improvement (data, not model).
```
