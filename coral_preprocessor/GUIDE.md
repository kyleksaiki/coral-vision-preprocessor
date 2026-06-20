# Coral Vision — Project Guide (Java pipeline + PyTorch card model)

This guide explains what you have, how the two halves connect, and exactly what to do next.
Read it top to bottom once; then use it as a checklist.

---

## 1. The big picture

You have **two halves** that meet at one file:

```
  PYTHON (offline, occasional)                 JAVA (your GUI app, runs all the time)
  ┌─────────────────────────────┐              ┌──────────────────────────────────────┐
  │ label photos  ->  train      │   exports    │ CardFinderOnnx loads card_seg.onnx    │
  │ a U-Net  ->  card_seg.onnx   │ ───────────► │ and returns a card mask, exactly like │
  │ (coral-card-segmentation/)   │  one file    │ the old CardFinder did                │
  └─────────────────────────────┘              └──────────────────────────────────────┘
```

- **Python** is only used to *train* the model and *export* one file, `card_seg.onnx`.
  You run it yourself, occasionally. End users never see Python.
- **Java** is your real product. It loads that one `.onnx` file and runs the model in-process
  with ONNX Runtime. No Python installed on any user's machine. This is the "seamless" path.

The bridge is **ONNX**, a portable model format. Train in PyTorch, run anywhere — including Java.

---

## 2. What's in the two zips

### `coral-vision-java/` (your Java project files)
- **Modified:** `CardFinder.java` (added `renderCardOverlay`), `TrayCleaner.java`, `Main.java`
  (now also saves `9_card_mask.png`, the raw mask).
- **New:** `CardFinderOnnx.java` — runs the trained model inside Java. Drop-in for the old finder.
- **Unchanged (included so the project is complete):** `CoralMaskRefiner`, `LabelCoral`,
  `LabelAlgae`, `LabelSilt`, `HoughLines`, `TrayLightingNormalizer`.
- Copy these into your existing `coral-vision-preprocessor` folder, overwriting the three
  modified ones. Keep your `run.bat` and `.vscode/settings.json` as they are.

### `coral-card-segmentation/` (the Python model project)
Everything to label, train, evaluate, and export the model. See its own README, plus section 5.

---

## 3. Your data: ~20 trays, 24 photos

20 distinct trays is good — almost no near-duplicates, so a normal split is fine. Rule: **a tray
must live in only one split.** Name files so the tray id comes first, e.g. `tray03_a.jpg`.

Suggested split (group whole trays):

| Split | Trays | ~Images | Purpose |
|---|---|---|---|
| train | ~14 | ~17 | the model learns from these |
| val   | ~3  | ~3–4 | picks the best epoch / early stop |
| test  | ~3  | ~3–4 | the honest final score (never touched during training) |

The config does this automatically from the `tray##_` prefix. With a test set this small the IoU
is *indicative, not definitive* — report it honestly. The biggest future improvement is simply
**more distinct trays**.

---

## 4. Step-by-step: from photos to a working model file

### Step A — Label (this is the real work; the model is only as good as these masks)
1. Sign up for **Roboflow** (free) → new **Semantic Segmentation** project.
2. Upload your 24 photos. Use the **Smart Polygon / SAM-assisted** tool: click a card, it
   proposes a mask, you fix the edges. Label every card; **fill the mask across the coral** so
   each card is one solid region (the whole card footprint, coral included).
3. Export as **semantic masks (PNG)**, 255 = card.
4. Put photos in `coral-card-segmentation/data/images/` and masks in `.../data/masks/`, with
   matching names and the `tray##_` prefix.

### Step B — Install & train
```bash
cd coral-card-segmentation
# Install PyTorch for your machine first (skip on Google Colab — it's preinstalled):
#   https://pytorch.org/get-started/locally/
pip install -r requirements.txt
python train.py --config config.yaml        # saves checkpoints/best.pt
```
Use Google Colab's free GPU if you don't have one. Training this dataset is minutes per epoch.

### Step C — Evaluate (and compare to your classical baseline)
```bash
python evaluate.py --ckpt checkpoints/best.pt
# Optional, the resume money-shot — compare learned vs classical on the SAME test trays:
python evaluate.py --ckpt checkpoints/best.pt --baseline_dir path/to/java_9_card_mask_pngs/
```
To make the baseline folder: run your Java pipeline on the **test** photos and collect their
`9_card_mask.png` files, renamed to match each test image's stem.

### Step D — Export the one file Java needs
```bash
python export_onnx.py --ckpt checkpoints/best.pt --out card_seg.onnx
```
Copy `card_seg.onnx` somewhere your Java app can read it.

---

## 5. Connecting it to Java (the important part)

### 5.1 Add ONNX Runtime to your classpath (like you did OpenCV)
Get `com.microsoft.onnxruntime:onnxruntime:1.26.0` (Maven), or download that jar from Maven
Central and add it to your classpath next to the OpenCV jar. The jar bundles native libraries
for Windows/macOS/Linux and unpacks them at runtime, so usually the jar on the classpath is all
you need. In `run.bat`, add the jar to your `-cp` list (use `;` between entries on Windows).

### 5.2 Use the model
`CardFinderOnnx.findCards(Mat bgr)` returns the **same** `CV_8UC1` mask (255 = card) the old
`CardFinder.findCards` returned — so it's a literal drop-in. Create it **once** at startup
(loading the model is the slow part) and reuse it:

```java
// once, when your GUI starts:
CardFinderOnnx finder = new CardFinderOnnx("card_seg.onnx");

// per photo (feed the RAW image — the same kind the model trained on, NOT the CLAHE output):
Mat cardMask = finder.findCards(rawBgr);
Mat overlay  = CardFinder.renderCardOverlay(rawBgr, cardMask);

// when the app closes:
finder.close();
```

### 5.3 Wiring it into TrayCleaner (optional, when you're ready)
Right now `TrayCleaner` uses the **classical** finder, so the project still builds without the
ONNX jar or a model. When you want the learned path, the cleanest move is to let the GUI choose.
Add an overload that accepts a ready-made mask:

```java
// in TrayCleaner, replace the two card-finding lines with a mask you pass in:
//   Mat cardMask = CardFinder.findCards(clahe, medianBlur);   // OLD classical
//   Mat cardMask = finder.findCards(rawBgr);                  // NEW learned (built once, passed in)
Mat segmentCards = CardFinder.renderCardOverlay(rawBgr, cardMask);
```

Keep both paths and add a toggle in the GUI — showing classical vs. learned side by side is a
strong demo and is literally your baseline-vs-model comparison made interactive.

> **Important:** train on and feed the **raw** photos, not `2_clahe_white_balance.png`. Your CLAHE
> output came out all-black (a bug in that step) — training on raw photos avoids it, and the model
> learns its own normalization anyway. You can still use CLAHE for the coral/algae labeling stages.

---

## 6. End-to-end checklist

- [ ] Label 24 photos in Roboflow, export PNG masks, name files `tray##_...`
- [ ] `pip install -r requirements.txt` (PyTorch separately)
- [ ] `python train.py` → `python evaluate.py` (+ `--baseline_dir` for the comparison)
- [ ] `python export_onnx.py` → `card_seg.onnx`
- [ ] Add the ONNX Runtime jar to the Java classpath
- [ ] `new CardFinderOnnx("card_seg.onnx")` once; `finder.findCards(rawBgr)` per photo
- [ ] Add a GUI toggle: classical finder vs. learned finder

---

## 7. Improvements, by payoff

1. **More distinct trays** — by far the biggest lever for generalization.
2. **Mask quality** — fill across coral so the model learns the full card footprint.
3. **Test-time augmentation** — average predictions over horizontal/vertical flips for a free bump.
4. **Per-card instance segmentation** — so the GUI can track each card (e.g. "DHEL-12") and its
   coral over time. Natural next milestone for a coral-vision product; needs instance labels and a
   model like Mask R-CNN or a SAM-2-based pipeline.
5. **Try `smp.FPN` or `smp.DeepLabV3Plus`** (same arguments as `smp.Unet`) if U-Net plateaus.

## 8. Honest expectations

24 photos with a pretrained encoder and strong augmentation gives a solid **proof-of-concept**,
not a production-robust model — and that's fine for a resume project. Cards are large and
high-contrast, so binary card/not-card is the *easy* target and should work well; the coral
detail is the hard part and is out of scope for this model. Your strongest story is the
**methodology**: classical baseline vs. learned model, leakage-free split by tray, IoU/Dice on a
held-out set, deployed into a real Java GUI via ONNX. That reads as senior work regardless of the
final number.
