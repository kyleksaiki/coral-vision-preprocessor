# coral-vision-preprocessor

Small Java + OpenCV project for preprocessing coral card images.

## What it does

Given a single cropped card image, the pipeline labels pixels as:

- `C` / `c` = coral
- `A` / `a` = algae
- `S` / `s` = silt
- `H` / `h` = shadow
- `.` = unlabeled

It then creates a cleaned output image by removing unwanted material and replacing it with an estimated card background color.

## Notes


- Main color logic is based in **Lab** color space.

## Files

- `Main.java` – batch runner and command-line entry point
- `TrayCleaner.java` – main processing pipeline
- `TrayLightingNormalizer.java` - CLAHE and white-balancing preprocessing
- `LabelCoral.java` – coral labeling
- `LabelAlgae.java` – algae labeling
- `LabelSilt.java` – silt labeling
- `LabelShadow.java` – shadow labeling
- `BackgroundColorEstimator.java` – background color estimation

## Run

Download openCV Java `.jar`
https://opencv.org/releases

Insert file path to opencv.jar in .vscode/settings.json

Update filepaths in run.bat to your local instance (optional)

Pass an output folder first, then one or more cropped card images:

```bash
java -Djava.library.path=/path/to/opencv/native -cp .:opencv-4xx.jar Main <output-dir> <card1> <card2> ...
