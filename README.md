# coral-vision-preprocessor

Small Java + OpenCV project for preprocessing coral card images.

## What it does

Given a tray of cards, the pipeline labels pixels as:

- `C` / `c` = coral
- `A` / `a` = algae
- `S` / `s` = silt
- `H` / `h` = shadow
- `.` = unlabeled

It then creates a cleaned output image by removing unwanted pixels.

## Files

- `Main.java` – batch runner and command-line entry point
- `TrayCleaner.java` – main processing pipeline
- `TrayLightingNormalizer.java` - CLAHE and white-balancing preprocessing
- `LabelCoral.java` – coral labeling
- `LabelAlgae.java` – algae labeling
- `LabelSilt.java` – silt labeling

## Run

Download openCV Java `.jar`
https://opencv.org/releases

Insert file path to opencv.jar in .vscode/settings.json

Update filepaths in run.bat to your local instance 

type .\run.bat in terminal to compile and run program

```bash

