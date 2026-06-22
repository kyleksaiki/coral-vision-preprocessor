import ai.onnxruntime.OrtException;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.*;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Batch runner. Takes an output directory and a list of tray images, and writes 6 numbered
 * PNGs per tray into {@code <output>/trays/tray_NN_<name>/}.
 *
 * <p>Two-model pipeline:</p>
 * <ol>
 *   <li>The RAW image is fed to the CARD model ({@link CardFinderOnnx}) to get a card mask.</li>
 *   <li>The RAW image is normalized (CLAHE + white balance) inside {@link TrayCleaner}.</li>
 *   <li>For each card the card model found, the CORAL model ({@link CoralFinderOnnx}) runs on a
 *       crop of the CLAHE image; the results are stitched and gated to the card mask.</li>
 *   <li>The coral mask is grown into similar-colored edge pixels ({@link CoralMaskGrower}).</li>
 * </ol>
 *
 * <p>Both models are loaded ONCE here and reused for every image. Paths are configurable:
 * {@code -Dcard.model=...} and {@code -Dcoral.model=...} (run.bat sets both).</p>
 */
public class Main {

    /** Load the OpenCV native library once when the class loads. */
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /** Card-segmentation model path. Override with {@code -Dcard.model=...}. Forward slashes are fine on Windows. */
    private static final String CARD_MODEL_PATH = System.getProperty(
            "card.model",
            "C:/Users/kylek/OneDrive/Desktop/Personal_Projects/coral-vision-preprocessor/card_segmenter/card_seg.onnx");

    /** Coral-segmentation model path. Override with {@code -Dcoral.model=...}. */
    private static final String CORAL_MODEL_PATH = System.getProperty(
            "coral.model",
            "C:/Users/kylek/OneDrive/Desktop/Personal_Projects/coral-vision-preprocessor/coral_segmenter/coral_seg.onnx");

    /**
     * Program entry point.
     *
     * @param args {@code args[0]} = output directory; {@code args[1...]} = input image paths
     * @throws IOException if directories/images can't be read or written
     */
    public static void main(String[] args) throws IOException {
        if (args.length < 2) {
            System.out.println(
                    "Usage: java -Djava.library.path=/path/to/opencv/native "
                            + "-cp .:opencv-4xx.jar:onnxruntime-1.26.0.jar "
                            + "-Dcard.model=/path/card_seg.onnx -Dcoral.model=/path/coral_seg.onnx "
                            + "Main <output-dir> <image1> <image2> ...");
            System.out.println("On Windows, replace ':' with ';' in the classpath.");
            return;
        }

        Path outputDir = Paths.get(args[0]);
        Path traysDir = outputDir.resolve("trays");
        Files.createDirectories(outputDir);
        Files.createDirectories(traysDir);
        clearDirectoryContents(traysDir);

        // Both model files must be present and readable before we start.
        if (!Files.exists(Paths.get(CARD_MODEL_PATH))) {
            System.err.println("Card model not found at: " + CARD_MODEL_PATH);
            System.err.println("Pass -Dcard.model=/full/path/card_seg.onnx, or fix the default in Main.");
            return;
        }
        if (!Files.exists(Paths.get(CORAL_MODEL_PATH))) {
            System.err.println("Coral model not found at: " + CORAL_MODEL_PATH);
            System.err.println("Pass -Dcoral.model=/full/path/coral_seg.onnx, or fix the default in Main.");
            return;
        }

        // Load BOTH models once and reuse them for every image (loading is the slow part).
        CardFinderOnnx cardFinder;
        CoralFinderOnnx coralFinder;
        try {
            cardFinder = new CardFinderOnnx(CARD_MODEL_PATH);
            coralFinder = new CoralFinderOnnx(CORAL_MODEL_PATH);
        } catch (OrtException e) {
            System.err.println("Could not load a model: " + e.getMessage());
            System.err.println("Fixes:");
            System.err.println("  - Make sure the ONNX Runtime jar (onnxruntime-1.26.0.jar) is on your classpath.");
            System.err.println("  - If a model is in OneDrive and shows a cloud icon, right-click it ->");
            System.err.println("    \"Always keep on this device\" so Java can actually read it.");
            return;
        } catch (UnsatisfiedLinkError e) {
            System.err.println("ONNX Runtime native libraries failed to load: " + e.getMessage());
            System.err.println("Make sure the ONNX Runtime jar is on your classpath; it unpacks its natives at runtime.");
            return;
        }

        try {
            for (int i = 1; i < args.length; i++) {
                Path inputImage = Paths.get(args[i]);
                validateSingleImageInput(inputImage);
                processTrayImage(cardFinder, coralFinder, inputImage, traysDir, i);
            }
        } finally {
            closeQuietly(cardFinder);
            closeQuietly(coralFinder);
        }

        System.out.println("Finished. Results written to: " + outputDir.toAbsolutePath());
    }

    /**
     * Processes one tray image and writes its 6 outputs. The card model runs on the RAW image;
     * the coral model runs per-card on CLAHE crops inside {@link TrayCleaner}. A single image that
     * fails inference is skipped rather than killing the whole batch.
     */
    private static void processTrayImage(CardFinderOnnx cardFinder, CoralFinderOnnx coralFinder,
                                         Path imagePath, Path traysDir, int index) throws IOException {
        Mat raw = Imgcodecs.imread(imagePath.toString(), Imgcodecs.IMREAD_COLOR);
        if (raw.empty()) {
            throw new IOException("Could not read image: " + imagePath);
        }

        TrayCleaner.TrayResult result;
        try {
            // CARD model on the RAW image.
            Mat cardMask = cardFinder.findCards(raw);
            // Normalize + per-card CORAL model + grow + cleaned, all inside processTray.
            result = TrayCleaner.processTray(raw, cardMask, coralFinder);
        } catch (OrtException e) {
            System.err.printf(Locale.US, "Skipping %s -- model inference failed: %s%n",
                    imagePath.getFileName(), e.getMessage());
            return;
        }

        String baseName = stripExtension(imagePath.getFileName().toString());
        Path trayDir = traysDir.resolve(baseName);
        Files.createDirectories(trayDir);

        // 1 raw -> 2 clahe -> 3 card -> 4 coral -> 5 coral grown -> 6 cleaned
        Imgcodecs.imwrite(trayDir.resolve("1_raw.png").toString(), result.rawInput);
        Imgcodecs.imwrite(trayDir.resolve("2_clahe_white_balance.png").toString(), result.claheWhiteBalanced);
        Imgcodecs.imwrite(trayDir.resolve("3_card.png").toString(), result.card);
        Imgcodecs.imwrite(trayDir.resolve("4_coral.png").toString(), result.coral);
        Imgcodecs.imwrite(trayDir.resolve("5_coral_grown.png").toString(), result.coralGrown);
        Imgcodecs.imwrite(trayDir.resolve("6_cleaned.png").toString(), result.cleaned);

        System.out.printf(Locale.US, "Processed %s%n", imagePath.getFileName());
    }

    private static void closeQuietly(AutoCloseable c) {
        try {
            c.close();
        } catch (Exception e) {
            // Nothing useful to do on shutdown.
        }
    }

    private static void validateSingleImageInput(Path input) {
        if (!Files.exists(input)) {
            throw new IllegalArgumentException("Input image does not exist: " + input);
        }
        if (!Files.isRegularFile(input)) {
            throw new IllegalArgumentException("Input must be a file, not a directory: " + input);
        }
        if (!isImageFile(input)) {
            throw new IllegalArgumentException("Unsupported image format: " + input.getFileName());
        }
    }

    private static boolean isImageFile(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.US);
        return name.endsWith(".jpg")
                || name.endsWith(".jpeg")
                || name.endsWith(".png")
                || name.endsWith(".tif")
                || name.endsWith(".tiff");
    }

    private static String stripExtension(String filename) {
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    private static void clearDirectoryContents(Path dir) throws IOException {
        if (!Files.exists(dir)) {
            return;
        }
        try (Stream<Path> stream = Files.list(dir)) {
            for (Path path : stream.collect(Collectors.toList())) {
                deleteRecursively(path);
            }
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        if (Files.isDirectory(path)) {
            try (Stream<Path> stream = Files.list(path)) {
                for (Path child : stream.collect(Collectors.toList())) {
                    deleteRecursively(child);
                }
            }
        }
        Files.deleteIfExists(path);
    }
}