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
 * Batch runner. Takes an output directory and a list of tray images, and writes 7 numbered
 * PNGs per tray into {@code <output>/trays/tray_NN_<name>/}.
 *
 * <p>Forked architecture (see {@link TrayCleaner}):</p>
 * <ol>
 *   <li>The RAW image is fed to the ONNX card model ({@link CardFinderOnnx}) to get a card mask.</li>
 *   <li>The RAW image is also normalized (CLAHE + white balance) inside {@link TrayCleaner}.</li>
 *   <li>The classical coral/algae/silt labelers run on the normalized image, gated to the mask.</li>
 * </ol>
 *
 * <p>The model file ({@code card_seg.onnx}) is loaded ONCE here and reused for every image,
 * because loading it is the slow part. By default it is read from {@code card_seg.onnx} in the
 * current working directory; override with {@code -Dcard.model=/full/path/card_seg.onnx}.</p>
 */
public class Main {

    /**
     * Load the OpenCV native library one time when the class gets loaded.
     * If this is missing, OpenCV image I/O and processing calls will fail.
     */
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    /**
     * Path to the exported card-segmentation model. Configurable, so the file can move without
     * editing logic: override on the command line with {@code -Dcard.model=...} (run.bat does
     * this), otherwise it falls back to the default below.
     *
     * <p>Forward slashes are used on purpose: Windows accepts them, and they avoid the
     * backslash-escaping trap in Java string literals.</p>
     */
    private static final String MODEL_PATH = System.getProperty(
            "card.model",
            "C:/Users/kylek/OneDrive/Desktop/Personal_Projects/coral-vision-preprocessor/card_segmenter/card_seg.onnx");

    /**
     * Program entry point.
     *
     * <p>The first command-line argument is the output directory.
     * Every argument after is treated as an input tray image.</p>
     *
     * @param args command-line arguments:
     *             {@code args[0]} is the output directory;
     *             {@code args[1...]} are input image paths
     * @throws IOException if creating directories, reading images, writing images or deleting
     *                     old output files fails
     */
    public static void main(String[] args) throws IOException {
        // Expect:
        // args[0] = output folder
        // args[1...] = one or more input image paths
        if (args.length < 2) {
            System.out.println(
                    "Usage: java -Djava.library.path=/path/to/opencv/native "
                            + "-cp .:opencv-4xx.jar:onnxruntime-1.26.0.jar "
                            + "-Dcard.model=/path/to/card_seg.onnx Main <output-dir> <image1> <image2> ...");
            System.out.println("On Windows, replace ':' with ';' in the classpath.");
            return;
        }

        Path outputDir = Paths.get(args[0]);
        Path traysDir = outputDir.resolve("trays");

        // Make sure the output folders exist before writing anything.
        Files.createDirectories(outputDir);
        Files.createDirectories(traysDir);

        // Clear old tray result folders so each run starts clean.
        clearDirectoryContents(traysDir);

        // Load the card-segmentation model ONCE and reuse it for every image (loading is the
        // slow part). If this fails, it's almost always a wrong path, a OneDrive "online-only"
        // file, or a missing ONNX Runtime jar on the classpath, so say so clearly and stop.
        Path modelFile = Paths.get(MODEL_PATH);
        if (!Files.exists(modelFile)) {
            System.err.println("Card model not found at: " + MODEL_PATH);
            System.err.println("Pass -Dcard.model=/full/path/card_seg.onnx, or fix the default in Main.");
            return;
        }

        CardFinderOnnx finder;
        try {
            finder = new CardFinderOnnx(MODEL_PATH);
        } catch (OrtException e) {
            System.err.println("Could not load the card model from \"" + MODEL_PATH + "\": " + e.getMessage());
            System.err.println("Fixes:");
            System.err.println("  - Make sure the ONNX Runtime jar (onnxruntime-1.26.0.jar) is on your classpath.");
            System.err.println("  - If the file is in OneDrive and shows a cloud icon, right-click it ->");
            System.err.println("    \"Always keep on this device\" so Java can actually read it.");
            return;
        } catch (UnsatisfiedLinkError e) {
            System.err.println("ONNX Runtime native libraries failed to load: " + e.getMessage());
            System.err.println("Make sure the ONNX Runtime jar is on your classpath; it unpacks its natives at runtime.");
            return;
        }

        try {
            // Process every input image passed in after the output directory.
            for (int i = 1; i < args.length; i++) {
                Path inputImage = Paths.get(args[i]);

                // Basic checks before trying to read the image.
                validateSingleImageInput(inputImage);

                // Process the image and save outputs into its own folder.
                processTrayImage(finder, inputImage, traysDir, i);
            }
        } finally {
            // Release the model session when the batch is done.
            try {
                finder.close();
            } catch (OrtException e) {
                // Nothing useful to do on shutdown; ignore.
            }
        }

        System.out.println("Finished. Results written to: " + outputDir.toAbsolutePath());
    }

    /**
     * Processes a single tray image and writes its 7 output images to a dedicated folder.
     *
     * <p>The card mask comes from the model, fed the RAW image. The normalized (CLAHE) image
     * and the gated labeling are produced inside {@link TrayCleaner#processTray(Mat, Mat)}.</p>
     *
     * @param finder    the already-loaded card model (reused across images)
     * @param imagePath path to the input image
     * @param traysDir  output directory for per-tray folders
     * @param index     image id, used to name the per-tray folder
     * @throws IOException if the image cannot be read or the output folder/files cannot be created
     */
    private static void processTrayImage(CardFinderOnnx finder, Path imagePath, Path traysDir, int index)
            throws IOException {
        // Read the input image as a color (BGR) image.
        Mat raw = Imgcodecs.imread(imagePath.toString(), Imgcodecs.IMREAD_COLOR);
        if (raw.empty()) {
            throw new IOException("Could not read image: " + imagePath);
        }

        // STEP 3: card mask from the model, fed the RAW image (NOT the CLAHE image). If a single
        // image fails inference, skip just that image instead of killing the whole batch.
        Mat cardMask;
        try {
            cardMask = finder.findCards(raw);
        } catch (OrtException e) {
            System.err.printf(Locale.US, "Skipping %s -- card model inference failed: %s%n",
                    imagePath.getFileName(), e.getMessage());
            return;
        }

        // STEP 2 (normalize) + gated coral/algae/silt labeling all happen inside processTray.
        TrayCleaner.TrayResult result = TrayCleaner.processTray(raw, cardMask);

        // Make a per-tray folder name using the input order and original file name.
        String baseName = stripExtension(imagePath.getFileName().toString());
        Path trayDir = traysDir.resolve(String.format(Locale.US, "tray_%02d_%s", index, baseName));
        Files.createDirectories(trayDir);

        // The 7 numbered outputs, in order:
        // 1 raw -> 2 clahe -> 3 card (model) -> 4 coral -> 5 algae -> 6 cleaned -> 7 components
        Imgcodecs.imwrite(trayDir.resolve("1_raw.png").toString(), result.rawInput);
        Imgcodecs.imwrite(trayDir.resolve("2_clahe_white_balance.png").toString(), result.claheWhiteBalanced);
        Imgcodecs.imwrite(trayDir.resolve("3_card.png").toString(), result.card);
        Imgcodecs.imwrite(trayDir.resolve("4_coral.png").toString(), result.labelCoral);
        Imgcodecs.imwrite(trayDir.resolve("5_algae.png").toString(), result.labelAlgae);
        Imgcodecs.imwrite(trayDir.resolve("6_cleaned.png").toString(), result.cleaned);
        Imgcodecs.imwrite(trayDir.resolve("7_coral_components.png").toString(), result.coralComponents);

        System.out.printf(Locale.US, "Processed %s -> %s%n", imagePath.getFileName(), trayDir.getFileName());
    }

    /**
     * Validates that an image path exists and has a supported image format.
     *
     * @param input path to validate
     * @throws IllegalArgumentException if the path does not exist, is not a file, or does not
     *                                  have a supported image extension
     */
    private static void validateSingleImageInput(Path input) {
        // Make sure the path exists.
        if (!Files.exists(input)) {
            throw new IllegalArgumentException("Input image does not exist: " + input);
        }

        // This tool expects actual files, not folders.
        if (!Files.isRegularFile(input)) {
            throw new IllegalArgumentException("Input must be a file, not a directory: " + input);
        }

        // Quick extension check so bad inputs fail early.
        if (!isImageFile(input)) {
            throw new IllegalArgumentException("Unsupported image format: " + input.getFileName());
        }
    }

    /**
     * Checks whether a file path has a supported image extension.
     * <p>Supported formats are jpg, jpeg, png, tif, tiff. The check is case-insensitive.</p>
     *
     * @param path file path to check
     * @return {@code true} if the file extension is supported; {@code false} otherwise
     */
    private static boolean isImageFile(Path path) {
        // Lowercase the name first so .JPG and .jpg both pass.
        String name = path.getFileName().toString().toLowerCase(Locale.US);
        return name.endsWith(".jpg")
                || name.endsWith(".jpeg")
                || name.endsWith(".png")
                || name.endsWith(".tif")
                || name.endsWith(".tiff");
    }

    /**
     * Removes the final file extension to get the filename without extension.
     *
     * @param filename file name with extension to modify
     * @return filename without its final extension
     */
    private static String stripExtension(String filename) {
        // Remove the last extension part from the file name.
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    /**
     * Deletes directory contents to remove previous tray results.
     *
     * @param dir directory path whose contents should be removed
     * @throws IOException if deleting any file or subfolder fails
     */
    private static void clearDirectoryContents(Path dir) throws IOException {
        // Nothing to do if the folder is not there yet.
        if (!Files.exists(dir)) {
            return;
        }

        // Delete everything inside the folder, but keep the folder itself.
        try (Stream<Path> stream = Files.list(dir)) {
            for (Path path : stream.collect(Collectors.toList())) {
                deleteRecursively(path);
            }
        }
    }

    /**
     * Recursively deletes a file or directory.
     *
     * @param path file or directory to delete
     * @throws IOException if deletion fails
     */
    private static void deleteRecursively(Path path) throws IOException {
        // If this is a directory, clear its children first.
        if (Files.isDirectory(path)) {
            try (Stream<Path> stream = Files.list(path)) {
                for (Path child : stream.collect(Collectors.toList())) {
                    deleteRecursively(child);
                }
            }
        }

        // Delete the file or empty directory.
        Files.deleteIfExists(path);
    }
}