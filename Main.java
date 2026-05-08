import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.*;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {

    // Load the OpenCV native library one time when the class gets loaded.
    // If this is missing, OpenCV image I/O and processing calls will fail.
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) throws IOException {
        // Expect:
        // args[0] = output folder
        // args[1...] = one or more input image paths
        if (args.length < 2) {
            System.out.println(
                    "Usage: java -Djava.library.path=/path/to/opencv/native -cp .:opencv-4xx.jar Main <output-dir> <image1> <image2> ...");
            System.out.println("On Windows, replace ':' with ';' in the classpath.");
            return;
        }

        Path outputDir = Paths.get(args[0]);
        Path cardsDir = outputDir.resolve("cards");

        // Make sure the output folders exist before writing anything.
        Files.createDirectories(outputDir);
        Files.createDirectories(cardsDir);

        // Clear old card result folders so each run starts clean.
        clearDirectoryContents(cardsDir);

        // Process every input image passed in after the output directory.
        for (int i = 1; i < args.length; i++) {
            Path inputImage = Paths.get(args[i]);

            // Basic checks before trying to read the image.
            validateSingleImageInput(inputImage);

            // Process the image and save outputs into its own folder.
            processCardImage(inputImage, cardsDir, i);
        }

        System.out.println("Finished. Results written to: " + outputDir.toAbsolutePath());
    }

    private static void processCardImage(Path imagePath, Path cardsDir, int index) throws IOException {
        // Read the input image as a color image.
        Mat raw = Imgcodecs.imread(imagePath.toString(), Imgcodecs.IMREAD_COLOR);
        if (raw.empty()) {
            throw new IOException("Could not read image: " + imagePath);
        }

        // Run the card-cleaning pipeline.
        CardCleaner.CardResult result = CardCleaner.processCard(raw);

        // Make a per-card folder name using the input order and original file name.
        String baseName = stripExtension(imagePath.getFileName().toString());
        Path cardDir = cardsDir.resolve(String.format(Locale.US, "card_%02d_%s", index, baseName));
        Files.createDirectories(cardDir);

        // Save every stage with numbered names so they appear in order:
        // raw -> clahe -> color separation -> clean
        Imgcodecs.imwrite(cardDir.resolve("01_raw.png").toString(), result.rawInput);
        Imgcodecs.imwrite(cardDir.resolve("02_clahe_white_balance.png").toString(), result.claheWhiteBalanced);
        Imgcodecs.imwrite(cardDir.resolve("03_clean.png").toString(), result.corrected);

        System.out.printf(Locale.US, "Processed %s -> %s%n", imagePath.getFileName(), cardDir.getFileName());
    }

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

    private static boolean isImageFile(Path path) {
        // Lowercase the name first so .JPG and .jpg both pass.
        String name = path.getFileName().toString().toLowerCase(Locale.US);
        return name.endsWith(".jpg")
                || name.endsWith(".jpeg")
                || name.endsWith(".png")
                || name.endsWith(".tif")
                || name.endsWith(".tiff");
    }

    private static String stripExtension(String filename) {
        // Remove the last extension part from the file name.
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

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