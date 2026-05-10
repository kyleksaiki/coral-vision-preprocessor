import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.IOException;
import java.nio.file.*;
import java.util.Locale;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Main program that takes input images ending in JPG from the input directory and processes each image before writing output images under the output directory.
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
     * Program entry point.
     * 
     * <p>The first command-line argument is the output directory.  
     * Every argument after is treated as an input tray image</p>
     * 
     * @param args command-line arguments:
     *             {@code args[0]} is the output directory.
     *             {@code args[1...]} are input image paths
     * @throws IOException if creating directories, reading images, writing images or deleting old output files fails.
     * 
     * */
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
        Path traysDir = outputDir.resolve("trays");

        // Make sure the output folders exist before writing anything.
        Files.createDirectories(outputDir);
        Files.createDirectories(traysDir);

        // Clear old tray result folders so each run starts clean.
        clearDirectoryContents(traysDir);

        // Process every input image passed in after the output directory.
        for (int i = 1; i < args.length; i++) {
            Path inputImage = Paths.get(args[i]);

            // Basic checks before trying to read the image.
            validateSingleImageInput(inputImage);

            // Process the image and save outputs into its own folder.
            processTrayImage(inputImage, traysDir, i);
        }

        System.out.println("Finished. Results written to: " + outputDir.toAbsolutePath());
    }

    /**
     * Process a single tray image and writes its output images to a dedicated folder.
     * 
     * <p>The method reads the image using OpenCV, creates a folder for each input image
     * and then saves each of the processed images as .png</p>
     *
     * @param imagePath path to input image
     * @param traysDir output directory path
     * @param index image id to distinguish image folders in the output directory
     * @throws IOException if the image cannot be read or the output folder/files cannot be creates
     *
     * */
    private static void processTrayImage(Path imagePath, Path traysDir, int index) throws IOException {
        // Read the input image as a color image.
        Mat raw = Imgcodecs.imread(imagePath.toString(), Imgcodecs.IMREAD_COLOR);
        if (raw.empty()) {
            throw new IOException("Could not read image: " + imagePath);
        }

        // Run the tray-cleaning pipeline.
        TrayCleaner.TrayResult result = TrayCleaner.processTray(raw);

        // Make a per-tray folder name using the input order and original file name.
        String baseName = stripExtension(imagePath.getFileName().toString());
        Path trayDir = traysDir.resolve(String.format(Locale.US, "tray_%02d_%s", index, baseName));
        Files.createDirectories(trayDir);

        // Save every stage with numbered names so they appear in order:
        // raw -> clahe -> color separation -> clean
        Imgcodecs.imwrite(trayDir.resolve("01_raw.png").toString(), result.rawInput);
        Imgcodecs.imwrite(trayDir.resolve("02_clahe_white_balance.png").toString(), result.claheWhiteBalanced);
        Imgcodecs.imwrite(trayDir.resolve("03_clean.png").toString(), result.corrected);

        System.out.printf(Locale.US, "Processed %s -> %s%n", imagePath.getFileName(), trayDir.getFileName());
    }

    /**
     * Validates that a image path exists and has supported image format
     *
     * @param input path to validate
     * @throws IllegalArgumentException if the path does not exist, if not a file, or doesn't have correct image extension
     * */
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
     * <p>Supported formats are jpg, jpeg, png, tif, tiff.  The check is case-insensitive</p>
     * @param path file path to check
     * @return {@code true} if the file extension is supported;
     *         {@code false} otherwise
     * */
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
     * Removes final file extension to get filename without extension
     *
     * @param filename final name with extension to modify
     * @return filename without its final extension
     *
     * */
    private static String stripExtension(String filename) {
        // Remove the last extension part from the file name.
        int dot = filename.lastIndexOf('.');
        return dot > 0 ? filename.substring(0, dot) : filename;
    }

    /**
     * Deletes directory contents to remove previous card results
     * @param dir directory path whose contents should be removed
     * @throws IOException if deleting any file or subfolder fails
     * */
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
     * Recursively delete a file or directory.
     *
     * @param path file or directory to delete
     * @throws IOException if deletion fails
     * */
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