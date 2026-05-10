import org.opencv.core.Mat;
import org.opencv.core.Scalar;

/**
 * Cleans a tray image by identifying coral and removable non-coral distractions,
 * then replacing the removable areas with a fixed dim white color.
 *
 * <p>The tray cleaning pipeline works in this order:</p>
 *
 * <ol>
 *   <li>Normalize lighting using CLAHE and white balance.</li>
 *   <li>Initialize a label grid for every pixel.</li>
 *   <li>Label coral pixels first so they are protected.</li>
 *   <li>Label non-coral distractions such as algae and silt.</li>
 *   <li>Replace removable labeled pixels with dim white.</li>
 * </ol>
 *
 * <p>Label meanings expected by this class:</p>
 *
 * <ul>
 *   <li>{@code 'C'} or {@code 'c'} = coral, preserved</li>
 *   <li>{@code 'A'} or {@code 'a'} = algae, replaced</li>
 *   <li>{@code 'S'} or {@code 's'} = silt, replaced</li>
 *   <li>{@code 'H'} or {@code 'h'} = shadow, replaced if present</li>
 *   <li>{@code '.'} = background / unlabeled pixel</li>
 * </ul>
 */
public class TrayCleaner {

    // Fixed replacement color for removable non-coral pixels.
    // OpenCV uses BGR order, so this is dim white / light gray.
    private static final Scalar DIM_WHITE = new Scalar(220, 220, 220);

    /**
     * Runs the full tray-cleaning pipeline on a warped raw tray image.
     *
     * <p>The method first normalizes lighting, then labels coral before labeling
     * algae and silt. Coral is labeled first so later labelers can avoid overwriting
     * it. Finally, removable labels are replaced with dim white while coral and
     * unlabeled pixels remain unchanged.</p>
     *
     * @param warpedRaw input tray image, already warped/cropped into the expected perspective
     * @return a {@link TrayResult} containing the raw input, normalized image,
     *         corrected output image, and final label grid
     */
    public static TrayResult processTray(Mat warpedRaw) {
        int rows = warpedRaw.rows();
        int cols = warpedRaw.cols();

        // STEP 1:
        // CLAHE + white balance happen in their own file.
        Mat claheWhiteBalanced = TrayLightingNormalizer.claheLabThenWhiteBalance(warpedRaw);

        char[][] labels = new char[rows][cols];

        // Initialize everything to '.'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                labels[r][c] = '.';
            }
        }

        // Coral must run first so algae/silt/shadow cannot overwrite it.
        labels = LabelCoral.labelCoralPixels(claheWhiteBalanced, labels);

        // Non-coral distraction labelers run after coral.
        // These should not override coral pixels.
        labels = LabelAlgae.labelAlgaePixels(claheWhiteBalanced, labels);
        labels = LabelSilt.labelSiltPixels(claheWhiteBalanced, labels);
    
        // Replace removable labels with fixed dim white.
        Mat corrected = processLabels(claheWhiteBalanced, labels, DIM_WHITE);

        TrayResult result = new TrayResult();
        result.rawInput = warpedRaw.clone();
        result.claheWhiteBalanced = claheWhiteBalanced.clone();
        result.corrected = corrected;
        result.labels = labels;

        return result;
    }

    /**
     * Creates a corrected copy of the source image by replacing removable labels
     * with a provided replacement color.
     *
     * <p>This method does not modify the source image directly. It clones the source,
     * edits the clone's pixel buffer, and returns the edited image.</p>
     *
     * <p>The following labels are replaced:</p>
     *
     * <ul>
     *   <li>{@code 'A'} or {@code 'a'} for algae</li>
     *   <li>{@code 'S'} or {@code 's'} for silt</li>
     *   <li>{@code 'H'} or {@code 'h'} for shadow, if shadow labels are present</li>
     * </ul>
     *
     * <p>Coral labels are intentionally preserved. The commented-out coral debug
     * block can be used during development to visualize coral pixels in purple,
     * but it is disabled in the normal cleaning output.</p>
     *
     * @param source source image to clone and edit
     * @param labels label grid with one label per image pixel
     * @param replacementColor BGR replacement color used for removable pixels
     * @return corrected image with removable labeled pixels replaced
     */
    private static Mat processLabels(Mat source, char[][] labels, Scalar replacementColor) {
        Mat output = source.clone();

        int rows = output.rows();
        int cols = output.cols();
        int channels = output.channels();

        int bgB = clampToByte((int) Math.round(replacementColor.val[0]));
        int bgG = clampToByte((int) Math.round(replacementColor.val[1]));
        int bgR = clampToByte((int) Math.round(replacementColor.val[2]));

        byte[] data = new byte[(int) (output.total() * channels)];
        output.get(0, 0, data);

        int index = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                char label = labels[r][c];

                // Replace algae, silt, and shadow with dim white.
                if (label == 'A' || label == 'a'
                        || label == 'S' || label == 's'
                        || label == 'H' || label == 'h') {

                    data[index] = (byte) bgB;
                    data[index + 1] = (byte) bgG;
                    data[index + 2] = (byte) bgR;
                }
                /*
                 * else if (label == 'C') {
                 * data[index] = (byte) 128; // B
                 * data[index + 1] = (byte) 0; // G
                 * data[index + 2] = (byte) 128; // R
                 * }
                 */

                index += channels;
            }
        }

        output.put(0, 0, data);
        return output;
    }

    /**
     * Clamps an integer into the valid unsigned byte range for image channels.
     *
     * <p>OpenCV image channels use values from 0 to 255. This helper prevents
     * replacement color values from falling outside that range.</p>
     *
     * @param value input value
     * @return {@code value} clamped between 0 and 255
     */
    private static int clampToByte(int value) {
        return Math.max(0, Math.min(255, value));
    }

    /**
     * Container for all major outputs of the tray-cleaning pipeline.
     *
     * <p>This object stores both intermediate and final results so callers can save,
     * inspect, or debug different stages of processing.</p>
     */
    public static class TrayResult {
        public Mat rawInput;
        public Mat claheWhiteBalanced;
        public Mat corrected;
        public char[][] labels;
    }
}
