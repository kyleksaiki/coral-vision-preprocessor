import org.opencv.core.Mat;
import org.opencv.core.Scalar;

public class TrayCleaner {

    // Fixed replacement color for removable non-coral pixels.
    // OpenCV uses BGR order, so this is dim white / light gray.
    private static final Scalar DIM_WHITE = new Scalar(220, 220, 220);

    public static TrayResult processTray(Mat warpedRaw) {
        int rows = warpedRaw.rows();
        int cols = warpedRaw.cols();

        // STEP 1:
        // CLAHE + white balance happen in their own file.
        Mat claheWhiteBalanced = CardLightingNormalizer.claheLabThenWhiteBalance(warpedRaw);

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

    private static int clampToByte(int value) {
        return Math.max(0, Math.min(255, value));
    }

    public static class TrayResult {
        public Mat rawInput;
        public Mat claheWhiteBalanced;
        public Mat corrected;
        public char[][] labels;
    }
}
