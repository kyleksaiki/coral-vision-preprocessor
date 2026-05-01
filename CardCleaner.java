import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class CardCleaner {

    public static CardResult processCard(Mat warpedRaw) {
        int rows = warpedRaw.rows();
        int cols = warpedRaw.cols();

        // One label per pixel. Starts empty, then each pass fills things in.
        char[][] labels = new char[rows][cols];

        // Start everything as unlabeled.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                labels[r][c] = '.';
            }
        }

        // Run the labeling passes in order.
        // Coral first so other passes can avoid wiping it out.
        labels = LabelCoral.labelCoralPixels(warpedRaw, labels);
        labels = LabelAlgae.labelAlgaePixels(warpedRaw, labels);
        labels = LabelSilt.labelSiltPixels(warpedRaw, labels);
        labels = LabelShadow.labelShadowPixels(warpedRaw, labels);

        // Estimate a card-like background color from the remaining unlabeled area.
        Scalar backgroundColor = BackgroundColorEstimator.estimateBackgroundColor(warpedRaw, labels);

        // Build the cleaned image by replacing unwanted labels with that background
        // color.
        Mat corrected = processLabels(warpedRaw, labels, backgroundColor);

        CardResult result = new CardResult();
        result.warpedRaw = warpedRaw.clone();
        result.corrected = corrected;
        result.labels = labels;

        return result;
    }

    private static Mat processLabels(Mat source, char[][] labels, Scalar backgroundColor) {
        // Work on a copy so the source image stays untouched.
        Mat output = source.clone();

        int rows = output.rows();
        int cols = output.cols();
        int channels = output.channels();

        // Turn the estimated background color into regular byte-safe values.
        int bgB = clampToByte((int) Math.round(backgroundColor.val[0]));
        int bgG = clampToByte((int) Math.round(backgroundColor.val[1]));
        int bgR = clampToByte((int) Math.round(backgroundColor.val[2]));

        // Pull image data into one flat byte array for faster pixel edits.
        // This is a lot faster than using the matrix calls for each value in matrix
        byte[] data = new byte[(int) (output.total() * channels)];
        output.get(0, 0, data);

        int index = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                // Anything marked as algae, silt, or shadow gets painted over
                // with the estimated background color.
                if (labels[r][c] == 'A' || labels[r][c] == 'S' || labels[r][c] == 'H') {
                    data[index] = (byte) bgB;
                    data[index + 1] = (byte) bgG;
                    data[index + 2] = (byte) bgR;
                }

                /*
                 * Optional debug block:
                 * if you want to draw coral in black instead of leaving it alone,
                 * uncomment this.
                 *
                 * else if (labels[r][c] == 'C') {
                 * data[index] = (byte) 0; // B
                 * data[index + 1] = (byte) 0; // G
                 * data[index + 2] = (byte) 0; // R
                 * }
                 */

                // Move to the next pixel in the flat BGR byte array.
                index += 3;
            }
        }

        output.put(0, 0, data);
        return output;
    }

    private static int clampToByte(int value) {
        // Keeps values inside the valid image byte range.
        return Math.max(0, Math.min(255, value));
    }

    public static class CardResult {
        // Original cropped card image.
        public Mat warpedRaw;

        // Final cleaned image after unwanted stuff is painted over.
        public Mat corrected;

        // Per-pixel labels from the pipeline.
        public char[][] labels;
    }
}