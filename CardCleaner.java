import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class CardCleaner {

    public static CardResult processCard(Mat warpedRaw) {
        int rows = warpedRaw.rows();
        int cols = warpedRaw.cols();

        // First thing: equalize each BGR channel separately
        Mat equalized = equalizeBgrChannels(warpedRaw);

        char[][] labels = new char[rows][cols];

        // Initialize everything to '.'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                labels[r][c] = '.';
            }
        }

        // Run labelers on equalized image
        labels = LabelCoral.labelCoralPixels(equalized, labels);
        labels = LabelAlgae.labelAlgaePixels(equalized, labels);
        labels = LabelSilt.labelSiltPixels(equalized, labels);

        // labels = LabelShadow.labelShadowPixels(equalized, labels);

        // Use original image for final background estimation and output
        Scalar backgroundColor = BackgroundColorEstimator.estimateBackgroundColor(warpedRaw, labels);
        Mat corrected = processLabels(equalized, labels, backgroundColor);

        CardResult result = new CardResult();
        result.warpedRaw = equalized.clone();
        result.corrected = corrected;
        result.labels = labels;

        return result;
    }

    private static Mat equalizeBgrChannels(Mat bgr) {
        List<Mat> channels = new ArrayList<>(3);
        Core.split(bgr, channels);

        Mat bEq = new Mat();
        Mat gEq = new Mat();
        Mat rEq = new Mat();

        Imgproc.equalizeHist(channels.get(0), bEq);
        Imgproc.equalizeHist(channels.get(1), gEq);
        Imgproc.equalizeHist(channels.get(2), rEq);

        Mat equalized = new Mat();
        List<Mat> merged = new ArrayList<>(3);
        merged.add(bEq);
        merged.add(gEq);
        merged.add(rEq);
        Core.merge(merged, equalized);

        return equalized;
    }

    private static Mat processLabels(Mat source, char[][] labels, Scalar backgroundColor) {
        Mat output = source.clone();

        int rows = output.rows();
        int cols = output.cols();
        int channels = output.channels();

        int bgB = clampToByte((int) Math.round(backgroundColor.val[0]));
        int bgG = clampToByte((int) Math.round(backgroundColor.val[1]));
        int bgR = clampToByte((int) Math.round(backgroundColor.val[2]));

        byte[] data = new byte[(int) (output.total() * channels)];
        output.get(0, 0, data);

        int index = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                if (labels[r][c] == 'A' || labels[r][c] == 'S' || labels[r][c] == 'H') {
                    data[index] = (byte) bgB;
                    data[index + 1] = (byte) bgG;
                    data[index + 2] = (byte) bgR;
                }

                /*
                 * else if (labels[r][c] == 'C') {
                 * data[index] = (byte) 0; // B
                 * data[index + 1] = (byte) 0; // G
                 * data[index + 2] = (byte) 0; // R
                 * }
                 */

                index += 3;
            }
        }

        output.put(0, 0, data);
        return output;
    }

    private static int clampToByte(int value) {
        return Math.max(0, Math.min(255, value));
    }

    public static class CardResult {
        public Mat warpedRaw;
        public Mat corrected;
        public char[][] labels;
    }
}