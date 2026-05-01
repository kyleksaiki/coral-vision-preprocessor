import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class BackgroundColorEstimator {

    // Minimum fraction of the unlabeled area that should count as usable background
    // before falling back to the whole allowed mask.
    private static final double CARD_BACKGROUND_FRAC_MIN = 0.08;

    public static Scalar estimateBackgroundColor(Mat bgrImage, char[][] labels) {
        // Convert to HSV because brightness and saturation are easier to work with
        // there.
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrImage, hsv, Imgproc.COLOR_BGR2HSV);

        // Split HSV into separate channels so we can threshold S and V directly.
        List<Mat> hsvChannels = new ArrayList<>();
        Core.split(hsv, hsvChannels);

        Mat s = hsvChannels.get(1);
        Mat v = hsvChannels.get(2);

        // Only unlabeled pixels are allowed to contribute to the estimated background
        // color.
        Mat allowedMask = buildAllowedBackgroundMask(labels);

        // If nothing is available, just return a light gray fallback.
        if (Core.countNonZero(allowedMask) == 0) {
            return new Scalar(245, 245, 245);
        }

        // Use masked percentiles instead of hard-coded values so this adapts a bit
        // to cards with different lighting.
        int vThr = percentile(v, 60.0, allowedMask);
        int sThr = percentile(s, 60.0, allowedMask);

        // Background is expected to be relatively bright and low saturation.
        Mat bright = new Mat();
        Mat lowSat = new Mat();
        Imgproc.threshold(v, bright, vThr, 255, Imgproc.THRESH_BINARY);
        Imgproc.threshold(s, lowSat, sThr, 255, Imgproc.THRESH_BINARY_INV);

        // Keep only pixels that are both bright and low saturation,
        // and also allowed by the label mask.
        Mat baseMask = new Mat();
        Core.bitwise_and(bright, lowSat, baseMask);
        Core.bitwise_and(baseMask, allowedMask, baseMask);

        // If the filtered background region is too small, use all allowed unlabeled
        // pixels instead.
        double allowedArea = Math.max(1.0, Core.countNonZero(allowedMask));
        if (Core.countNonZero(baseMask) < CARD_BACKGROUND_FRAC_MIN * allowedArea) {
            baseMask = allowedMask.clone();
        }

        // Final safety check in case something went weird.
        if (Core.countNonZero(baseMask) == 0) {
            return new Scalar(245, 245, 245);
        }

        // Average the BGR values over the chosen background mask.
        Scalar mean = Core.mean(bgrImage, baseMask);
        return new Scalar(mean.val[0], mean.val[1], mean.val[2]);
    }

    private static Mat buildAllowedBackgroundMask(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        Mat mask = new Mat(rows, cols, CvType.CV_8U);
        byte[] data = new byte[rows * cols];

        int index = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                // Only plain unlabeled pixels are background candidates.
                if (labels[r][c] == '.') {
                    data[index] = (byte) 255;
                } else {
                    data[index] = 0;
                }
                index++;
            }
        }

        mask.put(0, 0, data);
        return mask;
    }

    private static int percentile(Mat singleChannel8u, double pct, Mat mask) {
        // Pull channel data and mask data into arrays so we can build the histogram
        // ourselves.
        byte[] data = new byte[(int) singleChannel8u.total()];
        singleChannel8u.get(0, 0, data);

        byte[] maskData = new byte[(int) mask.total()];
        mask.get(0, 0, maskData);

        int[] hist = new int[256];
        int count = 0;

        // Histogram only over masked-in pixels.
        for (int i = 0; i < data.length; i++) {
            if ((maskData[i] & 0xFF) != 0) {
                hist[data[i] & 0xFF]++;
                count++;
            }
        }

        // No valid pixels -> just return 0.
        if (count == 0) {
            return 0;
        }

        // Convert percentile to a target rank in the sorted masked pixel list.
        int target = (int) Math.round((pct / 100.0) * (count - 1));
        int cumulative = 0;

        // Walk the histogram until we hit that rank.
        for (int value = 0; value < 256; value++) {
            cumulative += hist[value];
            if (cumulative > target) {
                return value;
            }
        }

        // Should not normally get here, but 255 is a safe last return.
        return 255;
    }
}