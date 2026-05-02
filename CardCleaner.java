import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class CardCleaner {

    public static CardResult[] processCard(Mat warpedRaw) {
        // First thing: equalize each BGR, HSV channel separately
        Mat bgrEqualized = equalizeBgrChannels(warpedRaw);
        Mat hsvEqualized = equalizeHsvChannels(warpedRaw);
        Mat labEqualized = equalizeLabChannels(warpedRaw);
        Mat labCLAHE = claheLab(warpedRaw);

        // Run labelers on equalized image
        char[][] bgrLabels = labelPixels(bgrEqualized);
        char[][] hsvLabels = labelPixels(hsvEqualized);
        char[][] labLabels = labelPixels(labEqualized);
        char[][] labCLAHELabels = labelPixels(labCLAHE);


        CardResult[] results = new CardResult[4];
        // Use original image for final background estimation and output
        results[0] = estimateBGandResult(hsvLabels, hsvEqualized, warpedRaw, "HSV");
        results[1] = estimateBGandResult(bgrLabels, bgrEqualized, warpedRaw, "BGR");
        results[2] = estimateBGandResult(labLabels, bgrEqualized, warpedRaw, "LAB");
        results[3] = estimateBGandResult(labCLAHELabels, bgrEqualized, warpedRaw, "CLAHElab");
        
        return results;
    }

    public static CardResult estimateBGandResult(char[][] labels, Mat equalized, Mat warpedRaw, String colorSpace) {
        // Use original image for final background estimation and output
        Scalar backgroundColor = BackgroundColorEstimator.estimateBackgroundColor(warpedRaw, labels);
        Mat corrected = processLabels(equalized, labels, backgroundColor);

        CardResult result = new CardResult();
        result.corrected = corrected;
        result.labels = labels;
        result.warpedRaw = warpedRaw;
        result.colorSpace = colorSpace;
        
        return result;
    } 

    public static char[][] labelPixels(Mat equalized) {
        int rows = equalized.rows();
        int cols = equalized.cols();
        char[][] labels = new char[rows][cols];

        // Initialize everything to '.'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                labels[r][c] = '.';
            }
        }
        labels = LabelCoral.labelCoralPixels(equalized, labels);
        labels = LabelAlgae.labelAlgaePixels(equalized, labels);
        labels = LabelSilt.labelSiltPixels(equalized, labels);
        // labels = LabelShadow.labelShadowPixels(equalized, labels);

        return labels;
    }

    private static Mat equalizeBgrChannels(Mat bgr) {
        List<Mat> channels = new ArrayList<>(3);
        Mat nbgr = bgr.clone();

        Core.split(nbgr, channels);

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

    //only equalizes saturation and value
    private static Mat equalizeHsvChannels(Mat bgr) {
        List<Mat> channels = new ArrayList<>(3);
        Mat hsv = new Mat();
        Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV);
        Core.split(hsv, channels);

        Mat h = channels.get(0); //likely harmful if equalized
        Mat sEq = new Mat();
        Mat vEq = new Mat();

        Imgproc.equalizeHist(channels.get(1), sEq);
        Imgproc.equalizeHist(channels.get(2), vEq);

        Mat equalized = new Mat();
        List<Mat> merged = new ArrayList<>(3);
        merged.add(h);
        merged.add(sEq);
        merged.add(vEq);
        Core.merge(merged, equalized);

        Imgproc.cvtColor(equalized, equalized, Imgproc.COLOR_HSV2BGR);
        return equalized;
    }

    //only equalizes lightness
    private static Mat equalizeLabChannels(Mat bgr) {
        List<Mat> channels = new ArrayList<>(3);
        Mat lab = new Mat();
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab);
        Core.split(lab, channels);

        Mat lEq = new Mat(); //likely harmful if equalized
        Mat a = channels.get(1);
        Mat b = channels.get(2);

        Imgproc.equalizeHist(channels.get(0), lEq);

        Mat equalized = new Mat();
        List<Mat> merged = new ArrayList<>(3);
        merged.add(lEq);
        merged.add(a);
        merged.add(b);
        Core.merge(merged, equalized);

        Imgproc.cvtColor(equalized, equalized, Imgproc.COLOR_Lab2BGR);
        return equalized;
    }

    //clahe lightness
    private static Mat claheLab(Mat bgr) {
        List<Mat> channels = new ArrayList<>(3);
        Mat lab = new Mat();
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab);
        Core.split(lab, channels);

        CLAHE clahe = Imgproc.createCLAHE(2.0, new Size(8,8));
        clahe.apply(channels.get(0), channels.get(0));

        Mat equalized = new Mat();
        Core.merge(channels, equalized);

        Imgproc.cvtColor(equalized, equalized, Imgproc.COLOR_Lab2BGR);
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
        public String colorSpace;
    }
}