import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

public class CardCleaner {
    // Fixed replacement color for removable non-coral pixels.
    // OpenCV uses BGR order, so this is dim white / light gray.
    private static final Scalar DIM_WHITE = new Scalar(220, 220, 220);

    public static CardResult[] processCard(Mat warpedRaw) {

        Mat before = warpedRaw.clone();
        // First thing: equalize each BGR, HSV channel separately
        Mat bgrEqualized = equalizeBgrChannels(warpedRaw);
        Mat hsvEqualized = equalizeHsvChannels(warpedRaw);
        Mat labEqualized = equalizeLabChannels(warpedRaw);
        Mat labCLAHE = claheLab(warpedRaw);
        Mat labCLAHEwhiteBalance = whiteBalancedBgr(claheLab(warpedRaw));

        // Run labelers on equalized image
        char[][] bgrLabels = labelPixels(bgrEqualized);
        char[][] hsvLabels = labelPixels(hsvEqualized);
        char[][] labLabels = labelPixels(labEqualized);
        char[][] labCLAHELabels = labelPixels(labCLAHE);
        char[][] labCLAHEwhiteBalanceLabels = labelPixels(labCLAHEwhiteBalance);

        CardResult[] results = new CardResult[19];
        // Use original image for final background estimation and output
        // results[0] = estimateBGandResult(hsvLabels, hsvEqualized, warpedRaw, "HSV-Equalization");
        // results[1] = estimateBGandResult(bgrLabels, bgrEqualized, warpedRaw, "BGR-Equalization");
        // results[2] = estimateBGandResult(labLabels, labEqualized, warpedRaw, "LAB-Equalization");
        // results[3] = estimateBGandResult(labCLAHELabels, labCLAHE, warpedRaw, "CLAHElab");
        results[4] = estimateBGandResult(labCLAHEwhiteBalanceLabels, labCLAHEwhiteBalance, warpedRaw, "CLAHElabWhiteBalance");
        
        //otsu thresholding on the images
        results[5] = applyOtsu(warpedRaw, "Otsu");
        // results[6] = applyOtsu(results[0].corrected, "HSVOtsu");
        // results[7] = applyOtsu(results[1].corrected, "BGROtsu");
        // results[8] = applyOtsu(results[2].corrected, "LABOtsu");
        // results[9] = applyOtsu(results[3].corrected, "CLAHElabOtsu");
        // results[10] = applyOtsu(results[4].corrected, "CLAHElabWhiteBalanceOtsu");
        // System.out.println("Rows: " + warpedRaw.rows() + " Cols: " + warpedRaw.cols() +"\n");
        // System.out.println("Corrected Rows: " + results[4].corrected.rows() + " Corrected Cols: " + results[4].corrected.cols() +"\n");

        //applying the otsu threshold image mask onto the processed images
        // results[11] = applyMask(warpedRaw, results[5].corrected,"OtsuMaskRaw");
        // results[12] = applyMask(results[0].corrected, results[6].corrected, "OtsuMaskHSV");
        // results[13] = applyMask(results[1].corrected, results[7].corrected,"OtsuMaskBGR");
        // results[14] = applyMask(results[2].corrected, results[8].corrected,"OtsuMaskLAB");
        // results[15] = applyMask(results[3].corrected, results[9].corrected,"OtsuMaskCLAHElab");
        // results[16] = applyMask(results[4].corrected, results[10].corrected,"OtsuMaskCLAHElabWhiteBalance");

        results[17] = connectedComponents(results[5].corrected, "OtsuConnectedComponents");

        results[18] = applyMask(warpedRaw,  results[17].corrected, "OtsuConnectedComponentsMask");


        return results;
    }

    public static CardResult connectedComponents(Mat input, String method) {

        Mat inverted = new Mat();
        Core.bitwise_not(input, inverted);

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int numLabels = Imgproc.connectedComponentsWithStats(inverted, labels, stats, centroids);

        Mat filteredOutput = Mat.zeros(input.size(), input.type());

        int minAreaThreshold = 1500; // Example: only keep components larger than 500px

        for (int i = 1; i < numLabels; i++) {
            // 3. Extract area for current label 'i'
            int area = (int) stats.get(i, Imgproc.CC_STAT_AREA)[0];

            if (area >= minAreaThreshold) {
                // 4. Create a mask for this specific component
                Mat mask = new Mat();
                Core.compare(labels, new Scalar(i), mask, Core.CMP_EQ);
                
                // Add component to final result
                Core.bitwise_or(filteredOutput, mask, filteredOutput);
            }
        }

        Mat inverted_filtered_Output = new Mat();
        Core.bitwise_not(filteredOutput, inverted_filtered_Output);

        CardResult result = new CardResult();
        result.corrected = inverted_filtered_Output;
        result.labels = null;
        result.input = input;
        result.method = method;

        return result;
    }

    public static CardResult applyMask(Mat base, Mat mask, String method) {
        // Mat corrected = new Mat();
        // Core.bitwise_and(warpedRaw, warpedRaw, corrected, mask);
        Mat invertedMask = new Mat();
        Core.bitwise_not(mask, invertedMask);

        // 1. Create a canvas of the same size and type, filled with white (255, 255, 255)
        Mat corrected = new Mat(base.size(), base.type(), new Scalar(255, 255, 255));
        
        // 2. Copy only the pixels where the mask is non-zero onto the white canvas
        base.copyTo(corrected, invertedMask);

        CardResult result = new CardResult();
        result.corrected = corrected;
        result.labels = null;
        result.input = base;
        result.method = method;
        
        return result;
    }

    public static CardResult estimateBGandResult(char[][] labels, Mat equalized, Mat warpedRaw, String method) {
        // Use original image for final background estimation and output
        // Scalar backgroundColor = BackgroundColorEstimator.estimateBackgroundColor(warpedRaw.clone(), labels);
        Mat corrected = processLabels(equalized, labels, DIM_WHITE);

        CardResult result = new CardResult();
        result.corrected = corrected;
        result.labels = labels;
        result.input = warpedRaw;
        result.method = method;
        
        return result;
    } 

    public static CardResult applyOtsu(Mat warpedRaw, String method) {
        int low = 0;
        int max_Value = 255;

        Mat otsu = new Mat();
        Mat gray = new Mat();
        Imgproc.cvtColor(warpedRaw, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(gray, gray, new Size(5, 5), 0);
        Imgproc.threshold(gray, otsu, low, max_Value, Imgproc.THRESH_OTSU);

        CardResult otsuResult = new CardResult();
        otsuResult.corrected = otsu;
        otsuResult.labels = null;
        otsuResult.input = warpedRaw;
        otsuResult.method = method;

        return otsuResult;
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

        CLAHE clahe = Imgproc.createCLAHE(3.0, new Size(8,8));
        clahe.apply(channels.get(0), channels.get(0));

        Mat equalized = new Mat();
        Core.merge(channels, equalized);

        Imgproc.cvtColor(equalized, equalized, Imgproc.COLOR_Lab2BGR);
        return equalized;
    }

    private static Mat whiteBalancedBgr(Mat bgr) {
        Mat floatImg = new Mat();


        bgr.convertTo(floatImg, CvType.CV_32F);

        List<Mat> channels = new ArrayList<>(3);
        Core.split(floatImg, channels);

        Mat b = channels.get(0);
        Mat g = channels.get(1);
        Mat r = channels.get(2);

        double mb = Core.mean(b).val[0];
        double mg = Core.mean(g).val[0];
        double mr = Core.mean(r).val[0];

        double m = (mb + mg + mr) / 3.0;

        double bScale = mb == 0.0 ? 1.0 : m / mb;
        double gScale = mg == 0.0 ? 1.0 : m / mg;
        double rScale = mr == 0.0 ? 1.0 : m / mr;

        b.convertTo(b, CvType.CV_32F, bScale);
        g.convertTo(g, CvType.CV_32F, gScale);
        r.convertTo(r, CvType.CV_32F, rScale);

        List<Mat> balancedChannels = new ArrayList<>(3);
        balancedChannels.add(b);
        balancedChannels.add(g);
        balancedChannels.add(r);

        Mat balancedFloat = new Mat();
        Core.merge(balancedChannels, balancedFloat);

        Mat whiteBalanced = new Mat();
        balancedFloat.convertTo(whiteBalanced, CvType.CV_8UC3);

        return whiteBalanced;
    }

    public static Mat claheLabWhiteBalanced(Mat bgr) {
        return whiteBalancedBgr(claheLab(bgr));
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
        public Mat input;
        public Mat corrected;
        public char[][] labels;
        public String method;
    }
}