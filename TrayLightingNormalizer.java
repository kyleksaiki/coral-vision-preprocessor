import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Normalizes tray image lighting before coral, algae, and silt labeling.
 *
 * <p>This class applies a simple preprocessing pipeline intended to make later
 * color-threshold-based labelers more stable across uneven lighting conditions.</p>
 *
 * <p>The normalization pipeline is:</p>
 *
 * <ol>
 *   <li>Apply a small Gaussian blur to reduce local noise.</li>
 *   <li>Apply CLAHE to the Lab L channel to improve local contrast.</li>
 *   <li>Apply gray-world white balance in BGR space.</li>
 * </ol>
 *
 * <p>All methods expect OpenCV images in BGR format, which is OpenCV's default
 * color channel order.</p>
 */
public class TrayLightingNormalizer {

    /**
     * Applies the full tray lighting normalization pipeline.
     *
     * <p>The input image is first blurred slightly, then contrast-enhanced using
     * CLAHE in Lab color space, and finally white-balanced using a gray-world
     * channel scaling method.</p>
     *
     * @param bgr input image in BGR color format
     * @return normalized BGR image after blur, CLAHE, and white balance
     */
    public static Mat claheLabThenWhiteBalance(Mat bgr) {
        Mat gaussian = new Mat();

        // Gaussian blur first.
        Imgproc.GaussianBlur(bgr, gaussian, new Size(5, 5), 0);

        // CLAHE in LAB.
        Mat claheLab = applyClaheLab(gaussian);

        // Gray-world white balance after CLAHE.
        Mat whiteBalanced = whiteBalancedBgr(claheLab);

        return whiteBalanced;
    }

    /**
     * Applies CLAHE contrast enhancement to the L channel of a Lab image.
     *
     * <p>The image is converted from BGR to Lab. The Lab color space separates
     * brightness from color, so this method enhances only the L channel while
     * leaving the a and b color channels unchanged. The channels are then merged
     * and converted back to BGR.</p>
     *
     * @param bgr input image in BGR color format
     * @return BGR image after CLAHE has been applied to the Lab L channel
     */
    private static Mat applyClaheLab(Mat bgr) {
        Mat lab = new Mat();

        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab);

        List<Mat> channels = new ArrayList<>(3);
        Core.split(lab, channels);

        Mat l = channels.get(0);
        Mat a = channels.get(1);
        Mat b = channels.get(2);

        CLAHE clahe = Imgproc.createCLAHE(3.0, new Size(8, 8));

        Mat lClahe = new Mat();
        clahe.apply(l, lClahe);

        List<Mat> mergedLabChannels = new ArrayList<>(3);
        mergedLabChannels.add(lClahe);
        mergedLabChannels.add(a);
        mergedLabChannels.add(b);

        Mat mergedLab = new Mat();
        Core.merge(mergedLabChannels, mergedLab);

        Mat claheBgr = new Mat();
        Imgproc.cvtColor(mergedLab, claheBgr, Imgproc.COLOR_Lab2BGR);

        return claheBgr;
    }

    /**
     * Applies gray-world white balance to a BGR image.
     *
     * <p>The gray-world assumption treats the average color of the scene as neutral
     * gray. This method computes the mean value of each BGR channel, computes the
     * overall average channel mean, and scales each channel so its mean moves toward
     * that shared average.</p>
     *
     * <p>The image is temporarily converted to 32-bit float so the channel scaling
     * can be performed accurately, then converted back to 8-bit BGR output.</p>
     *
     * @param bgr input image in BGR color format
     * @return white-balanced BGR image
     */
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
}