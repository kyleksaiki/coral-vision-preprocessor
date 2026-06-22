import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Normalizes tray image lighting before downstream processing.
 *
 * <p>Pipeline:</p>
 * <ol>
 *   <li>Apply CLAHE to the Lab L channel to improve local contrast.</li>
 *   <li>Apply gray-world white balance in BGR space.</li>
 * </ol>
 *
 * <p>(The old Gaussian pre-blur was removed: the models resize the image by ~4x, which blurs
 * far more than a 5x5 Gaussian, so the pre-blur added nothing for the learned path.)</p>
 *
 * <p>All methods expect OpenCV images in BGR format.</p>
 */
public class TrayLightingNormalizer {

    /**
     * Applies CLAHE (Lab L channel) followed by gray-world white balance.
     *
     * @param bgr input image in BGR color format
     * @return normalized BGR image
     */
    public static Mat claheLabThenWhiteBalance(Mat bgr) {
        // CLAHE in LAB (no pre-blur).
        Mat claheLab = applyClaheLab(bgr);

        // Gray-world white balance after CLAHE.
        Mat whiteBalanced = whiteBalancedBgr(claheLab);

        claheLab.release();
        return whiteBalanced;
    }

    /**
     * Applies CLAHE contrast enhancement to the L channel of a Lab image.
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

        lab.release();
        l.release();
        a.release();
        b.release();
        lClahe.release();
        mergedLab.release();
        return claheBgr;
    }

    /**
     * Applies gray-world white balance to a BGR image.
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

        floatImg.release();
        b.release();
        g.release();
        r.release();
        balancedFloat.release();
        return whiteBalanced;
    }
}