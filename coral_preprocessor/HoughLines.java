import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

/**
 * Hough line overlay: draws the HoughLinesP segments found in a Canny edge map
 * over a copy of the normalized (CLAHE + white-balanced) tray image.
 *
 * <p>Pixel-length knobs are expressed for the 4000x3000 reference image and scaled
 * to the actual resolution.</p>
 */
public class HoughLines {

    /** Resolution the pixel-length constants were chosen for (~12 MP). */
    private static final int REFERENCE_WIDTH  = 4000;
    private static final int REFERENCE_HEIGHT = 3000;

    /** HoughLinesP vote threshold (px on the reference image; scaled). */
    private static final int HOUGH_VOTES = 100;
    /** HoughLinesP minimum segment length (px on the reference image; scaled). */
    private static final double HOUGH_MIN_LINE_LENGTH_PX = 120;
    /** HoughLinesP maximum gap to bridge within a segment (px, reference; scaled). */
    private static final double HOUGH_MAX_LINE_GAP_PX = 20;

    /** Thickness of drawn lines (px on the reference image; scaled). */
    private static final int LINE_THICKNESS_PX = 2;
    /** Color of drawn lines (BGR). */
    private static final Scalar LINE_COLOR = new Scalar(0, 255, 0);

    /**
     * @param bgr   normalized BGR image to draw on
     * @param edges Canny edge map from {@link CannyEdges#detect(Mat)}
     * @return a BGR copy with detected line segments drawn
     */
    public static Mat overlay(Mat bgr, Mat edges) {
        double lengthScale = lengthScale(bgr);
        int votes = Math.max(1, (int) Math.round(HOUGH_VOTES * lengthScale));
        double minLen = HOUGH_MIN_LINE_LENGTH_PX * lengthScale;
        double maxGap = HOUGH_MAX_LINE_GAP_PX * lengthScale;
        int thickness = Math.max(1, (int) Math.round(LINE_THICKNESS_PX * lengthScale));

        Mat lines = new Mat();
        Imgproc.HoughLinesP(edges, lines, 1, Math.PI / 180.0, votes, minLen, maxGap);

        Mat out = bgr.clone();
        for (int i = 0; i < lines.rows(); i++) {
            double[] l = lines.get(i, 0);
            Imgproc.line(out, new Point(l[0], l[1]), new Point(l[2], l[3]), LINE_COLOR, thickness);
        }
        lines.release();
        return out;
    }

    /** sqrt of the pixel-count ratio vs the reference image, for scaling lengths. */
    private static double lengthScale(Mat img) {
        double areaScale = (double) img.rows() * img.cols()
                / ((double) REFERENCE_WIDTH * REFERENCE_HEIGHT);
        return Math.sqrt(areaScale);
    }
}