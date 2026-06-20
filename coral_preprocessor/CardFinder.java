import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDouble;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

/**
 * Finds the white ID cards in the normalized (CLAHE + white-balanced) tray image and
 * returns a card / not-card mask. No rectangle fitting.
 *
 * <p><b>What the threshold actually contains.</b> "Bright" is three different things and
 * brightness alone cannot separate them: cards (big solid blobs with dark coral holes and
 * coral-edge bites), mesh (rows of small bright dots / thin dotted chains), and background
 * sand (large bright blotches that run off the image edges). Two reliable facts: cards are
 * interior (background reaches the border), and a card column has a constant width even
 * though its area swings wildly (a column of stacked cards usually merges into one tall
 * strip because the thin mesh line between them dissolves under the median).</p>
 *
 * <p><b>The two ideas that make this robust.</b></p>
 * <ol>
 *   <li><b>Convex-hull coral fill.</b> A card is a convex rounded rectangle, so every
 *       concavity in its blob is coral &mdash; whether that coral is an enclosed hole or an
 *       open bite at the edge. Taking the convex hull of each blob therefore fills the coral
 *       in one operation (interior holes <i>and</i> edge bites), bounded by the card's own
 *       pixels so it cannot run away. This replaces the old close + fill-holes + open dance
 *       and, crucially, creates no thin strands &mdash; so the "fill the coral" and "no thin
 *       bridges" goals stop fighting each other.</li>
 *   <li><b>Smooth-and-blue rejection.</b> After filling, a few non-cards can still survive as
 *       big interior blobs: PVC-pipe reflections whose dark algae top floats free of the
 *       border, and pale sand patches. They are physically different from cards: a card
 *       carries dark coral on bright plastic (high internal brightness variance) and is
 *       neutral/warm, while pipe and sand are smooth and bluish. A blob is dropped only if it
 *       is BOTH smooth (low brightness std) AND blue (mean R below mean B). Real cards fail
 *       that test on the variance alone, so none are ever dropped.</li>
 * </ol>
 *
 * <p><b>Pipeline.</b></p>
 * <ol>
 *   <li>{@link #preprocess(Mat)} &mdash; strong median blur, Otsu (biased up) to a binary mask.</li>
 *   <li><b>open</b> hard to break strands (card&harr;background necks, dotted mesh chains) and
 *       remove mesh dots.</li>
 *   <li>drop <b>border-touching</b> blobs at ANY size (background, edge-pipe, labels). Cards
 *       are interior, so this no longer eats a clean column dragged out by a strand.</li>
 *   <li>moderate <b>close</b> to reconnect coral-split card slivers &mdash; safe here because
 *       the border background is already gone, so nothing re-glues a card to it.</li>
 *   <li><b>area floor</b> to drop specks / residual mesh fragments.</li>
 *   <li><b>convex-hull fill</b> per blob: fills the coral (holes and edge bites).</li>
 *   <li>drop <b>smooth+blue</b> blobs (pipe reflections, sand).</li>
 *   <li>keep <b>card-width</b> blobs (distance-transform thickness vs the calibrated width),
 *       which trims any thin warm leftover such as the ruler.</li>
 * </ol>
 *
 * <p>Pixel-length knobs are for the 4000x3000 reference image and scaled to the actual
 * resolution (areas by the pixel-count ratio, lengths by its square root). The intensity
 * knobs (bias, smoothness, colour) are in intensity units and are not scaled.</p>
 */
public class CardFinder {

    /** Resolution the pixel-length constants were chosen for (~12 MP). */
    private static final int REFERENCE_WIDTH  = 4000;
    private static final int REFERENCE_HEIGHT = 3000;

    /** STRONG median kernel (px, ref) for the regionizing preprocess. */
    private static final int MEDIAN_FULL_PX = 45;

    /** Bias added to the Otsu cutoff for the binary bright mask. Raise if too much white
     *  survives, lower (toward negative) if dim cards drop out. */
    private static final double BRIGHT_BIAS = 20;

    /**
     * Opening radius (px, ref) that breaks the thin strands gluing a card column to the
     * border-touching background (and removes mesh dots), BEFORE the border filter. Must
     * exceed half the strand/neck width. Raise it if clean columns are still dropped (their
     * strand is not being cut); lower it if narrow cards vanish.
     */
    private static final int STRAND_OPEN_PX = 22;

    /** Blobs whose bounding box comes within this of any edge are border blobs (not cards). */
    private static final int BORDER_MARGIN_PX = 8;

    /**
     * Closing radius (px, ref) applied AFTER the border filter to reconnect card slivers that
     * coral split apart, and to merge a stacked column into one blob. Safe at this stage
     * because the border background is already removed, so it cannot re-bridge a card to the
     * background. Larger fuses more (a whole column becomes one tall blob, which is fine for a
     * card/not-card mask); smaller keeps cards more separate but can drop coral-shattered ones.
     */
    private static final int RECONNECT_CLOSE_PX = 24;

    /** Area floor for a blob to be kept, as a fraction of the whole image. */
    private static final double MIN_BLOB_AREA_FRAC = 0.003;

    /**
     * Smooth-and-blue rejection. A filled blob is discarded only if its internal brightness
     * standard deviation is below {@link #SMOOTH_STD_MAX} AND its (mean red - mean blue) is
     * below {@link #BLUE_MAX}. Both conditions must hold, so a textured card (coral on plastic)
     * is never dropped on colour, and a neutral/warm card is never dropped on smoothness.
     * Measured over the FILLED footprint, which is what gives a card its high variance.
     */
    private static final double SMOOTH_STD_MAX = 35.0;
    private static final double BLUE_MAX = 0.0;

    /** Card width is calibrated from this upper percentile of blob thickness (the widest
     *  blobs anchor it). 0..1. */
    private static final double WIDTH_PERCENTILE = 0.70;

    /** Keep a blob if its thickness is at least this fraction of the calibrated card width
     *  (generous, so a coral-thinned part of a card still passes). */
    private static final double THICK_TOL_LOW = 0.35;

    /**
     * Strong-median, then binary bright mask. Saved as the threshold debug stage.
     *
     * @param bgr normalized BGR image
     * @return single-channel (CV_8UC1) binary mask (255 = bright), or empty Mat for empty input
     */
    public static Mat preprocess(Mat bgr) {
        Mat gray = new Mat();
        if (bgr == null || bgr.empty()) {
            return gray;
        }
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.medianBlur(gray, gray, oddFromRef(MEDIAN_FULL_PX, bgr));

        double otsu = Imgproc.threshold(gray, gray, 0, 255, Imgproc.THRESH_BINARY | Imgproc.THRESH_OTSU);
        double cutoff = Math.min(255.0, otsu + BRIGHT_BIAS);
        Imgproc.threshold(gray, gray, cutoff, 255, Imgproc.THRESH_BINARY);
        return gray;
    }

    /** Convenience: preprocess internally, find cards, release the temp. */
    public static Mat findCards(Mat bgr) {
        Mat bin = preprocess(bgr);
        Mat cards = findCards(bgr, bin);
        bin.release();
        return cards;
    }

    /**
     * Card / not-card grid from the binary bright mask produced by {@link #preprocess(Mat)}.
     *
     * @param bgr    normalized BGR image (gives the output size and the colour cues)
     * @param binary output of {@link #preprocess(Mat)} (caller owns it; not modified)
     * @return single-channel (CV_8UC1) card mask: 255 = card, 0 = not a card
     */
    public static Mat findCards(Mat bgr, Mat binary) {
        if (bgr == null || bgr.empty()) {
            return new Mat();
        }
        if (binary == null || binary.empty()) {
            return Mat.zeros(bgr.size(), CvType.CV_8UC1);
        }

        // 1) Break thin strands (card<->background necks, dotted mesh chains) and remove dots.
        Mat opened = binary.clone();
        morph(opened, Imgproc.MORPH_OPEN, ellipse(STRAND_OPEN_PX, bgr));

        // 2) Drop border-touching blobs at ANY size (background, edge-pipe, labels). Cards are
        //    now isolated, so a clean column is no longer dragged to the border by a strand.
        Mat interior = dropBorder(opened, bgr);
        opened.release();

        // 3) Reconnect coral-split card slivers (safe: the border background is already gone).
        morph(interior, Imgproc.MORPH_CLOSE, ellipse(RECONNECT_CLOSE_PX, bgr));

        // 4) Area floor: drop specks / residual mesh fragments.
        Mat big = areaFloor(interior, bgr);
        interior.release();

        // 5) Fill the coral: per-blob convex hull seals interior holes AND open edge bites.
        Mat filled = hullFill(big);
        big.release();

        // 6) Drop smooth+blue interior false positives (pipe reflections, sand patches).
        Mat deFalsePos = dropSmoothBlue(filled, bgr);
        filled.release();

        // 7) Keep card-width blobs; trims any thin warm leftover (e.g. the ruler).
        Mat cardMask = keepCardWidth(deFalsePos, bgr);
        deFalsePos.release();
        return cardMask; // CV_8UC1, 255 = card
    }

    /** Keep only blobs whose bounding box does not reach the border (any size). */
    private static Mat dropBorder(Mat mask, Mat bgr) {
        int rows = mask.rows();
        int cols = mask.cols();
        int margin = Math.max(1, (int) Math.round(BORDER_MARGIN_PX * lengthScale(bgr)));

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int n = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);

        Mat out = Mat.zeros(mask.size(), CvType.CV_8UC1);
        Mat comp = new Mat();
        for (int i = 1; i < n; i++) { // 0 = background label
            int left = (int) stats.get(i, Imgproc.CC_STAT_LEFT)[0];
            int top = (int) stats.get(i, Imgproc.CC_STAT_TOP)[0];
            int w = (int) stats.get(i, Imgproc.CC_STAT_WIDTH)[0];
            int h = (int) stats.get(i, Imgproc.CC_STAT_HEIGHT)[0];
            if (touchesBorder(left, top, w, h, cols, rows, margin)) {
                continue;
            }
            Core.compare(labels, new Scalar(i), comp, Core.CMP_EQ);
            Core.bitwise_or(out, comp, out);
        }
        comp.release();
        labels.release();
        stats.release();
        centroids.release();
        return out;
    }

    /** Keep only blobs at or above the area floor. */
    private static Mat areaFloor(Mat mask, Mat bgr) {
        int rows = mask.rows();
        int cols = mask.cols();
        double minArea = MIN_BLOB_AREA_FRAC * rows * cols;

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int n = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);

        Mat out = Mat.zeros(mask.size(), CvType.CV_8UC1);
        Mat comp = new Mat();
        for (int i = 1; i < n; i++) {
            if (stats.get(i, Imgproc.CC_STAT_AREA)[0] < minArea) {
                continue;
            }
            Core.compare(labels, new Scalar(i), comp, Core.CMP_EQ);
            Core.bitwise_or(out, comp, out);
        }
        comp.release();
        labels.release();
        stats.release();
        centroids.release();
        return out;
    }

    /** Fill each blob to its convex hull (fills interior coral holes AND open edge bites). */
    private static Mat hullFill(Mat mask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        Mat out = Mat.zeros(mask.size(), CvType.CV_8UC1);
        for (MatOfPoint c : contours) {
            MatOfInt hullIdx = new MatOfInt();
            Imgproc.convexHull(c, hullIdx);
            Point[] pts = c.toArray();
            int[] idx = hullIdx.toArray();
            Point[] hullPts = new Point[idx.length];
            for (int j = 0; j < idx.length; j++) {
                hullPts[j] = pts[idx[j]];
            }
            MatOfPoint hull = new MatOfPoint(hullPts);
            Imgproc.drawContours(out, Arrays.asList(hull), -1, new Scalar(255), -1); // -1 = filled
            hull.release();
            hullIdx.release();
            c.release();
        }
        hierarchy.release();
        return out;
    }

    /**
     * Drop blobs that are BOTH smooth (internal brightness std &lt; {@link #SMOOTH_STD_MAX})
     * AND blue (mean red - mean blue &lt; {@link #BLUE_MAX}). Pipe reflections and sand are
     * smooth and bluish; a card carries coral texture (high variance) and a neutral/warm cast,
     * so it fails the conjunction and is always kept. Brightness "value" is the per-pixel max
     * over the three channels.
     */
    private static Mat dropSmoothBlue(Mat mask, Mat bgr) {
        List<Mat> chans = new ArrayList<>();
        Core.split(bgr, chans); // 0 = B, 1 = G, 2 = R
        Mat blue = chans.get(0);
        Mat green = chans.get(1);
        Mat red = chans.get(2);

        Mat value = new Mat();
        Core.max(blue, green, value);
        Core.max(value, red, value);

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int n = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);

        Mat out = Mat.zeros(mask.size(), CvType.CV_8UC1);
        Mat comp = new Mat();
        for (int i = 1; i < n; i++) {
            Core.compare(labels, new Scalar(i), comp, Core.CMP_EQ);

            MatOfDouble mean = new MatOfDouble();
            MatOfDouble std = new MatOfDouble();
            Core.meanStdDev(value, mean, std, comp);
            double valStd = std.toArray()[0];
            mean.release();
            std.release();

            double meanR = Core.mean(red, comp).val[0];
            double meanB = Core.mean(blue, comp).val[0];

            boolean smooth = valStd < SMOOTH_STD_MAX;
            boolean bluish = (meanR - meanB) < BLUE_MAX;
            if (smooth && bluish) {
                continue; // drop: smooth & blue -> not a card
            }
            Core.bitwise_or(out, comp, out);
        }
        comp.release();
        labels.release();
        stats.release();
        centroids.release();
        value.release();
        for (Mat c : chans) {
            c.release();
        }
        return out;
    }

    /** Keep blobs that are card-width: thickness (= 2 x max distance-transform) within a
     *  generous band of the calibrated card width, and above the area floor. */
    private static Mat keepCardWidth(Mat mask, Mat bgr) {
        int rows = mask.rows();
        int cols = mask.cols();
        double minArea = MIN_BLOB_AREA_FRAC * rows * cols;

        Mat dist = new Mat();
        Imgproc.distanceTransform(mask, dist, Imgproc.DIST_L2, 3);
        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int n = Imgproc.connectedComponentsWithStats(mask, labels, stats, centroids);

        double[] thickness = new double[n];
        List<Double> widths = new ArrayList<>();
        Mat blob = new Mat();
        for (int i = 1; i < n; i++) {
            if (stats.get(i, Imgproc.CC_STAT_AREA)[0] < minArea) {
                continue;
            }
            Core.compare(labels, new Scalar(i), blob, Core.CMP_EQ);
            Core.MinMaxLocResult mm = Core.minMaxLoc(dist, blob);
            thickness[i] = 2.0 * mm.maxVal;
            widths.add(thickness[i]);
        }
        blob.release();
        dist.release();

        Mat out = Mat.zeros(mask.size(), CvType.CV_8UC1);
        double cardWidth = percentile(widths, WIDTH_PERCENTILE);
        if (cardWidth > 0) {
            double minThick = THICK_TOL_LOW * cardWidth;
            Mat tmp = new Mat();
            for (int i = 1; i < n; i++) {
                if (stats.get(i, Imgproc.CC_STAT_AREA)[0] >= minArea && thickness[i] >= minThick) {
                    Core.compare(labels, new Scalar(i), tmp, Core.CMP_EQ);
                    Core.bitwise_or(out, tmp, out);
                }
            }
            tmp.release();
        }
        labels.release();
        stats.release();
        centroids.release();
        return out;
    }

    /** True if a blob's bounding box comes within {@code margin} of any image edge. */
    private static boolean touchesBorder(int left, int top, int w, int h,
                                         int cols, int rows, int margin) {
        return left <= margin
                || top <= margin
                || left + w >= cols - margin
                || top + h >= rows - margin;
    }

    /** Elliptical kernel of the given reference radius, scaled to this image. */
    private static Mat ellipse(int refRadiusPx, Mat img) {
        int r = Math.max(1, (int) Math.round(refRadiusPx * lengthScale(img)));
        return Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE, new Size(2 * r + 1, 2 * r + 1));
    }

    /** Apply a morphology op with the given (caller-owned) kernel, which is released here. */
    private static void morph(Mat mask, int op, Mat kernel) {
        Imgproc.morphologyEx(mask, mask, op, kernel);
        kernel.release();
    }

    /** Value at the given percentile (0..1) of a list (returns 0 for empty). */
    private static double percentile(List<Double> values, double p) {
        if (values.isEmpty()) {
            return 0;
        }
        List<Double> s = new ArrayList<>(values);
        Collections.sort(s);
        int idx = (int) Math.round(p * (s.size() - 1));
        return s.get(Math.max(0, Math.min(s.size() - 1, idx)));
    }

    /** Reference-px length scaled to this image and forced to an odd kernel >= 3. */
    private static int oddFromRef(int refPx, Mat img) {
        return odd(Math.round(refPx * lengthScale(img)));
    }

    /** Nearest odd integer >= 3 (median/kernel sizes must be odd). */
    private static int odd(long v) {
        int k = (int) Math.max(3, v);
        return (k % 2 == 0) ? k + 1 : k;
    }

    /** sqrt of the pixel-count ratio vs the reference image, for scaling lengths. */
    private static double lengthScale(Mat img) {
        double areaScale = (double) img.rows() * img.cols()
                / ((double) REFERENCE_WIDTH * REFERENCE_HEIGHT);
        return Math.sqrt(areaScale);
    }

    // ------------------------------------------------------------------
    // Visualization
    // ------------------------------------------------------------------

    /**
     * Renders the card mask as an instance-segmentation overlay for humans to look at.
     *
     * <p>Every connected card blob is treated as one "instance" and gets:</p>
     * <ul>
     *   <li>a translucent fill in its own distinct color (so touching cards read as separate),</li>
     *   <li>a bright white outline,</li>
     *   <li>an index number drawn at its center.</li>
     * </ul>
     *
     * <p>Non-card pixels are dimmed slightly so the segmented cards pop. This image is for
     * eyeballing results and for screenshots; the machine-readable result is the raw
     * {@code cardMask} (255 = card).</p>
     *
     * @param bgr      normalized BGR image to draw on top of
     * @param cardMask CV_8UC1 card mask from {@link #findCards(Mat, Mat)} (255 = card)
     * @return a BGR visualization image
     */
    public static Mat renderCardOverlay(Mat bgr, Mat cardMask) {
        Mat out = bgr.clone();
        if (cardMask == null || cardMask.empty()) {
            return out;
        }

        int rows = out.rows();
        int cols = out.cols();
        int ch = out.channels();

        // Label each separate card blob 1..n-1 (0 = background).
        Mat labelsMat = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int n = Imgproc.connectedComponentsWithStats(cardMask, labelsMat, stats, centroids);

        // Pull pixels and labels into flat arrays for one fast pass.
        byte[] pix = new byte[(int) (out.total() * ch)];
        out.get(0, 0, pix);
        int[] lab = new int[(int) labelsMat.total()];
        labelsMat.get(0, 0, lab);

        // One distinct color per card instance.
        int[][] palette = new int[Math.max(1, n)][];
        for (int id = 1; id < n; id++) {
            palette[id] = componentColor(id); // BGR, 0..255
        }

        final double alpha = 0.45;          // how strong the colored fill is on cards
        final double dimNonCard = 0.70;     // darken everything that is not a card

        int i = 0; // index into pix (steps by ch)
        int p = 0; // index into lab (steps by 1)
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++, i += ch, p++) {
                int id = lab[p];
                if (id >= 1) {
                    int[] color = palette[id];
                    for (int k = 0; k < 3; k++) {
                        int orig = pix[i + k] & 0xFF;
                        int blended = (int) Math.round(alpha * color[k] + (1.0 - alpha) * orig);
                        pix[i + k] = (byte) Math.max(0, Math.min(255, blended));
                    }
                } else {
                    for (int k = 0; k < 3; k++) {
                        int orig = pix[i + k] & 0xFF;
                        pix[i + k] = (byte) Math.max(0, Math.min(255, (int) Math.round(orig * dimNonCard)));
                    }
                }
            }
        }
        out.put(0, 0, pix);

        // Bright white outline around every card.
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(cardMask, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        int outlineThick = Math.max(2, (int) Math.round(3 * lengthScale(bgr)));
        Imgproc.drawContours(out, contours, -1, new Scalar(255, 255, 255), outlineThick);
        for (MatOfPoint contour : contours) {
            contour.release();
        }
        hierarchy.release();

        // Index number at each card's centroid (black halo + white text so it reads on any color).
        double fontScale = Math.max(1.0, 2.5 * lengthScale(bgr));
        int textThick = Math.max(2, (int) Math.round(3 * lengthScale(bgr)));
        for (int id = 1; id < n; id++) {
            double cx = centroids.get(id, 0)[0];
            double cy = centroids.get(id, 1)[0];
            String text = String.valueOf(id);
            Point at = new Point(cx, cy);
            Imgproc.putText(out, text, at, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                    new Scalar(0, 0, 0), textThick + 3);
            Imgproc.putText(out, text, at, Imgproc.FONT_HERSHEY_SIMPLEX, fontScale,
                    new Scalar(255, 255, 255), textThick);
        }

        labelsMat.release();
        stats.release();
        centroids.release();
        return out;
    }

    /** Distinct BGR color per instance id, spread around the hue wheel by the golden angle. */
    private static int[] componentColor(int id) {
        double hue = ((id * 0.61803398875) % 1.0) * 360.0;
        double cc = 0.85;
        double x = cc * (1 - Math.abs(((hue / 60.0) % 2) - 1));
        double m = 1.0 - cc;
        double r;
        double g;
        double b;
        if (hue < 60)       { r = cc; g = x;  b = 0;  }
        else if (hue < 120) { r = x;  g = cc; b = 0;  }
        else if (hue < 180) { r = 0;  g = cc; b = x;  }
        else if (hue < 240) { r = 0;  g = x;  b = cc; }
        else if (hue < 300) { r = x;  g = 0;  b = cc; }
        else                { r = cc; g = 0;  b = x;  }
        return new int[] {to255(b + m), to255(g + m), to255(r + m)}; // BGR
    }

    /** Scales a 0..1 value to a clamped 0..255 int. */
    private static int to255(double v) {
        return Math.max(0, Math.min(255, (int) Math.round(v * 255)));
    }
}