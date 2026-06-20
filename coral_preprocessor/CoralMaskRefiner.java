import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Groups coral pixels into connected blobs, shrinks each blob down to a small
 * rounded core near the colony center, and measures the result.
 *
 * <p>Coral is marked with 'C' or 'c' in the label grid. The goal is NOT to cover
 * each colony; it is to leave one compact dot near the center of every colony and
 * nowhere else. A single input blob can span several colonies joined by the dark
 * mesh net, so shrinking is done per connected piece: as the thin necks erode
 * through, one blob falls apart into separate pieces and each piece is then
 * treated independently (so three small corals on a card become three dots).</p>
 *
 * <p>This class produces:</p>
 * <ul>
 *   <li>an id matrix: 0 = not coral, 1..N = the blob a pixel belongs to</li>
 *   <li>one {@link ComponentStat} per blob with its size, box, centroid, and shape</li>
 * </ul>
 */
public class CoralMaskRefiner {

    // ----------------------------------------------------------------------
    // Tuning knobs.
    // Sizes are in pixels on the reference image and are scaled automatically to
    // the actual image resolution (areas by the pixel-count ratio, lengths by its
    // square root), so the same numbers work on images that are not exactly 12 MP.
    // ----------------------------------------------------------------------

    /** Resolution the size constants below were measured on (~12 MP). */
    private static final int REFERENCE_WIDTH  = 4000;
    private static final int REFERENCE_HEIGHT = 3000;

    /**
     * Shrink a blob until its area is at or below this (px on the reference image).
     * Measured colony area is ~39,000..117,000 (median ~71,000); the default aims at
     * the median so each core ends up a centered dot rather than a full cover.
     */
    private static final double TARGET_AREA_PX = 71_000;

    /**
     * Shrink a blob until its longer bounding-box side is at or below this
     * (px on the reference image). Measured longer side is ~265..432 (median ~350).
     */
    private static final double TARGET_LONGER_SIDE_PX = 350;

    /**
     * Floor that protects small pieces: a blob at or below this area is frozen and
     * never eroded further, so small colonies survive as dots and nothing is eroded
     * away to nothing. Sits below the smallest real colony (~39,000). Raise it to
     * discard more small fragments; lower it to keep smaller pieces.
     */
    private static final double MIN_COMPONENT_AREA_PX = 20_000;

    /**
     * Minimum roundness required to stop shrinking a blob, as the minor/major axis
     * ratio from second-order central moments: 1.0 = perfect circle, lower = more
     * elongated. 0.55 accepts up to roughly a 1.8:1 ellipse.
     */
    private static final double MIN_ROUNDNESS = 0.55;

    /** Erosion radius removed from each blob edge per shrink step (px, reference image). */
    private static final int EROSION_RADIUS_PX = 5;

    /** Safety cap on shrink iterations. */
    private static final int MAX_SHRINK_ITERATIONS = 100;

    /**
     * Measurements for one connected coral blob.
     *
     * <p>The first group comes straight from OpenCV; the rest are derived from it.</p>
     */
    public static class ComponentStat {
        // --- straight from OpenCV ---
        public final int id;          // blob id (1..N)
        public final int area;        // number of coral pixels in the blob
        public final int left;        // bounding box: left edge x
        public final int top;         // bounding box: top edge y
        public final int width;       // bounding box width
        public final int height;      // bounding box height
        public final double centroidX; // center of mass x
        public final double centroidY; // center of mass y

        // --- derived ---
        public final int right;              // left + width - 1
        public final int bottom;             // top + height - 1
        public final long boundingBoxArea;   // width * height
        public final double extent;          // area / boundingBoxArea (0..1, how full the box is)
        public final double aspectRatio;     // longer side / shorter side (>= 1)
        public final double equivalentDiameter; // diameter of a circle with the same area

        ComponentStat(int id, int area, int left, int top, int width, int height,
                      double centroidX, double centroidY) {
            this.id = id;
            this.area = area;
            this.left = left;
            this.top = top;
            this.width = width;
            this.height = height;
            this.centroidX = centroidX;
            this.centroidY = centroidY;

            this.right = left + width - 1;
            this.bottom = top + height - 1;
            this.boundingBoxArea = (long) width * height;
            this.extent = (boundingBoxArea > 0) ? area / (double) boundingBoxArea : 0.0;
            this.aspectRatio = Math.max(width, height) / (double) Math.max(1, Math.min(width, height));
            this.equivalentDiameter = Math.sqrt(4.0 * area / Math.PI);
        }

        /** Column names for {@link #toCsv()}. */
        public static String csvHeader() {
            return "id,area,left,top,width,height,right,bottom,boundingBoxArea,"
                    + "centroidX,centroidY,extent,aspectRatio,equivalentDiameter";
        }

        /** This blob's stats as one CSV row (matches {@link #csvHeader()}). */
        public String toCsv() {
            return String.format("%d,%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%.4f,%.4f,%.3f",
                    id, area, left, top, width, height, right, bottom, boundingBoxArea,
                    centroidX, centroidY, extent, aspectRatio, equivalentDiameter);
        }

        @Override
        public String toString() {
            return String.format(
                    "id=%d area=%d bbox=(%d,%d %dx%d) centroid=(%.1f,%.1f) "
                            + "extent=%.3f aspect=%.2f eqDiam=%.1f",
                    id, area, left, top, width, height, centroidX, centroidY,
                    extent, aspectRatio, equivalentDiameter);
        }
    }

    /** Output of {@link #labelConnectedComponents}: the id matrix plus a stat per blob. */
    public static class CoralComponents {
        /** id per pixel: 0 = not coral, 1..N = blob id. */
        public final int[][] ids;
        /** one entry per blob, in id order (background excluded). */
        public final List<ComponentStat> stats;

        CoralComponents(int[][] ids, List<ComponentStat> stats) {
            this.ids = ids;
            this.stats = stats;
        }
    }

    /**
     * Shrinks each coral blob to a small rounded core (splitting blobs that pull
     * apart at thin necks), then labels and measures the resulting cores.
     *
     * @param labels the label grid ('C'/'c' = coral)
     * @return the id matrix and the per-blob stats, computed from the shrunk mask
     */
    public static CoralComponents labelConnectedComponents(char[][] labels) {
        int rows = labels.length;
        int cols = (rows == 0) ? 0 : labels[0].length;
        int[][] ids = new int[rows][cols];
        List<ComponentStat> stats = new ArrayList<>();
        if (rows == 0 || cols == 0) {
            return new CoralComponents(ids, stats);
        }

        // Scale the reference-image thresholds to this image's resolution.
        double areaScale = (double) rows * cols / ((double) REFERENCE_WIDTH * REFERENCE_HEIGHT);
        double lengthScale = Math.sqrt(areaScale);

        // coral -> white pixels in a binary mask
        byte[] maskBuf = new byte[rows * cols];
        int i = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++, i++) {
                char ch = labels[r][c];
                if (ch == 'C' || ch == 'c') {
                    maskBuf[i] = (byte) 255;
                }
            }
        }
        Mat mask = new Mat(rows, cols, CvType.CV_8UC1);
        mask.put(0, 0, maskBuf);

        // shrink each blob to a centered, rounded core
        shrinkToCores(mask, maskBuf, rows, cols, areaScale, lengthScale);

        // label + measure the final cores
        measure(mask, rows, cols, ids, stats);

        mask.release();
        return new CoralComponents(ids, stats);
    }

    /**
     * Erodes oversized / elongated blobs in {@code mask} one step at a time until
     * every remaining piece is small enough and round enough, working per connected
     * piece rather than over the whole mask at once.
     *
     * <p>Each iteration:</p>
     * <ol>
     *   <li>re-runs connected components (a piece that just split is now two pieces);</li>
     *   <li>measures every piece and decides freeze vs. erode;</li>
     *   <li>copies frozen pieces through unchanged and erodes only the rest.</li>
     * </ol>
     *
     * <p>A piece is frozen (kept as-is) when it is at or below the size target and
     * round enough, OR when it is at or below {@link #MIN_COMPONENT_AREA_PX} so small
     * pieces are never eroded to nothing.</p>
     */
    private static void shrinkToCores(Mat mask, byte[] maskBuf, int rows, int cols,
                                      double areaScale, double lengthScale) {
        double maxArea     = TARGET_AREA_PX * areaScale;
        double minArea     = MIN_COMPONENT_AREA_PX * areaScale;
        double maxLonger   = TARGET_LONGER_SIDE_PX * lengthScale;
        int radius = Math.max(1, (int) Math.round(EROSION_RADIUS_PX * lengthScale));
        Mat se = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE, new Size(2 * radius + 1, 2 * radius + 1));

        int total = rows * cols;
        int[] lab = new int[total];          // component id per pixel
        byte[] frozenBuf = new byte[total];  // pixels kept unchanged this step
        byte[] erodeBuf = new byte[total];   // pixels to erode this step
        Mat labelMat = new Mat();
        Mat erodeSrc = new Mat(rows, cols, CvType.CV_8UC1);
        Mat erodeDst = new Mat();

        for (int iter = 0; iter < MAX_SHRINK_ITERATIONS; iter++) {
            int count = Imgproc.connectedComponents(mask, labelMat, 8, CvType.CV_32S);
            if (count <= 1) {
                break; // nothing left but background
            }
            labelMat.get(0, 0, lab);

            // Per-component accumulators (index 0 is background).
            double[] area = new double[count];
            int[] minX = new int[count];
            int[] maxX = new int[count];
            int[] minY = new int[count];
            int[] maxY = new int[count];
            double[] sx = new double[count];
            double[] sy = new double[count];
            double[] sxx = new double[count];
            double[] syy = new double[count];
            double[] sxy = new double[count];
            Arrays.fill(minX, Integer.MAX_VALUE);
            Arrays.fill(minY, Integer.MAX_VALUE);
            Arrays.fill(maxX, -1);
            Arrays.fill(maxY, -1);

            int p = 0;
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++, p++) {
                    int k = lab[p];
                    if (k == 0) {
                        continue;
                    }
                    area[k]++;
                    if (c < minX[k]) minX[k] = c;
                    if (c > maxX[k]) maxX[k] = c;
                    if (r < minY[k]) minY[k] = r;
                    if (r > maxY[k]) maxY[k] = r;
                    sx[k] += c;
                    sy[k] += r;
                    sxx[k] += (double) c * c;
                    syy[k] += (double) r * r;
                    sxy[k] += (double) c * r;
                }
            }

            // Decide, per component, whether it still needs eroding.
            boolean[] erodeComp = new boolean[count];
            boolean anyErode = false;
            for (int k = 1; k < count; k++) {
                int w = maxX[k] - minX[k] + 1;
                int h = maxY[k] - minY[k] + 1;
                int longer = Math.max(w, h);
                double roundness = minorMajorAxisRatio(area[k], sx[k], sy[k], sxx[k], syy[k], sxy[k]);

                boolean smallEnough = area[k] <= maxArea && longer <= maxLonger;
                boolean tooSmallToErode = area[k] <= minArea;
                boolean keep = tooSmallToErode || (smallEnough && roundness >= MIN_ROUNDNESS);

                erodeComp[k] = !keep;
                anyErode |= erodeComp[k];
            }
            if (!anyErode) {
                break; // every piece is a core; done
            }

            // Split the current mask: frozen pieces stay, the rest get eroded.
            Arrays.fill(frozenBuf, (byte) 0);
            Arrays.fill(erodeBuf, (byte) 0);
            for (p = 0; p < total; p++) {
                int k = lab[p];
                if (k == 0) {
                    continue;
                }
                if (erodeComp[k]) {
                    erodeBuf[p] = (byte) 255;
                } else {
                    frozenBuf[p] = (byte) 255;
                }
            }

            // Erode only the to-shrink pixels, then union with the frozen pixels.
            // (Erosion is local, so eroding the separate pieces together gives the
            // same result as eroding each one on its own.)
            erodeSrc.put(0, 0, erodeBuf);
            Imgproc.erode(erodeSrc, erodeDst, se);
            erodeDst.get(0, 0, erodeBuf);

            for (p = 0; p < total; p++) {
                maskBuf[p] = (frozenBuf[p] != 0 || erodeBuf[p] != 0) ? (byte) 255 : 0;
            }
            mask.put(0, 0, maskBuf);
        }

        se.release();
        labelMat.release();
        erodeSrc.release();
        erodeDst.release();
    }

    /**
     * Minor/major axis ratio of the best-fit ellipse from second-order central
     * moments. Returns a value in (0, 1]: 1.0 = circle, smaller = more elongated.
     * Degenerate or near-point blobs are treated as round.
     */
    private static double minorMajorAxisRatio(double area, double sx, double sy,
                                              double sxx, double syy, double sxy) {
        if (area <= 0) {
            return 1.0;
        }
        double meanX = sx / area;
        double meanY = sy / area;
        double u20 = sxx / area - meanX * meanX; // variance in x
        double u02 = syy / area - meanY * meanY; // variance in y
        double u11 = sxy / area - meanX * meanY; // covariance

        double common = Math.sqrt(Math.max(0.0, (u20 - u02) * (u20 - u02) + 4.0 * u11 * u11));
        double major = (u20 + u02 + common) / 2.0; // larger eigenvalue
        double minor = (u20 + u02 - common) / 2.0; // smaller eigenvalue
        if (major <= 1e-9) {
            return 1.0;
        }
        if (minor < 0) {
            minor = 0;
        }
        return Math.sqrt(minor / major);
    }

    /**
     * Labels the final mask (8-connected) and fills {@code ids} and {@code stats}.
     * Identical measurement to before, just run on the shrunk mask.
     */
    private static void measure(Mat mask, int rows, int cols, int[][] ids, List<ComponentStat> stats) {
        Mat labelMat = new Mat();
        Mat statsMat = new Mat();
        Mat centroidsMat = new Mat();
        int count = Imgproc.connectedComponentsWithStats(
                mask, labelMat, statsMat, centroidsMat, 8, CvType.CV_32S);

        // per-pixel ids
        int[] flat = new int[rows * cols];
        labelMat.get(0, 0, flat);
        int i = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++, i++) {
                ids[r][c] = flat[i];
            }
        }

        // per-blob stats (row 0 is the background, so start at id 1)
        int[] st = new int[count * 5];
        double[] ct = new double[count * 2];
        statsMat.get(0, 0, st);
        centroidsMat.get(0, 0, ct);
        for (int id = 1; id < count; id++) {
            int b = id * 5;
            stats.add(new ComponentStat(
                    id,
                    st[b + Imgproc.CC_STAT_AREA],
                    st[b + Imgproc.CC_STAT_LEFT],
                    st[b + Imgproc.CC_STAT_TOP],
                    st[b + Imgproc.CC_STAT_WIDTH],
                    st[b + Imgproc.CC_STAT_HEIGHT],
                    ct[id * 2],
                    ct[id * 2 + 1]));
        }

        labelMat.release();
        statsMat.release();
        centroidsMat.release();
    }
}