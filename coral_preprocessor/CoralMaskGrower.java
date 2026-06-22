import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.Arrays;

/**
 * Edge-growth for the coral mask.
 *
 * <p>The coral model finds essentially every colony, but it sometimes trims a thin rim of real
 * coral right at the boundary. This class nudges the mask outward by a few pixels &mdash; but
 * ONLY into pixels whose color is similar to the coral they're touching &mdash; so it recovers
 * that missed edge without ballooning into algae, silt, or bare plastic.</p>
 *
 * <p>It does not touch the model or any other stage; it is a pure post-process on the model's
 * binary mask. Everything you'd want to adjust is a single constant in the TUNING block below.</p>
 *
 * <p><b>How it works.</b> From the edge of each coral region it grows outward in rings
 * (breadth-first). A neighboring pixel is absorbed when both are true:</p>
 * <ol>
 *   <li>it is within {@link #MAX_GROW_RADIUS_PX} of the original edge &mdash; this is the
 *       "only extend out a bit" limit; and</li>
 *   <li>its Lab color is within {@link #COLOR_TOLERANCE} of that coral region's average color
 *       &mdash; this is the "similar to the pixels in the mask" test.</li>
 * </ol>
 * <p>Growth is also gated to the card mask, so it can never leak off a card.</p>
 *
 * <p><b>Usage:</b></p>
 * <pre>
 *   Mat grown = CoralMaskGrower.grow(claheBgr, coralMask, cardMask); // CV_8UC1, 255 = coral
 * </pre>
 */
public final class CoralMaskGrower {

    // ===================== TUNING — the only things you should need to touch =====================

    /**
     * How many pixels the mask may extend past its original edge.
     * Bigger = recovers more edge coral, but risks bleeding into fouling/plastic.
     * Reasonable range to try: 4–12. (Pixels at the full tray resolution; see
     * {@link #SCALE_RADIUS_WITH_RESOLUTION}.)
     */
    public static final int MAX_GROW_RADIUS_PX = 25;

    /**
     * How close a neighbor's color must be to the coral to get absorbed, as a Euclidean distance
     * in OpenCV 8-bit Lab space (L, a, b each on a 0–255 scale).
     * Smaller = stricter (adds fewer pixels); larger = looser (adds more).
     * Reasonable range to try: 8–24.
     */
    public static final double COLOR_TOLERANCE = 20.0;

    /** Never grow onto a pixel the card model did not call "card". Leave this true. */
    public static final boolean GATE_TO_CARD = true;

    /**
     * Keep the reach consistent across image sizes by scaling {@link #MAX_GROW_RADIUS_PX} with
     * resolution (the constant is calibrated for ~12 MP / 4000×3000). Set false to use the raw
     * pixel value regardless of resolution.
     */
    public static final boolean SCALE_RADIUS_WITH_RESOLUTION = true;

    /** Resolution {@link #MAX_GROW_RADIUS_PX} was chosen for; used only when scaling is on. */
    private static final double REFERENCE_PIXELS = 4000.0 * 3000.0;

    // =============================================================================================

    private CoralMaskGrower() {
    }

    /**
     * Returns a NEW CV_8UC1 mask (255 = coral) equal to {@code coralMask} grown outward into
     * similar-colored neighbors, bounded by radius, color tolerance, and the card mask. The
     * input masks are not modified.
     *
     * @param claheBgr  the normalized BGR image the mask lives on (used to read pixel color)
     * @param coralMask CV_8UC1 model coral mask (255 = coral), at {@code claheBgr}'s resolution
     * @param cardMask  CV_8UC1 card mask (255 = card); may be {@code null} to skip card gating
     * @return a new CV_8UC1 grown mask (caller owns it)
     */
    public static Mat grow(Mat claheBgr, Mat coralMask, Mat cardMask) {
        final int rows = coralMask.rows();
        final int cols = coralMask.cols();
        final int nPix = rows * cols;

        final int maxRadius = scaledRadius(rows, cols);

        // --- read pixel color once, in Lab (good for "similar color") ---
        Mat lab = new Mat();
        Imgproc.cvtColor(claheBgr, lab, Imgproc.COLOR_BGR2Lab);
        byte[] labBuf = new byte[(int) (lab.total() * lab.channels())];
        lab.get(0, 0, labBuf);
        lab.release();

        byte[] coral = new byte[nPix];
        coralMask.get(0, 0, coral);

        byte[] card = null;
        if (GATE_TO_CARD && cardMask != null && !cardMask.empty()
                && cardMask.rows() == rows && cardMask.cols() == cols) {
            card = new byte[nPix];
            cardMask.get(0, 0, card);
        }

        // --- per-coral-component average Lab = "the color of the pixels in the mask" ---
        // Each colony is usually its own connected component, so brown vs dark-ridged colonies
        // each get matched against their OWN average color rather than one global blend.
        Mat compMat = new Mat();
        int nComp = Imgproc.connectedComponents(coralMask, compMat); // CV_32S, 0 = background
        int[] comp = new int[nPix];
        compMat.get(0, 0, comp);
        compMat.release();

        double[] sumL = new double[nComp];
        double[] sumA = new double[nComp];
        double[] sumB = new double[nComp];
        long[] cnt = new long[nComp];
        for (int p = 0, q = 0; p < nPix; p++, q += 3) {
            int c = comp[p];
            if (c == 0) {
                continue;
            }
            sumL[c] += labBuf[q] & 0xFF;
            sumA[c] += labBuf[q + 1] & 0xFF;
            sumB[c] += labBuf[q + 2] & 0xFF;
            cnt[c]++;
        }
        double[] meanL = new double[nComp];
        double[] meanA = new double[nComp];
        double[] meanB = new double[nComp];
        for (int c = 1; c < nComp; c++) {
            if (cnt[c] > 0) {
                meanL[c] = sumL[c] / cnt[c];
                meanA[c] = sumA[c] / cnt[c];
                meanB[c] = sumB[c] / cnt[c];
            }
        }

        // --- breadth-first grow from the coral boundary, ring by ring ---
        // grown[] starts as the original coral and we only ever ADD pixels.
        byte[] grown = coral.clone();
        // dist[] = how many rings a pixel is past the original edge (-1 = untouched).
        short[] dist = new short[nPix];
        Arrays.fill(dist, (short) -1);
        // We reuse comp[] to carry each grown pixel's source component id outward.

        final int[] dr = {-1, -1, -1, 0, 0, 1, 1, 1};
        final int[] dc = {-1, 0, 1, -1, 1, -1, 0, 1};

        ArrayDeque<Integer> queue = new ArrayDeque<>();

        // Seed only the boundary coral pixels (coral that touches a non-coral pixel) at ring 0.
        for (int p = 0; p < nPix; p++) {
            if (coral[p] == 0) {
                continue;
            }
            int r = p / cols;
            int c = p % cols;
            boolean boundary = false;
            for (int k = 0; k < 8 && !boundary; k++) {
                int nr = r + dr[k];
                int nc = c + dc[k];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                    continue;
                }
                if (coral[nr * cols + nc] == 0) {
                    boundary = true;
                }
            }
            if (boundary) {
                dist[p] = 0;
                queue.add(p); // comp[p] already holds this pixel's component id
            }
        }

        while (!queue.isEmpty()) {
            int p = queue.removeFirst();
            int d = dist[p];
            if (d >= maxRadius) {
                continue; // reached the "only a bit" limit
            }
            int r = p / cols;
            int c = p % cols;
            int compId = comp[p];
            double mL = meanL[compId];
            double mA = meanA[compId];
            double mB = meanB[compId];

            for (int k = 0; k < 8; k++) {
                int nr = r + dr[k];
                int nc = c + dc[k];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                    continue;
                }
                int np = nr * cols + nc;
                if (grown[np] != 0) {
                    continue; // already coral or already added
                }
                if (card != null && card[np] == 0) {
                    continue; // off-card: never grow here
                }
                int q = np * 3;
                double dL = (labBuf[q] & 0xFF) - mL;
                double dA = (labBuf[q + 1] & 0xFF) - mA;
                double dB = (labBuf[q + 2] & 0xFF) - mB;
                if (Math.sqrt(dL * dL + dA * dA + dB * dB) > COLOR_TOLERANCE) {
                    continue; // not similar enough to this colony's color
                }
                grown[np] = (byte) 255;
                dist[np] = (short) (d + 1);
                comp[np] = compId; // carry the colony id outward
                queue.add(np);
            }
        }

        Mat out = new Mat(rows, cols, CvType.CV_8UC1);
        out.put(0, 0, grown);
        return out;
    }

    /** Scales {@link #MAX_GROW_RADIUS_PX} to the actual image size when scaling is enabled. */
    private static int scaledRadius(int rows, int cols) {
        if (!SCALE_RADIUS_WITH_RESOLUTION) {
            return MAX_GROW_RADIUS_PX;
        }
        double scale = Math.sqrt((rows * (double) cols) / REFERENCE_PIXELS);
        return Math.max(1, (int) Math.round(MAX_GROW_RADIUS_PX * scale));
    }
}