import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Edge-growth and shape-cleanup for the coral mask.
 */
public final class CoralMaskGrower {

    // ===================== TUNING =====================

    /**
     * Max pixels the mask may extend past its original edge. 
     * Serves as a safety backstop.
     */
    public static final int MAX_GROW_RADIUS_PX = 200;

    /**
     * Lightness tolerance. Set generously to ~50.0 to allow the mask to span 
     * a coral's full dark-to-light range (valleys vs ridges).
     */
    public static final double COLOR_TOL_L = 50.0;

    /** 
     * Hue half-tolerance in Lab a/b. Widened to 25.0 to capture the 
     * green-brown and yellow-brown edges of the colonies, while still 
     * stopping before bright green algae or pure white card.
     */
    public static final double COLOR_TOL_AB = 25.0;

    /**
     * Solidity window radius (px @ ref resolution). 
     */
    public static final int SOLID_RADIUS_PX = 10;

    /**
     * Minimum fraction of the solidity window that must be the colony's color.
     * Lowered slightly to 0.55 to allow growth over mottled, textured corals.
     */
    public static final double SOLID_MIN_FRACTION = 0.7;

    /** Never grow onto a pixel the card model did not call "card". */
    public static final boolean GATE_TO_CARD = true;

    // ---- 2) rounding ----
    public static final int ROUND_CLOSE_PX = 16;
    public static final int ROUND_OPEN_PX = 8;
    public static final int ROUND_SMOOTH_PX = 10;

    // ---- resolution scaling ----
    public static final boolean SCALE_RADIUS_WITH_RESOLUTION = true;
    private static final double REFERENCE_PIXELS = 4000.0 * 3000.0;

    // =============================================================================================

    private CoralMaskGrower() {
    }

    public static Mat grow(Mat claheBgr, Mat coralMask, Mat cardMask) {
        final int rows = coralMask.rows();
        final int cols = coralMask.cols();
        final int nPix = rows * cols;

        final int maxRadius = scalePx(MAX_GROW_RADIUS_PX, rows, cols);
        final int solidRad = Math.max(1, scalePx(SOLID_RADIUS_PX, rows, cols));

        // Lab conversion
        Mat lab = new Mat();
        Imgproc.cvtColor(claheBgr, lab, Imgproc.COLOR_BGR2Lab);
        byte[] labBuf = new byte[(int) (lab.total() * lab.channels())];
        lab.get(0, 0, labBuf);
        lab.release();

        byte[] coral = new byte[nPix];
        coralMask.get(0, 0, coral);

        Mat cardGate = null;
        byte[] card = null;
        if (GATE_TO_CARD && cardMask != null && !cardMask.empty()
                && cardMask.rows() == rows && cardMask.cols() == cols) {
            cardGate = cardMask.clone();
            fillHoles(cardGate);
            Core.bitwise_or(cardGate, coralMask, cardGate);
            card = new byte[nPix];
            cardGate.get(0, 0, card);
        }

        Mat compMat = new Mat();
        int nComp = Imgproc.connectedComponents(coralMask, compMat);
        int[] comp = new int[nPix];
        compMat.get(0, 0, comp);
        compMat.release();

        if (nComp <= 1) {
            return Mat.zeros(coralMask.size(), CvType.CV_8UC1);
        }

        double[] sumL = new double[nComp];
        double[] sumA = new double[nComp];
        double[] sumB = new double[nComp];
        long[] cnt = new long[nComp];
        
        for (int p = 0, q = 0; p < nPix; p++, q += 3) {
            int c = comp[p];
            if (c == 0) continue;
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

        byte[] grown = coral.clone();
        short[] dist = new short[nPix];
        Arrays.fill(dist, (short) -1);

        final int[] dr = {-1, -1, -1, 0, 0, 1, 1, 1};
        final int[] dc = {-1, 0, 1, -1, 1, -1, 0, 1};

        ArrayDeque<Integer> queue = new ArrayDeque<>();
        for (int p = 0; p < nPix; p++) {
            if (coral[p] == 0) continue;
            int r = p / cols;
            int c = p % cols;
            boolean boundary = false;
            for (int k = 0; k < 8 && !boundary; k++) {
                int nr = r + dr[k];
                int nc = c + dc[k];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                if (coral[nr * cols + nc] == 0) boundary = true;
            }
            if (boundary) {
                dist[p] = 0;
                queue.add(p);
            }
        }

        while (!queue.isEmpty()) {
            int p = queue.removeFirst();
            int d = dist[p];
            if (d >= maxRadius) continue;
            
            int r = p / cols;
            int c = p % cols;
            int compId = comp[p];
            double mL = meanL[compId];
            double mA = meanA[compId];
            double mB = meanB[compId];

            for (int k = 0; k < 8; k++) {
                int nr = r + dr[k];
                int nc = c + dc[k];
                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
                
                int np = nr * cols + nc;
                if (grown[np] != 0) continue;
                if (card != null && card[np] == 0) continue;
                
                int q = np * 3;
                double dL = (labBuf[q] & 0xFF) - mL;
                double dA = (labBuf[q + 1] & 0xFF) - mA;
                double dB = (labBuf[q + 2] & 0xFF) - mB;
                
                if (!colorClose(dL, dA, dB)) continue;
                
                double solid = localSolidFraction(labBuf, card, nr, nc, solidRad, mL, mA, mB, rows, cols);
                if (solid < SOLID_MIN_FRACTION) continue;
                
                grown[np] = (byte) 255;
                dist[np] = (short) (d + 1);
                comp[np] = compId;
                queue.add(np);
            }
        }

        Mat result = new Mat(rows, cols, CvType.CV_8UC1);
        result.put(0, 0, grown);

        roundMask(result, rows, cols);
        if (cardGate != null) {
            Core.bitwise_and(result, cardGate, result);
            cardGate.release();
        }

        // --- Final merge guard: prevents fusion of distinct colonies ---
        revertStillMergedToBase(result, coralMask, rows, cols);

        fillHoles(result);
        return result;
    }

    private static boolean colorClose(double dL, double dA, double dB) {
        return (dL * dL) / (COLOR_TOL_L * COLOR_TOL_L)
                + (dA * dA + dB * dB) / (COLOR_TOL_AB * COLOR_TOL_AB) <= 1.0;
    }

    private static double localSolidFraction(byte[] labBuf, byte[] card, int r, int c, int rad,
                                             double mL, double mA, double mB,
                                             int rows, int cols) {
        int y0 = Math.max(0, r - rad);
        int y1 = Math.min(rows - 1, r + rad);
        int x0 = Math.max(0, c - rad);
        int x1 = Math.min(cols - 1, c + rad);
        int total = 0;
        int hit = 0;
        for (int yy = y0; yy <= y1; yy++) {
            int rowBase = yy * cols;
            for (int xx = x0; xx <= x1; xx++) {
                int pp = rowBase + xx;
                total++;
                if (card != null && card[pp] == 0) continue;
                
                int qq = pp * 3;
                double dL = (labBuf[qq] & 0xFF) - mL;
                double dA = (labBuf[qq + 1] & 0xFF) - mA;
                double dB = (labBuf[qq + 2] & 0xFF) - mB;
                if (colorClose(dL, dA, dB)) hit++;
            }
        }
        return total == 0 ? 0.0 : (double) hit / total;
    }

    private static void revertStillMergedToBase(Mat result, Mat baseCoral, int rows, int cols) {
        int nPix = rows * cols;

        Mat baseLab = new Mat();
        Imgproc.connectedComponents(baseCoral, baseLab);
        int[] base = new int[nPix];
        baseLab.get(0, 0, base);
        baseLab.release();

        Mat resLab = new Mat();
        int nRes = Imgproc.connectedComponents(result, resLab);
        int[] res = new int[nPix];
        resLab.get(0, 0, res);
        resLab.release();

        int[] firstBase = new int[nRes];   
        boolean[] merged = new boolean[nRes];
        boolean any = false;
        
        for (int p = 0; p < nPix; p++) {
            int rc = res[p];
            int bb = base[p];
            if (rc <= 0 || bb <= 0) continue;
            
            if (firstBase[rc] == 0) {
                firstBase[rc] = bb;
            } else if (firstBase[rc] != bb && !merged[rc]) {
                merged[rc] = true; 
                any = true;
            }
        }
        if (!any) return;

        byte[] out = new byte[nPix];
        result.get(0, 0, out);
        for (int p = 0; p < nPix; p++) {
            int rc = res[p];
            if (rc > 0 && merged[rc]) {
                out[p] = (base[p] > 0) ? (byte) 255 : 0;
            }
        }
        result.put(0, 0, out);
    }

    private static void roundMask(Mat mask, int rows, int cols) {
        int closeK = scalePx(ROUND_CLOSE_PX, rows, cols);
        int openK = scalePx(ROUND_OPEN_PX, rows, cols);
        int smoothK = scalePx(ROUND_SMOOTH_PX, rows, cols);

        if (closeK > 0) {
            Mat k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2 * closeK + 1, 2 * closeK + 1));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_CLOSE, k);
            k.release();
        }
        if (openK > 0) {
            Mat k = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(2 * openK + 1, 2 * openK + 1));
            Imgproc.morphologyEx(mask, mask, Imgproc.MORPH_OPEN, k);
            k.release();
        }
        if (smoothK > 0) {
            int g = 2 * smoothK + 1;
            Imgproc.GaussianBlur(mask, mask, new Size(g, g), 0);
            Imgproc.threshold(mask, mask, 127, 255, Imgproc.THRESH_BINARY);
        }
    }

    private static void fillHoles(Mat mask) {
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Mat probe = mask.clone();
        Imgproc.findContours(probe, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        probe.release();
        Imgproc.drawContours(mask, contours, -1, new Scalar(255), -1);
        for (MatOfPoint c : contours) c.release();
        hierarchy.release();
    }

    private static int scalePx(int px, int rows, int cols) {
        if (!SCALE_RADIUS_WITH_RESOLUTION) return Math.max(0, px);
        double scale = Math.sqrt((rows * (double) cols) / REFERENCE_PIXELS);
        return Math.max(0, (int) Math.round(px * scale));
    }
}