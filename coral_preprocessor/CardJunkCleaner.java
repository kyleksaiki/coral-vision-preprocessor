import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Builds the "cleaned" tray image (output 6).
 *
 * <p>Goal: make each card look like the same card with the junk simply not there &mdash; no green
 * algae specks, no brown/silt blobs, no dark halos &mdash; while keeping the card's real gentle
 * lighting variation (NOT flattening it to one uniform color).</p>
 *
 * <p>How it works:</p>
 * <ol>
 *   <li><b>Classify</b> every on-card, non-coral pixel as either CLEAN card (bright + near-neutral
 *       color) or JUNK (anything colorful or dark: algae, silt, brown, coral halo).</li>
 *   <li><b>Reconstruct per card</b>: for each card separately, fill its junk with a weighted local
 *       average of <i>that card's own</i> clean pixels (a normalized convolution). Sourcing only
 *       from the same card means a heavily-fouled card is filled from its own clean plastic and its
 *       own mean &mdash; never drifting toward the whole tray's average (which looked flat/grey).
 *       No coral purple or off-card mesh-green can bleed in, and the fill follows the card's
 *       lighting instead of being one flat tone.</li>
 *   <li><b>Composite</b>: keep coral, clean card, and off-card background as they are; replace only
 *       junk pixels with the reconstructed background.</li>
 *   <li><b>De-speck the coral</b>: repaint near-white specks/glare sitting ON the coral with that
 *       colony's own color.</li>
 * </ol>
 */
public final class CardJunkCleaner {

    // ============================== TUNING ==============================

    /**
     * "Clean card" brightness floor (Lab L, 0–255). On-card non-coral pixels DARKER than this are
     * treated as junk (covers coral halos and dark blobs). LOWER keeps more; HIGHER removes more.
     */
    public static final int CLEAN_MIN_L = 160;

    /**
     * "Clean card" color limit: max Lab chroma = sqrt((a-128)^2 + (b-128)^2). On-card non-coral
     * pixels MORE colorful than this are treated as junk (covers green algae, pink/red silt, brown).
     * LOWER removes more color; HIGHER keeps more.
     */
    public static final double CLEAN_MAX_CHROMA = 16.0;

    /** Working resolution (longest side, px) for each card's smooth background estimate. */
    public static final int BG_WORK_MAX_DIM = 768;

    /** Reach of the local averaging, as a fraction of the card's working longest side. */
    public static final double BG_BLUR_FRAC = 0.06;

    /**
     * Tiny "prior" weight toward the card's own clean mean. Negligible where clean card exists
     * nearby; it just guarantees a sensible fill (and no divide-by-zero) in spots fully surrounded
     * by junk. Leave small.
     */
    private static final double BG_PRIOR_WEIGHT = 0.005;

    /** Fallback card color (BGR) if a card has literally no clean pixels to learn from. */
    private static final double[] FALLBACK_BGR = {220, 220, 220};

    // ---- white specks / glare sitting ON the coral ----
    /**
     * A coral pixel is treated as a white speck (debris/glare from the photo) and repainted with
     * its colony's color when ALL of these hold:
     *   (a) near-neutral color: Lab chroma &le; {@link #SPECK_MAX_CHROMA},
     *   (b) bright in absolute terms: Lab L &ge; {@link #SPECK_MIN_L}, and
     *   (c) much brighter than its colony: Lab L &ge; colony median L + {@link #SPECK_BRIGHT_DELTA}.
     * (a)+(c) keep this from touching a coral's real (brownish, not-much-brighter) polyp centers.
     */
    public static final double SPECK_MAX_CHROMA = 20.0;
    public static final int SPECK_MIN_L = 150;
    public static final int SPECK_BRIGHT_DELTA = 55;

    // ====================================================================

    private CardJunkCleaner() {
    }

    /**
     * Returns the cleaned image: junk replaced by a per-card reconstructed clean-card background;
     * coral, clean card, and off-card background left as-is; white specks on coral repainted.
     *
     * @param clahe     normalized BGR tray image
     * @param cardMask  CV_8UC1 (255 = card)
     * @param coralMask CV_8UC1 (255 = coral); normally the GROWN mask
     */
    public static Mat renderCleaned(Mat clahe, Mat cardMask, Mat coralMask) {
        int rows = clahe.rows();
        int cols = clahe.cols();
        int nPix = rows * cols;

        // ---- 1) classify pixels into clean-card vs junk ----
        Mat lab = new Mat();
        Imgproc.cvtColor(clahe, lab, Imgproc.COLOR_BGR2Lab);
        byte[] labBuf = new byte[(int) (lab.total() * lab.channels())];
        lab.get(0, 0, labBuf);
        lab.release();

        byte[] card = new byte[nPix];
        cardMask.get(0, 0, card);
        byte[] coral = new byte[nPix];
        coralMask.get(0, 0, coral);

        byte[] cleanBuf = new byte[nPix];
        byte[] junkBuf = new byte[nPix];
        for (int p = 0, q = 0; p < nPix; p++, q += 3) {
            boolean onCard = card[p] != 0;
            boolean isCoral = coral[p] != 0;
            if (!onCard || isCoral) {
                continue; // off-card and coral are neither clean-fill-source nor junk
            }
            int L = labBuf[q] & 0xFF;
            double da = (labBuf[q + 1] & 0xFF) - 128.0;
            double db = (labBuf[q + 2] & 0xFF) - 128.0;
            boolean clean = L >= CLEAN_MIN_L && Math.sqrt(da * da + db * db) <= CLEAN_MAX_CHROMA;
            if (clean) {
                cleanBuf[p] = (byte) 255;
            } else {
                junkBuf[p] = (byte) 255; // on-card, not coral, not clean => junk to fill
            }
        }

        // ---- 2+3) reconstruct each card's background from ITS OWN clean pixels, fill its junk ----
        Mat out = clahe.clone();

        Mat cardLabelsMat = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int nCards = Imgproc.connectedComponentsWithStats(cardMask, cardLabelsMat, stats, centroids);
        int[] cardLab = new int[nPix];
        cardLabelsMat.get(0, 0, cardLab);
        cardLabelsMat.release();
        centroids.release();

        for (int cardId = 1; cardId < nCards; cardId++) {
            int left = (int) stats.get(cardId, Imgproc.CC_STAT_LEFT)[0];
            int top  = (int) stats.get(cardId, Imgproc.CC_STAT_TOP)[0];
            int w    = (int) stats.get(cardId, Imgproc.CC_STAT_WIDTH)[0];
            int h    = (int) stats.get(cardId, Imgproc.CC_STAT_HEIGHT)[0];
            if (w <= 0 || h <= 0) {
                continue;
            }
            Rect roi = new Rect(left, top, w, h);
            int rw = w;
            int rh = h;
            int rn = rw * rh;

            // this card's clean-source weight and junk-target masks, within the ROI
            byte[] cleanW = new byte[rn];
            byte[] junkW = new byte[rn];
            boolean anyJunk = false;
            for (int yy = 0; yy < rh; yy++) {
                int gy = top + yy;
                for (int xx = 0; xx < rw; xx++) {
                    int gp = gy * cols + (left + xx);
                    if (cardLab[gp] != cardId) {
                        continue;
                    }
                    int idx = yy * rw + xx;
                    if (cleanBuf[gp] != 0) {
                        cleanW[idx] = (byte) 255;
                    } else if (junkBuf[gp] != 0) {
                        junkW[idx] = (byte) 255;
                        anyJunk = true;
                    }
                }
            }
            if (!anyJunk) {
                continue; // nothing to fill on this card
            }

            Mat claheRoi = new Mat(clahe, roi);
            Mat outRoi = new Mat(out, roi);
            fillCardJunk(claheRoi, outRoi, cleanW, junkW, rw, rh);
            claheRoi.release();
            outRoi.release();
        }
        stats.release();

        // ---- 4) repaint white specks/glare sitting ON the coral with that colony's own color ----
        fillCoralSpecks(out, coralMask, labBuf, rows, cols);

        return out;
    }

    /**
     * Reconstructs the clean-card background for ONE card ROI from its own clean pixels (a weighted
     * local average / normalized convolution), then writes that background into {@code outRoi} only
     * where this card's junk is. The fill follows the card's own lighting and falls back to the
     * card's own clean mean where junk is large &mdash; never the whole tray's average.
     */
    private static void fillCardJunk(Mat claheRoi, Mat outRoi, byte[] cleanW, byte[] junkW,
                                     int rw, int rh) {
        Mat cleanMaskRoi = new Mat(rh, rw, CvType.CV_8UC1);
        cleanMaskRoi.put(0, 0, cleanW);

        // this card's own clean mean = the prior / last-resort fill
        double[] cardMean;
        if (Core.countNonZero(cleanMaskRoi) > 0) {
            Scalar m = Core.mean(claheRoi, cleanMaskRoi);
            cardMean = new double[]{m.val[0], m.val[1], m.val[2]};
        } else {
            cardMean = FALLBACK_BGR;
        }

        double scale = Math.min(1.0, BG_WORK_MAX_DIM / (double) Math.max(rw, rh));
        Size small = new Size(Math.max(1, Math.round(rw * scale)),
                              Math.max(1, Math.round(rh * scale)));

        Mat claheSmall = new Mat();
        Imgproc.resize(claheRoi, claheSmall, small, 0, 0, Imgproc.INTER_AREA);
        Mat claheSmallF = new Mat();
        claheSmall.convertTo(claheSmallF, CvType.CV_32F); // values stay 0..255
        claheSmall.release();

        Mat cleanSmall = new Mat();
        Imgproc.resize(cleanMaskRoi, cleanSmall, small, 0, 0, Imgproc.INTER_AREA);
        cleanMaskRoi.release();
        Mat weight = new Mat();
        cleanSmall.convertTo(weight, CvType.CV_32F, 1.0 / 255.0); // fractional clean weight 0..1
        cleanSmall.release();

        int k = oddKernel(BG_BLUR_FRAC * Math.max(small.width, small.height));
        Size ksize = new Size(k, k);

        // den = blur(weight) + prior
        Mat den = new Mat();
        Imgproc.blur(weight, den, ksize);
        Core.add(den, new Scalar(BG_PRIOR_WEIGHT), den);

        List<Mat> chans = new ArrayList<>(3);
        Core.split(claheSmallF, chans);
        claheSmallF.release();

        List<Mat> bgChans = new ArrayList<>(3);
        for (int c = 0; c < 3; c++) {
            Mat wm = new Mat();
            Core.multiply(chans.get(c), weight, wm);   // clean pixels' color, others zeroed
            Mat num = new Mat();
            Imgproc.blur(wm, num, ksize);
            Core.add(num, new Scalar(BG_PRIOR_WEIGHT * cardMean[c]), num);
            Mat bgc = new Mat();
            Core.divide(num, den, bgc);                // local weighted clean-card average
            bgChans.add(bgc);
            wm.release();
            num.release();
            chans.get(c).release();
        }
        weight.release();
        den.release();

        Mat bgSmallF = new Mat();
        Core.merge(bgChans, bgSmallF);
        for (Mat m : bgChans) {
            m.release();
        }
        Mat bgSmall = new Mat();
        bgSmallF.convertTo(bgSmall, CvType.CV_8UC3);
        bgSmallF.release();

        Mat bgFull = new Mat();
        Imgproc.resize(bgSmall, bgFull, new Size(rw, rh), 0, 0, Imgproc.INTER_LINEAR);
        bgSmall.release();

        Mat junkMaskRoi = new Mat(rh, rw, CvType.CV_8UC1);
        junkMaskRoi.put(0, 0, junkW);
        bgFull.copyTo(outRoi, junkMaskRoi); // outRoi is a submat of `out`, so this writes through
        bgFull.release();
        junkMaskRoi.release();
    }

    /**
     * Repaints near-white specks/glare that sit ON the coral (a source-photo artifact) with the
     * color of that specific coral component, so each colony reads as a clean, solid colony.
     * Operates in place on {@code out}. Real coral (including lighter polyp centers) is left alone;
     * see {@link #isSpeck}. Specks are excluded when computing the fill color so they can't drag it.
     */
    private static void fillCoralSpecks(Mat out, Mat coralMask, byte[] labBuf, int rows, int cols) {
        int nPix = rows * cols;

        Mat compMat = new Mat();
        int nComp = Imgproc.connectedComponents(coralMask, compMat); // CV_32S, 0 = background
        if (nComp <= 1) {
            compMat.release();
            return; // no coral
        }
        int[] comp = new int[nPix];
        compMat.get(0, 0, comp);
        compMat.release();

        // Pass A: per-colony median brightness (Lab L) — the "typical" coral darkness.
        int[][] lHist = new int[nComp][256];
        int[] coralCnt = new int[nComp];
        for (int p = 0, q = 0; p < nPix; p++, q += 3) {
            int c = comp[p];
            if (c == 0) {
                continue;
            }
            lHist[c][labBuf[q] & 0xFF]++;
            coralCnt[c]++;
        }
        int[] medianL = new int[nComp];
        for (int c = 1; c < nComp; c++) {
            medianL[c] = medianFromHist(lHist[c], coralCnt[c]);
        }

        // Pass B: per-colony fill color (median BGR) over the NON-speck coral pixels.
        byte[] outBuf = new byte[nPix * 3];
        out.get(0, 0, outBuf); // coral pixels here still hold the real coral color
        int[][] bHist = new int[nComp][256];
        int[][] gHist = new int[nComp][256];
        int[][] rHist = new int[nComp][256];
        int[] fillCnt = new int[nComp];
        for (int p = 0, q = 0; p < nPix; p++, q += 3) {
            int c = comp[p];
            if (c == 0 || isSpeck(labBuf, q, medianL[c])) {
                continue;
            }
            bHist[c][outBuf[q] & 0xFF]++;
            gHist[c][outBuf[q + 1] & 0xFF]++;
            rHist[c][outBuf[q + 2] & 0xFF]++;
            fillCnt[c]++;
        }
        int[] fillB = new int[nComp];
        int[] fillG = new int[nComp];
        int[] fillR = new int[nComp];
        for (int c = 1; c < nComp; c++) {
            if (fillCnt[c] > 0) {
                fillB[c] = medianFromHist(bHist[c], fillCnt[c]);
                fillG[c] = medianFromHist(gHist[c], fillCnt[c]);
                fillR[c] = medianFromHist(rHist[c], fillCnt[c]);
            }
        }

        // Pass C: repaint the speck pixels with their colony's color.
        boolean changed = false;
        for (int p = 0, q = 0; p < nPix; p++, q += 3) {
            int c = comp[p];
            if (c == 0 || fillCnt[c] == 0) {
                continue;
            }
            if (isSpeck(labBuf, q, medianL[c])) {
                outBuf[q]     = (byte) fillB[c];
                outBuf[q + 1] = (byte) fillG[c];
                outBuf[q + 2] = (byte) fillR[c];
                changed = true;
            }
        }
        if (changed) {
            out.put(0, 0, outBuf);
        }
    }

    /** True if the Lab pixel at offset {@code q} is a near-white speck relative to its colony. */
    private static boolean isSpeck(byte[] labBuf, int q, int colonyMedianL) {
        int L = labBuf[q] & 0xFF;
        double da = (labBuf[q + 1] & 0xFF) - 128.0;
        double db = (labBuf[q + 2] & 0xFF) - 128.0;
        double chroma = Math.sqrt(da * da + db * db);
        return chroma <= SPECK_MAX_CHROMA
                && L >= SPECK_MIN_L
                && L >= colonyMedianL + SPECK_BRIGHT_DELTA;
    }

    /** Median value (0..255) of a 256-bin histogram holding {@code total} samples. */
    private static int medianFromHist(int[] hist, int total) {
        if (total <= 0) {
            return 0;
        }
        int half = (total + 1) / 2;
        int cumulative = 0;
        for (int v = 0; v < 256; v++) {
            cumulative += hist[v];
            if (cumulative >= half) {
                return v;
            }
        }
        return 255;
    }

    /** Nearest odd integer >= 3 (box kernels must be positive; odd keeps them centered). */
    private static int oddKernel(double v) {
        int k = (int) Math.round(v);
        if (k < 3) {
            k = 3;
        }
        if ((k & 1) == 0) {
            k++;
        }
        return k;
    }
}