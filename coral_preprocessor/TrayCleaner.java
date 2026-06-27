import ai.onnxruntime.OrtException;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;

import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Two-model tray pipeline (no classical labeling).
 *
 * <pre>
 *   raw  ----------------------------------------------->  CARD model (in Main)  -> cardMask
 *        card model sees the RAW photo                                              255 = card
 *
 *   raw  --> CLAHE (Lab L) + gray-world white balance  -->  clahe (normalized BGR)
 *
 *   For each card the card model found:
 *       crop that card out of `clahe`  -->  CORAL model  -->  coral mask for the crop
 *   Stitch the per-card coral masks back together, AND with cardMask  -->  coralMask
 *                                                                           255 = coral, on cards only
 *
 *   Per card: if that card contains DARK coral, drop its bright ("light purple") pixels as
 *             false positives. Cards whose coral is light are left alone (light coral is real).
 *   coralMask --> CoralMaskGrower (grow into similar-colored edge pixels) --> coralMaskGrown
 * </pre>
 *
 * <p>Why crop per card: the coral model runs on a single card instead of the whole tray, so
 * each colony fills far more of the model's fixed input and its fine texture survives the
 * resize. The card model already localizes the cards, so the coral model never has to reason
 * about mesh, pipe, ruler, or sand &mdash; and because we AND with the card mask at the end,
 * coral can only ever exist on a card.</p>
 *
 * <p>Outputs (6):
 * 1 raw, 2 clahe, 3 card (card-model overlay), 4 coral (coral-model overlay),
 * 5 coral grown (overlay of the edge-grown mask),
 * 6 cleaned (on-card junk removed via clean-card background reconstruction, coral preserved).</p>
 */
public class TrayCleaner {

    // Colors in OpenCV BGR order.
    private static final int[] CORAL_FILL = {128, 0, 128};   // purple coral tint for the overlay

    /** Skip card-mask blobs smaller than this fraction of the image (noise specks). */
    private static final double CARD_AREA_FRAC_MIN = 0.001;
    /** Pad each card crop by this fraction of its larger side, so coral at the edge isn't clipped. */
    private static final double CROP_PAD_FRAC = 0.03;

    // ============ TUNING: dark-vs-light coral, and the "light purple" removal ============
    // Light coral and dark coral never share a card. So we only remove bright ("light purple")
    // coral on cards that actually contain DARK coral; on all-light-coral cards we keep everything.

    /** A coral pixel counts as DARK coral when its CLAHE background grayscale is <= this (0–255). */
    private static final int DARK_CORAL_MAX_BRIGHTNESS = 110;

    /**
     * A card is treated as a DARK-CORAL card when at least this fraction of its coral pixels are
     * dark. Because light/dark coral never mix on one card, this fraction is near 0 on a light
     * card and high on a dark card, so the exact value isn't sensitive. RAISE it if a light card
     * is wrongly treated as dark; LOWER it if a dark card isn't being recognized.
     */
    private static final double DARK_CORAL_MIN_FRACTION = 0.15;

    /**
     * On DARK-CORAL cards only, coral pixels whose CLAHE background grayscale is brighter than
     * this are dropped (the pale "light purple" false positives). LOWER = more aggressive;
     * HIGHER = removes less. Set to 255 to disable removal entirely. Light-coral cards are never
     * touched regardless of this value.
     */
    private static final int CORAL_MAX_BRIGHTNESS = 150;
    // =====================================================================================

    /**
     * Runs the pipeline and returns the 6 output images.
     *
     * @param raw         the original BGR photo (the SAME image the card model ran on)
     * @param cardMask    CV_8UC1 mask from the card model (255 = card), at {@code raw}'s resolution
     * @param coralFinder the loaded coral model (reused; this method runs it once per card crop)
     * @return the pipeline outputs
     * @throws OrtException if coral inference fails
     */
    public static TrayResult processTray(Mat raw, Mat cardMask, CoralFinderOnnx coralFinder)
            throws OrtException {

        // ---- STEP 2: normalize lighting from RAW, with a sanity-check + raw fallback. ----
        Mat clahe = normalizeOrFallback(raw);

        // ---- STEP 3: card-model mask, visualized on the normalized image. ----
        Mat card = CardFinder.renderCardOverlay(clahe, cardMask);          // image 3: card

        // ---- STEP 4: coral mask = coral model per card crop, gated to cards, then the per-card
        //              "light purple" removal (only on cards that have dark coral). ----
        Mat coralMask = buildCoralMask(clahe, cardMask, coralFinder);
        Mat coral = renderCoralOverlay(clahe, coralMask);                  // image 4: coral

        // ---- STEP 5: grow the coral mask into similar-colored edge pixels, then overlay it. ----
        // All tuning for the growth lives in CoralMaskGrower (radius + color tolerance).
        Mat coralMaskGrown = CoralMaskGrower.grow(clahe, coralMask, cardMask);
        Mat coralGrown = renderCoralOverlay(clahe, coralMaskGrown);        // image 5: coral grown

        // ---- STEP 6: cleaned = remove on-card junk by reconstructing clean-card background. ----
        // Junk (algae/silt/brown/halo) is filled from nearby clean-card pixels; coral is kept.
        // Uses the GROWN mask so the recovered edge coral is preserved. See CardJunkCleaner.
        // (To clean with the un-grown mask instead, pass `coralMask` here.)
        Mat cleaned = CardJunkCleaner.renderCleaned(clahe, cardMask, coralMaskGrown); // image 6: cleaned

        TrayResult result = new TrayResult();
        result.rawInput = raw.clone();
        result.claheWhiteBalanced = clahe;   // we own this Mat
        result.card = card;
        result.coral = coral;
        result.coralGrown = coralGrown;
        result.cleaned = cleaned;
        result.cardMask = cardMask;
        result.coralMask = coralMask;
        result.coralMaskGrown = coralMaskGrown;
        return result;
    }

    /**
     * Builds the full-resolution coral mask by running the coral model on each card crop.
     *
     * <p>For every card blob in {@code cardMask}: take its bounding box (padded a little), crop
     * that region out of the CLAHE image, run the coral model on the crop, and OR the result
     * into the full-size coral mask. Then AND with the card mask so coral lives only on cards
     * (rectangular crops can include mesh/neighbor pixels; this clips them). Finally, apply the
     * per-card "light purple" removal (see {@link #removeBrightCoralOnDarkCards}).</p>
     *
     * <p>Touching cards that the card model merged into one blob become one larger crop &mdash;
     * still correct, just with less of the per-card resolution benefit for those.</p>
     */
    private static Mat buildCoralMask(Mat clahe, Mat cardMask, CoralFinderOnnx coralFinder)
            throws OrtException {
        int rows = cardMask.rows();
        int cols = cardMask.cols();

        Mat coralMask = Mat.zeros(cardMask.size(), CvType.CV_8UC1);

        Mat labels = new Mat();
        Mat stats = new Mat();
        Mat centroids = new Mat();
        int n = Imgproc.connectedComponentsWithStats(cardMask, labels, stats, centroids);

        double minArea = CARD_AREA_FRAC_MIN * rows * cols;

        // Component 0 is the background; real cards are 1..n-1.
        for (int i = 1; i < n; i++) {
            double area = stats.get(i, Imgproc.CC_STAT_AREA)[0];
            if (area < minArea) {
                continue; // tiny speck in the card mask; not a real card
            }

            int left = (int) stats.get(i, Imgproc.CC_STAT_LEFT)[0];
            int top  = (int) stats.get(i, Imgproc.CC_STAT_TOP)[0];
            int w    = (int) stats.get(i, Imgproc.CC_STAT_WIDTH)[0];
            int h    = (int) stats.get(i, Imgproc.CC_STAT_HEIGHT)[0];

            int pad = (int) Math.round(CROP_PAD_FRAC * Math.max(w, h));
            int x0 = Math.max(0, left - pad);
            int y0 = Math.max(0, top - pad);
            int x1 = Math.min(cols, left + w + pad);
            int y1 = Math.min(rows, top + h + pad);
            Rect roi = new Rect(x0, y0, x1 - x0, y1 - y0);

            // Crop the CLAHE image to this card and run the coral model ONLY on that crop.
            Mat crop = new Mat(clahe, roi);
            Mat coralCrop = coralFinder.findCoral(crop);   // CV_8UC1, 255 = coral, crop resolution

            // OR the crop's coral into the full mask at the same place.
            Mat dstRoi = new Mat(coralMask, roi);
            Core.bitwise_or(dstRoi, coralCrop, dstRoi);

            crop.release();
            coralCrop.release();
            dstRoi.release();
        }

        // Coral can only exist on a card. This also clips any mesh/neighbor pixels the
        // rectangular crops may have included.
        Core.bitwise_and(coralMask, cardMask, coralMask);

        // Per-card "light purple" removal, gated on whether the card has dark coral.
        removeBrightCoralOnDarkCards(clahe, coralMask, labels, n);

        labels.release();
        stats.release();
        centroids.release();
        return coralMask;
    }

    /**
     * Removes the pale "light purple" false-positive coral, but ONLY on cards that actually
     * contain dark coral. Light coral and dark coral never share a card, so:
     *
     * <ul>
     *   <li>If a card's coral is mostly dark (a dark-coral card), its bright coral pixels are
     *       false positives and get dropped.</li>
     *   <li>If a card has no meaningful dark coral (a light-coral card), nothing is removed &mdash;
     *       the whole pale colony is real and kept.</li>
     * </ul>
     *
     * <p>Operates in place on {@code coralMask}. Thresholds are the TUNING constants above.</p>
     *
     * @param cardLabels CV_32S connected-component labels of the card mask (0 = background)
     * @param nCards     number of components (0..nCards-1) from connectedComponentsWithStats
     */
    private static void removeBrightCoralOnDarkCards(Mat clahe, Mat coralMask, Mat cardLabels, int nCards) {
        if (CORAL_MAX_BRIGHTNESS >= 255) {
            return; // removal disabled
        }

        int nPix = coralMask.rows() * coralMask.cols();

        Mat gray = new Mat();
        Imgproc.cvtColor(clahe, gray, Imgproc.COLOR_BGR2GRAY);
        byte[] g = new byte[nPix];
        gray.get(0, 0, g);
        gray.release();

        int[] lab = new int[nPix];
        cardLabels.get(0, 0, lab);

        byte[] m = new byte[nPix];
        coralMask.get(0, 0, m);

        // Pass A: per card, count coral pixels and how many of them are dark.
        long[] coralCount = new long[nCards];
        long[] darkCount = new long[nCards];
        for (int p = 0; p < nPix; p++) {
            if (m[p] == 0) {
                continue;
            }
            int card = lab[p];
            if (card <= 0) {
                continue; // coral is gated to cards, but stay safe
            }
            coralCount[card]++;
            if ((g[p] & 0xFF) <= DARK_CORAL_MAX_BRIGHTNESS) {
                darkCount[card]++;
            }
        }

        // Decide which cards are dark-coral cards (enough of their coral is dark).
        boolean[] darkCard = new boolean[nCards];
        for (int c = 1; c < nCards; c++) {
            darkCard[c] = coralCount[c] > 0
                    && darkCount[c] >= DARK_CORAL_MIN_FRACTION * coralCount[c];
        }

        // Pass B: on dark-coral cards only, drop coral pixels that are too bright.
        boolean changed = false;
        for (int p = 0; p < nPix; p++) {
            if (m[p] == 0) {
                continue;
            }
            int card = lab[p];
            if (card > 0 && darkCard[card] && (g[p] & 0xFF) > CORAL_MAX_BRIGHTNESS) {
                m[p] = 0;
                changed = true;
            }
        }
        if (changed) {
            coralMask.put(0, 0, m);
        }
    }

    /**
     * Normalizes {@code raw} with CLAHE + gray-world white balance, returning a clone of raw
     * instead if normalization throws or comes out near-black (the old all-black bug). Keeps the
     * pipeline from ever blanking a tray.
     */
    private static Mat normalizeOrFallback(Mat raw) {
        try {
            Mat normalized = TrayLightingNormalizer.claheLabThenWhiteBalance(raw);
            if (isUsable(normalized)) {
                return normalized;
            }
            if (normalized != null) {
                normalized.release();
            }
            System.err.println("[TrayCleaner] Normalized (CLAHE) image looked degenerate; using raw instead.");
        } catch (RuntimeException e) {
            System.err.println("[TrayCleaner] Normalization failed (" + e.getMessage() + "); using raw instead.");
        }
        return raw.clone();
    }

    /** True if the image exists, has pixels, and is not near-black. */
    private static boolean isUsable(Mat bgr) {
        if (bgr == null || bgr.empty() || bgr.rows() == 0 || bgr.cols() == 0) {
            return false;
        }
        Scalar mean = Core.mean(bgr);
        double avg = (mean.val[0] + mean.val[1] + mean.val[2]) / 3.0;
        return avg >= 5.0;
    }

    /**
     * Coral-model overlay (used for images 4 and 5): a translucent purple fill on coral pixels
     * plus a white outline, drawn over the CLAHE image, so you can eyeball exactly what the model
     * called coral (and, for image 5, what the grow step added).
     */
    private static Mat renderCoralOverlay(Mat clahe, Mat coralMask) {
        Mat out = clahe.clone();
        int rows = out.rows();
        int cols = out.cols();
        int ch = out.channels();

        byte[] pix = new byte[(int) (out.total() * ch)];
        out.get(0, 0, pix);
        byte[] coral = new byte[rows * cols];
        coralMask.get(0, 0, coral);

        final double alpha = 0.45; // strength of the purple tint on coral
        int i = 0;
        int p = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++, i += ch, p++) {
                if (coral[p] != 0) {
                    for (int k = 0; k < 3; k++) {
                        int orig = pix[i + k] & 0xFF;
                        int blended = (int) Math.round(alpha * CORAL_FILL[k] + (1 - alpha) * orig);
                        pix[i + k] = (byte) Math.max(0, Math.min(255, blended));
                    }
                }
            }
        }
        out.put(0, 0, pix);

        // White outline around each coral region.
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(coralMask, contours, hierarchy,
                Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
        int thickness = Math.max(2, cols / 1000);
        Imgproc.drawContours(out, contours, -1, new Scalar(255, 255, 255), thickness);
        for (MatOfPoint cnt : contours) {
            cnt.release();
        }
        hierarchy.release();
        return out;
    }

    /** The 6 pipeline outputs plus the binary masks. */
    public static class TrayResult {
        public Mat rawInput;            // 1: raw
        public Mat claheWhiteBalanced;  // 2: CLAHE + white balance (or raw, on fallback)
        public Mat card;                // 3: card-model mask overlay
        public Mat coral;               // 4: coral-model mask overlay (after per-card bright removal)
        public Mat coralGrown;          // 5: overlay of the edge-grown mask
        public Mat cleaned;             // 6: on-card junk removed (clean-card bg reconstruction), coral kept
        public Mat cardMask;            // CV_8UC1: 255 = card  (from the card model)
        public Mat coralMask;           // CV_8UC1: 255 = coral (coral model, gated to cards, filtered)
        public Mat coralMaskGrown;      // CV_8UC1: 255 = coral (after edge growth)
    }
}