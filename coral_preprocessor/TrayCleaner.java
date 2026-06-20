import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;

import java.util.Arrays;

/**
 * Cleans a tray image using a FORKED pipeline:
 *
 * <pre>
 *   raw  ----------------------------------------------->  ONNX card model (in the caller)
 *        the model sees the RAW photo                       -> cardMask (CV_8UC1, 255 = card)
 *
 *   raw  --> CLAHE (Lab L) + gray-world white balance  -->  clahe (normalized BGR)
 *
 *   clahe + cardMask --> coral -> algae -> silt labelers, GATED so a pixel the model
 *                        did not call "card" is never labeled coral/algae/silt.
 * </pre>
 *
 * <p>Why fork: the card model was trained on raw photos, so it must be fed raw. The classical
 * color thresholds, on the other hand, need normalized lighting, so they run on {@code clahe}.
 * The model decides <i>where</i> the cards are; the classical CV decides <i>what</i> is on them,
 * but only inside the card footprint. Distractors (sand, mesh, PVC pipe, ruler, paper labels)
 * are outside the mask and therefore can never be labeled coral/algae/silt.</p>
 *
 * <p>Per-pixel labels:</p>
 * <ul>
 *   <li>{@code 'C'/'c'} coral, {@code 'A'/'a'} algae, {@code 'S'/'s'} silt</li>
 *   <li>{@code 'G'/'g'} glue, {@code 'D'/'d'} dark algae, {@code 'N'/'n'} noise,
 *       {@code 'H'/'h'} shadow (all removable)</li>
 *   <li>{@code '.'} unlabeled (includes everything outside the card mask)</li>
 * </ul>
 */
public class TrayCleaner {

    // Colors in OpenCV BGR order.
    private static final int[] PURPLE    = {128, 0, 128};
    private static final int[] GREEN     = {0, 255, 0};
    private static final int[] RED       = {0, 0, 255};
    private static final int[] DIM_WHITE = {220, 220, 220};

    // Per-label color tables, indexed by label char. A null entry leaves the pixel unchanged.
    private static final int[][] T_CORAL        = new int[128][]; // coral only
    private static final int[][] T_CORAL_ALGAE  = new int[128][]; // coral + algae + silt
    private static final int[][] T_REMOVABLE    = new int[128][]; // removable -> dim white
    static {
        put(T_CORAL, "Cc", PURPLE);
        put(T_CORAL_ALGAE, "Cc", PURPLE);
        put(T_CORAL_ALGAE, "Aa", GREEN);
        put(T_CORAL_ALGAE, "Ss", RED);
        put(T_REMOVABLE, "AaSsGgDdNnHh", DIM_WHITE);
    }

    /**
     * Runs the forked pipeline and returns the 7 output images plus the label grid.
     *
     * <p>The card mask is produced by the caller (load {@link CardFinderOnnx} once at startup,
     * then call {@code finder.findCards(raw)} per image) and passed in here. That keeps this
     * class free of any ONNX dependency, and lets the caller swap in a different mask source
     * (e.g. the old classical {@link CardFinder#findCards(Mat)}) without touching this code.</p>
     *
     * @param raw      the original BGR tray photo &mdash; the SAME image the card model ran on
     * @param cardMask CV_8UC1 mask from the model (255 = card, 0 = not), at {@code raw}'s
     *                 resolution. Pixels where this is 0 are never labeled coral/algae/silt.
     * @return the pipeline outputs
     */
    public static TrayResult processTray(Mat raw, Mat cardMask) {
        int rows = raw.rows();
        int cols = raw.cols();

        // ---- STEP 2: normalize lighting from the RAW image. -----------------------------
        // Sanity-check the result (we once had an all-black CLAHE bug) and fall back to raw
        // if the normalized image is unusable, so a bad normalization can never blank a tray.
        Mat clahe = normalizeOrFallback(raw);

        // ---- STEP 3: the card mask (from the model, fed RAW) is just visualized here. ----
        // We overlay it on the normalized image so images 3-7 all share the same backdrop;
        // switch the first arg to `raw` if you'd rather see what the model literally saw.
        Mat card = CardFinder.renderCardOverlay(clahe, cardMask);        // image 3: card

        // Empty label grid; '.' everywhere.
        char[][] labels = new char[rows][cols];
        for (char[] row : labels) {
            Arrays.fill(row, '.');
        }

        // ================================================================================
        // ALL CLASSICAL LABELING RUNS ON `clahe` (normalized) AND IS GATED BY `cardMask`.
        // Each labeler skips any pixel where cardMask == 0, so nothing off-card is labeled.
        // ================================================================================

        // Coral first so algae/silt cannot overwrite it.
        labels = LabelCoral.labelCoralPixels(clahe, labels, cardMask);    // gated
        Mat labelCoral = renderLabels(clahe, labels, T_CORAL);            // image 4: coral

        labels = LabelAlgae.labelAlgaePixels(clahe, labels, cardMask);    // gated
        labels = LabelSilt.labelSiltPixels(clahe, labels, cardMask);      // gated (feeds "cleaned")
        Mat labelAlgae = renderLabels(clahe, labels, T_CORAL_ALGAE);      // image 5: algae

        Mat cleaned = renderLabels(clahe, labels, T_REMOVABLE);           // image 6: cleaned

        CoralMaskRefiner.CoralComponents components = CoralMaskRefiner.labelConnectedComponents(labels);
        int[][] componentIds = components.ids;
        Mat coralComponents = renderComponents(clahe, labels, componentIds); // image 7: components

        TrayResult result = new TrayResult();
        result.rawInput = raw.clone();
        result.claheWhiteBalanced = clahe;     // we own this Mat; no extra clone needed
        result.card = card;
        result.labelCoral = labelCoral;
        result.labelAlgae = labelAlgae;
        result.cleaned = cleaned;
        result.coralComponents = coralComponents;
        result.labels = labels;
        result.cardMask = cardMask;
        result.coralComponentIds = componentIds;
        return result;
    }

    /**
     * Normalizes {@code raw} with CLAHE + gray-world white balance, but only returns the
     * normalized image if it looks valid. If normalization throws, or the result is empty /
     * near-black (the old all-black bug), it returns a clone of {@code raw} instead so the
     * downstream labelers always get a real image to work on.
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

    /** True if the image exists, has pixels, and is not near-black (mean brightness too low). */
    private static boolean isUsable(Mat bgr) {
        if (bgr == null || bgr.empty() || bgr.rows() == 0 || bgr.cols() == 0) {
            return false;
        }
        Scalar mean = Core.mean(bgr);
        double avg = (mean.val[0] + mean.val[1] + mean.val[2]) / 3.0;
        return avg >= 5.0; // an all-black / near-black image has avg ~ 0
    }

    /**
     * Clones {@code source} and recolors each pixel whose label has an entry in
     * {@code colorByLabel}; labels with a null entry keep their original pixel. Off-card
     * pixels are '.' (never labeled), so they always keep the normalized background.
     */
    private static Mat renderLabels(Mat source, char[][] labels, int[][] colorByLabel) {
        Mat out = source.clone();
        int rows = out.rows();
        int cols = out.cols();
        int ch = out.channels();

        byte[] data = new byte[(int) (out.total() * ch)];
        out.get(0, 0, data);

        int i = 0;
        for (int r = 0; r < rows; r++) {
            char[] row = labels[r];
            for (int c = 0; c < cols; c++, i += ch) {
                int[] color = colorByLabel[row[c]];
                if (color != null) {
                    data[i] = (byte) color[0];
                    data[i + 1] = (byte) color[1];
                    data[i + 2] = (byte) color[2];
                }
            }
        }

        out.put(0, 0, data);
        return out;
    }

    /**
     * Clones {@code source}, replaces removable labels with dim white, and paints
     * each connected coral component (id from {@code ids}) its own color. Coral
     * pixels whose id is 0 (eroded away) keep their natural pixel.
     */
    private static Mat renderComponents(Mat source, char[][] labels, int[][] ids) {
        Mat out = source.clone();
        int rows = out.rows();
        int cols = out.cols();
        int ch = out.channels();

        int maxId = 0;
        for (int[] row : ids) {
            for (int v : row) {
                if (v > maxId) {
                    maxId = v;
                }
            }
        }
        int[][] palette = new int[maxId + 1][];
        for (int id = 1; id <= maxId; id++) {
            palette[id] = componentColor(id);
        }

        byte[] data = new byte[(int) (out.total() * ch)];
        out.get(0, 0, data);

        int i = 0;
        for (int r = 0; r < rows; r++) {
            char[] lrow = labels[r];
            int[] crow = ids[r];
            for (int c = 0; c < cols; c++, i += ch) {
                char label = lrow[c];
                int[] color = T_REMOVABLE[label]; // removable -> dim white, else null
                if (color == null && (label == 'C' || label == 'c') && crow[c] >= 1) {
                    color = palette[crow[c]];
                }
                if (color != null) {
                    data[i] = (byte) color[0];
                    data[i + 1] = (byte) color[1];
                    data[i + 2] = (byte) color[2];
                }
            }
        }

        out.put(0, 0, data);
        return out;
    }

    /** Distinct BGR color per component id, spread via the golden angle. */
    private static int[] componentColor(int id) {
        double hue = ((id * 0.61803398875) % 1.0) * 360.0;
        double c = 0.85;                 // saturation * value (value = 1)
        double x = c * (1 - Math.abs(((hue / 60.0) % 2) - 1));
        double m = 1.0 - c;
        double r;
        double g;
        double b;
        if (hue < 60)       { r = c; g = x; b = 0; }
        else if (hue < 120) { r = x; g = c; b = 0; }
        else if (hue < 180) { r = 0; g = c; b = x; }
        else if (hue < 240) { r = 0; g = x; b = c; }
        else if (hue < 300) { r = x; g = 0; b = c; }
        else                { r = c; g = 0; b = x; }
        return new int[] {to255(b + m), to255(g + m), to255(r + m)}; // BGR
    }

    /** Maps label chars to a color in the given table. */
    private static void put(int[][] table, String chars, int[] color) {
        for (int i = 0; i < chars.length(); i++) {
            table[chars.charAt(i)] = color;
        }
    }

    /** Scales a 0..1 value to a clamped 0..255 int. */
    private static int to255(double v) {
        return Math.max(0, Math.min(255, (int) Math.round(v * 255)));
    }

    /** The 7 pipeline outputs plus the label grid, the binary card mask, and coral ids. */
    public static class TrayResult {
        public Mat rawInput;           // 1: raw
        public Mat claheWhiteBalanced; // 2: CLAHE + white balance (or raw, on fallback)
        public Mat card;               // 3: model card mask, visualized as an instance overlay
        public Mat labelCoral;         // 4: coral (purple), inside the card mask only
        public Mat labelAlgae;         // 5: coral purple + algae green + silt red, mask only
        public Mat cleaned;            // 6: removable -> dim white, coral preserved
        public Mat coralComponents;    // 7: each connected coral blob its own color
        public char[][] labels;
        public Mat cardMask;               // CV_8UC1: 255 = card, 0 = not (from the model)
        public int[][] coralComponentIds;  // 0 = not coral, 1..N = connected coral blob id
    }
}