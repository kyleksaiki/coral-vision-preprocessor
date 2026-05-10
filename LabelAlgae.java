import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Labels algae-like pixels in a card image using color thresholds in the Lab color space.
 *
 * <p>This class is designed to detect yellow or olive-colored algae-like regions while
 * avoiding pixels that have already been labeled as coral.</p>
 *
 * <p>The labeling process uses three main passes:</p>
 *
 * <ol>
 *   <li>Find relaxed algae candidates and label them as weak algae using {@code 'a'}.</li>
 *   <li>Select the strongest yellow/olive candidates and promote them to strong algae using {@code 'A'}.</li>
 *   <li>Grow strong algae regions through connected weak algae pixels.</li>
 * </ol>
 *
 * <p>Label meanings used by this class:</p>
 *
 * <ul>
 *   <li>{@code 'A'} = strong algae</li>
 *   <li>{@code 'a'} = weak algae candidate</li>
 *   <li>{@code 'C'} or {@code 'c'} = coral, which algae detection will not overwrite</li>
 * </ul>
 */
public class LabelAlgae {

    // Simple Lab thresholds for the first algae pass.
    // This is aimed at yellow / olive-ish stuff, not coral.
    /**
     * Minimum Lab L-channel value required for a pixel to be considered algae-like.
     *
     * <p>The L channel represents brightness. This threshold prevents very dark pixels
     * from being labeled as algae during the first pass.</p>
     */
    private static final int MIN_L = 50;
    /**
     * Maximum Lab a-channel value allowed for algae candidates.
     *
     * <p>The a channel separates green from red/magenta. This threshold helps avoid
     * labeling redder coral-like pixels as algae.</p>
     */
    private static final int MAX_A = 145;
    /**
     * Minimum Lab b-channel value required for algae candidates.
     *
     * <p>The b channel separates blue from yellow. Higher values indicate stronger
     * yellow tones, which are useful for detecting yellow/olive algae.</p>
     */
    private static final int MIN_B = 132;

    /**
     * Fraction of algae candidates that are promoted to strong algae seeds.
     *
     * <p>Only the highest-scoring algae-like pixels are promoted to {@code 'A'} initially.
     * Remaining candidates stay as weak algae {@code 'a'} unless connected growth promotes them.</p>
     */
    // Only the strongest chunk of algae-like pixels become seed A pixels at first.
    private static final double TOP_YELLOW_FRACTION = 0.20;

    /**
     * Labels algae-like pixels in the provided label map.
     *
     * <p>The input image is converted from BGR to Lab color space. Each pixel is then checked
     * against relaxed yellow/olive thresholds. Candidate algae pixels are first marked as
     * weak algae {@code 'a'}, then the strongest candidates are promoted to strong algae
     * {@code 'A'}. Finally, strong algae regions grow through connected weak algae pixels.</p>
     *
     * <p>This method does not overwrite pixels already labeled as coral using {@code 'C'} or
     * {@code 'c'}.</p>
     *
     * @param bgrImage OpenCV image in BGR color format
     * @param labels existing label grid to update; must match the image dimensions
     * @return the same {@code labels} array after algae pixels have been labeled
     */
    public static char[][] labelAlgaePixels(Mat bgrImage, char[][] labels) {

        // Convert once to Lab so brightness and color separation are easier to work
        // with.
        Mat lab = new Mat();
        Imgproc.cvtColor(bgrImage, lab, Imgproc.COLOR_BGR2Lab);

        int rows = lab.rows();
        int cols = lab.cols();

        // Keep a score per pixel so we can rank candidates later.
        double[][] yellowScores = new double[rows][cols];
        boolean[][] candidateMask = new boolean[rows][cols];
        List<Double> candidateScores = new ArrayList<>();

        // Pull Lab image into one flat byte array for faster access.
        byte[] data = new byte[(int) (lab.total() * lab.channels())];
        lab.get(0, 0, data);
        lab.release();

        int index = 0;

        // PASS 1:
        // mark possible algae pixels as weak algae 'a'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                int l = data[index] & 0xFF;
                int a = data[index + 1] & 0xFF;
                int b = data[index + 2] & 0xFF;
                index += 3;

                yellowScores[r][c] = Double.NEGATIVE_INFINITY;
                candidateMask[r][c] = false;

                // Do not let algae overwrite coral.
                if (labels[r][c] == 'C' || labels[r][c] == 'c') {
                    continue;
                }

                // Skip anything that does not look algae-like.
                if (!passesRelaxedYellowGate(l, a, b)) {
                    continue;
                }

                double score = yellowScore(l, a, b);

                yellowScores[r][c] = score;
                candidateMask[r][c] = true;
                candidateScores.add(score);

                labels[r][c] = 'a';
            }
        }

        // Nothing matched, so just return as-is.
        if (candidateScores.isEmpty()) {
            return labels;
        }

        // Sort high to low so we can keep only the strongest algae seeds.
        Collections.sort(candidateScores, Collections.reverseOrder());
        int topCount = Math.max(1, (int) Math.ceil(candidateScores.size() * TOP_YELLOW_FRACTION));
        double cutoffScore = candidateScores.get(topCount - 1);

        // PASS 2:
        // strongest candidates become strong algae 'A'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (candidateMask[r][c] && yellowScores[r][c] >= cutoffScore) {
                    labels[r][c] = 'A';
                }
            }
        }

        // PASS 3:
        // grow outward from strong algae through connected weak algae
        growConnectedAlgae(labels);

        return labels;
    }

    /**
     * Checks whether a Lab pixel passes the relaxed yellow/olive algae threshold.
     *
     * <p>A pixel is considered a candidate if it is bright enough, not too red,
     * and yellow enough.</p>
     *
     * @param l Lab L-channel value, representing brightness
     * @param a Lab a-channel value, representing green-to-red balance
     * @param b Lab b-channel value, representing blue-to-yellow balance
     * @return {@code true} if the pixel is a possible algae candidate;
     *         {@code false} otherwise
     */
    private static boolean passesRelaxedYellowGate(int l, int a, int b) {
        // Bright enough, not too red, and yellow enough.
        return l >= MIN_L
                && a <= MAX_A
                && b >= MIN_B;
    }

    /**
     * Computes a yellow/olive algae score for a Lab pixel.
     *
     * <p>The score rewards yellow pixels using the b channel, penalizes pixels that are
     * too far from the desired a-channel range, and slightly penalizes darker pixels.</p>
     *
     * @param l Lab L-channel value, representing brightness
     * @param a Lab a-channel value, representing green-to-red balance
     * @param b Lab b-channel value, representing blue-to-yellow balance
     * @return algae-likeness score; higher values represent stronger algae candidates
     */
    private static double yellowScore(int l, int a, int b) {
        double score = b - 128.0;

        // Best algae is yellow-ish but not too far off in the a channel.
        score -= 0.80 * Math.abs(a - 129.0);

        // Darker yellow still counts, but gets a small penalty.
        if (l < 60) {
            score -= 0.35 * (60.0 - l);
        }

        return score;
    }

    /**
     * Promotes weak algae pixels connected to strong algae pixels.
     *
     * <p>This method performs a breadth-first search starting from every strong algae pixel
     * labeled {@code 'A'}. Any neighboring weak algae pixel labeled {@code 'a'} is promoted
     * to strong algae and added to the search queue.</p>
     *
     * <p>The search uses an 8-connected neighborhood, meaning diagonal neighbors are included.</p>
     *
     * @param labels label grid containing strong algae {@code 'A'} and weak algae {@code 'a'} pixels
     */
    private static void growConnectedAlgae(char[][] labels) {

        int rows = labels.length;
        int cols = labels[0].length;

        ArrayDeque<int[]> queue = new ArrayDeque<>();

        // Start BFS from every strong algae pixel.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (labels[r][c] == 'A') {
                    queue.add(new int[] { r, c });
                }
            }
        }

        // 8-connected neighborhood
        int[] dr = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dc = { -1, 0, 1, -1, 1, -1, 0, 1 };

        while (!queue.isEmpty()) {

            int[] cur = queue.removeFirst();
            int r = cur[0];
            int c = cur[1];

            for (int k = 0; k < 8; k++) {
                int nr = r + dr[k];
                int nc = c + dc[k];

                if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                    continue;
                }

                // Any weak algae touching strong algae gets promoted.
                if (labels[nr][nc] == 'a') {
                    labels[nr][nc] = 'A';
                    queue.add(new int[] { nr, nc });
                }
            }
        }
    }
}