import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LabelShadow {

    // Tight-ish shadow gate:
    // dark, low-color, close to neutral.
    // The main point is to catch gray shadow without grabbing brown coral.
    private static final int MIN_L = 20;
    private static final int MAX_L = 145;

    private static final int MAX_A_DEVIATION = 7; // how far a* can drift from neutral
    private static final int MAX_B_DEVIATION = 10; // how far b* can drift from neutral
    private static final int MAX_TOTAL_DEVIATION = 14;

    // Only the strongest chunk of shadow-like pixels become seed H pixels first.
    private static final double TOP_SHADOW_FRACTION = 0.20;

    // Shadow should usually sit right next to coral, not randomly elsewhere on the
    // card.
    private static final int CORAL_NEIGHBOR_RADIUS = 2;

    public static char[][] labelShadowPixels(Mat bgrImage, char[][] labels) {

        // Convert to Lab once so brightness and neutral-vs-warm checks are easier.
        Mat lab = new Mat();
        Imgproc.cvtColor(bgrImage, lab, Imgproc.COLOR_BGR2Lab);

        int rows = lab.rows();
        int cols = lab.cols();

        // Save a score per pixel so we can rank the best shadow seeds later.
        double[][] shadowScores = new double[rows][cols];
        boolean[][] candidateMask = new boolean[rows][cols];
        List<Double> candidateScores = new ArrayList<>();

        // Pull the Lab image into a flat byte array for faster looping.
        byte[] data = new byte[(int) (lab.total() * lab.channels())];
        lab.get(0, 0, data);
        lab.release();

        int index = 0;

        // PASS 1:
        // find weak shadow candidates and mark them as 'h'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                int l = data[index] & 0xFF;
                int a = data[index + 1] & 0xFF;
                int b = data[index + 2] & 0xFF;
                index += 3;

                shadowScores[r][c] = Double.NEGATIVE_INFINITY;
                candidateMask[r][c] = false;

                // Never let shadow override coral.
                if (labels[r][c] == 'C' || labels[r][c] == 'c') {
                    continue;
                }

                // Skip anything that does not look shadow-like.
                if (!passesShadowGate(l, a, b)) {
                    continue;
                }

                // Extra safety: if it looks even somewhat brown/coral-like, leave it alone.
                if (looksBrownEnoughToProtect(l, a, b)) {
                    continue;
                }

                double score = shadowScore(l, a, b);

                shadowScores[r][c] = score;
                candidateMask[r][c] = true;
                candidateScores.add(score);

                labels[r][c] = 'h';
            }
        }

        // No shadow-like pixels found.
        if (candidateScores.isEmpty()) {
            return labels;
        }

        // Sort high to low and keep only the strongest shadow seeds.
        Collections.sort(candidateScores, Collections.reverseOrder());
        int topCount = Math.max(1, (int) Math.ceil(candidateScores.size() * TOP_SHADOW_FRACTION));
        double cutoffScore = candidateScores.get(topCount - 1);

        // PASS 2:
        // strongest shadow candidates become strong shadow 'H'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (candidateMask[r][c] && shadowScores[r][c] >= cutoffScore) {
                    labels[r][c] = 'H';
                }
            }
        }

        // PASS 3:
        // grow out from strong shadow through connected weak shadow
        growConnectedShadow(labels);

        // PASS 4:
        // keep only shadow components that are actually near coral
        keepShadowComponentsNearCoral(labels);

        // PASS 5:
        // wipe leftover weak shadow labels
        clearWeakShadowLabels(labels);

        return labels;
    }

    private static boolean passesShadowGate(int l, int a, int b) {

        // Shadow should be in a darker brightness range.
        if (l < MIN_L || l > MAX_L) {
            return false;
        }

        int da = Math.abs(a - 128);
        int db = Math.abs(b - 128);

        // True shadow should stay pretty close to neutral in Lab.
        if (da > MAX_A_DEVIATION || db > MAX_B_DEVIATION) {
            return false;
        }

        if ((da + db) > MAX_TOTAL_DEVIATION) {
            return false;
        }

        // If both warm channels are drifting up, it may be brown rather than shadow.
        if (a >= 130 && b >= 123) {
            return false;
        }

        // Too much yellow also starts looking less like gray shadow.
        if (b >= 132) {
            return false;
        }

        return true;
    }

    private static boolean looksBrownEnoughToProtect(int l, int a, int b) {

        // Broad coral-protection rule:
        // if it is dark-ish and warm enough to plausibly be brown coral,
        // shadow should not claim it.
        if (l > 155) {
            return false;
        }

        if (a >= 130 && b >= 122 && (a - b) <= 18) {
            return true;
        }

        if (a >= 129 && b >= 126 && (a + b) >= 255 && (a - b) <= 18) {
            return true;
        }

        return false;
    }

    private static double shadowScore(int l, int a, int b) {
        int da = Math.abs(a - 128);
        int db = Math.abs(b - 128);

        double score = 0.0;

        // Darker is more shadow-like.
        score += 1.25 * (140.0 - l);

        // Strong preference for low-color, near-neutral pixels.
        score -= 4.00 * da;
        score -= 3.50 * db;

        // Penalize warmth creeping in.
        score -= 1.50 * Math.max(0, a - 128);
        score -= 1.80 * Math.max(0, b - 128);

        return score;
    }

    private static void growConnectedShadow(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        ArrayDeque<int[]> queue = new ArrayDeque<>();

        // Start BFS from every strong shadow pixel.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (labels[r][c] == 'H') {
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

                // Any weak shadow touching strong shadow gets promoted.
                if (labels[nr][nc] == 'h') {
                    labels[nr][nc] = 'H';
                    queue.add(new int[] { nr, nc });
                }
            }
        }
    }

    private static void keepShadowComponentsNearCoral(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        boolean[][] visited = new boolean[rows][cols];
        int[] dr = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dc = { -1, 0, 1, -1, 1, -1, 0, 1 };

        // Walk through each strong shadow component.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                if (labels[r][c] != 'H' || visited[r][c]) {
                    continue;
                }

                List<int[]> component = new ArrayList<>();
                ArrayDeque<int[]> queue = new ArrayDeque<>();
                queue.add(new int[] { r, c });
                visited[r][c] = true;

                boolean nearCoral = false;

                while (!queue.isEmpty()) {
                    int[] cur = queue.removeFirst();
                    int cr = cur[0];
                    int cc = cur[1];

                    component.add(cur);

                    // If any part of this component is near coral, keep the whole thing.
                    if (isNearCoral(labels, cr, cc, CORAL_NEIGHBOR_RADIUS)) {
                        nearCoral = true;
                    }

                    for (int k = 0; k < 8; k++) {
                        int nr = cr + dr[k];
                        int nc = cc + dc[k];

                        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                            continue;
                        }

                        if (visited[nr][nc]) {
                            continue;
                        }

                        if (labels[nr][nc] != 'H') {
                            continue;
                        }

                        visited[nr][nc] = true;
                        queue.add(new int[] { nr, nc });
                    }
                }

                // If the whole component is floating away from coral, drop it.
                if (!nearCoral) {
                    for (int[] px : component) {
                        labels[px[0]][px[1]] = '.';
                    }
                }
            }
        }
    }

    private static boolean isNearCoral(char[][] labels, int r, int c, int radius) {
        int rows = labels.length;
        int cols = labels[0].length;

        // Look in a small square neighborhood for any coral label.
        for (int rr = Math.max(0, r - radius); rr <= Math.min(rows - 1, r + radius); rr++) {
            for (int cc = Math.max(0, c - radius); cc <= Math.min(cols - 1, c + radius); cc++) {
                if (labels[rr][cc] == 'C' || labels[rr][cc] == 'c') {
                    return true;
                }
            }
        }

        return false;
    }

    private static void clearWeakShadowLabels(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        // Any weak shadow left over at the end gets cleared out.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (labels[r][c] == 'h') {
                    labels[r][c] = '.';
                }
            }
        }
    }
}