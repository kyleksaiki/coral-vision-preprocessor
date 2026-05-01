import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LabelSilt {

    // Loose-ish Lab gate for removable pink / red / orange / purple junk.
    // Main job is to catch the warm stuff without pulling in the dark coral.
    private static final int MAX_L = 210;
    private static final int MIN_A = 132;
    private static final int MIN_B = 118;

    // Only the strongest chunk of candidates start as strong silt seeds.
    private static final double TOP_SILT_FRACTION = 0.20;

    public static char[][] labelSiltPixels(Mat bgrImage, char[][] labels) {
        // Convert once to Lab so color thresholds are easier to reason about.
        Mat lab = new Mat();
        Imgproc.cvtColor(bgrImage, lab, Imgproc.COLOR_BGR2Lab);

        int rows = lab.rows();
        int cols = lab.cols();

        // Per-pixel score and mask so we can rank candidates before promoting them.
        double[][] siltScores = new double[rows][cols];
        boolean[][] candidateMask = new boolean[rows][cols];
        List<Double> candidateScores = new ArrayList<>();

        // Pull image data into one flat array for faster looping.
        byte[] data = new byte[(int) (lab.total() * lab.channels())];
        lab.get(0, 0, data);
        lab.release();

        int index = 0;

        // PASS 1:
        // mark possible silt pixels as weak silt 's'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                int l = data[index] & 0xFF;
                int a = data[index + 1] & 0xFF;
                int b = data[index + 2] & 0xFF;
                index += 3;

                siltScores[r][c] = Double.NEGATIVE_INFINITY;
                candidateMask[r][c] = false;

                // Coral stays protected.
                if (labels[r][c] == 'C' || labels[r][c] == 'c') {
                    continue;
                }

                // Skip anything that does not look silt-like.
                if (!passesSiltGate(l, a, b)) {
                    continue;
                }

                double score = siltScore(l, a, b);

                siltScores[r][c] = score;
                candidateMask[r][c] = true;
                candidateScores.add(score);

                labels[r][c] = 's';
            }
        }

        // No candidates, nothing to do.
        if (candidateScores.isEmpty()) {
            return labels;
        }

        // Sort high to low and keep only the strongest part as seed S pixels.
        Collections.sort(candidateScores, Collections.reverseOrder());
        int topCount = Math.max(1, (int) Math.ceil(candidateScores.size() * TOP_SILT_FRACTION));
        double cutoffScore = candidateScores.get(topCount - 1);

        // PASS 2:
        // strongest removable pixels become strong silt 'S'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (candidateMask[r][c] && siltScores[r][c] >= cutoffScore) {
                    labels[r][c] = 'S';
                }
            }
        }

        // PASS 3:
        // grow outward from strong silt through connected weak silt
        growConnectedSilt(labels);

        return labels;
    }

    private static boolean passesSiltGate(int l, int a, int b) {
        // Ignore very dark stuff first so coral does not get pulled in.
        if (l < 55) {
            return false;
        }

        // Warm / pink / red / orange / purple directions usually push a upward.
        if (a < MIN_A) {
            return false;
        }

        // Keep some warmth in b too so pink/orange/red still pass.
        // Purple can sit a bit lower, so this stays fairly loose.
        if (b < 108) {
            return false;
        }

        // If it is dark and also pretty brown-like, leave it alone.
        // This is mainly to protect real coral.
        if (l < 110 && a <= 148 && b <= 145) {
            return false;
        }

        return true;
    }

    private static double siltScore(int l, int a, int b) {
        double score = 0.0;

        // Main signal: warm / red / pink / magenta direction.
        score += 1.20 * (a - 128.0);

        // Add some support from yellow/orange too.
        score += 0.35 * (b - 120.0);

        // Slightly prefer lighter removable stuff over dark coral.
        score += 0.45 * (l - 90.0);

        return score;
    }

    private static void growConnectedSilt(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        ArrayDeque<int[]> queue = new ArrayDeque<>();

        // Start BFS from every strong silt pixel.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (labels[r][c] == 'S') {
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

                // Any weak silt touching strong silt gets promoted.
                if (labels[nr][nc] == 's') {
                    labels[nr][nc] = 'S';
                    queue.add(new int[] { nr, nc });
                }
            }
        }
    }
}