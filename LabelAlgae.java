import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LabelAlgae {

    // Simple Lab thresholds for the first algae pass.
    // This is aimed at yellow / olive-ish stuff, not coral.
    private static final int MIN_L = 50;
    private static final int MAX_A = 145;
    private static final int MIN_B = 132;

    // Only the strongest chunk of algae-like pixels become seed A pixels at first.
    private static final double TOP_YELLOW_FRACTION = 0.20;

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

    private static boolean passesRelaxedYellowGate(int l, int a, int b) {
        // Bright enough, not too red, and yellow enough.
        return l >= MIN_L
                && a <= MAX_A
                && b >= MIN_B;
    }

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