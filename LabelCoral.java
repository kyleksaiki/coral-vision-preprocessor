import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LabelCoral {

    // Tight Lab range for dark brown coral.
    // Goal here is to keep real brown coral and avoid the pink/red rim.
    private static final int MAX_L = 140;
    private static final int MIN_A = 128;
    private static final int MAX_A = 148;
    private static final int MIN_B = 112;
    private static final int MAX_B = 150;

    // Only the strongest chunk of coral-like pixels become seed coral at first.
    private static final double TOP_BROWN_FRACTION = 0.20;

    // Rule for keeping multiple coral components:
    // if the next biggest ones are close enough in size, keep them too.
    private static final double SECOND_BIGGEST_MIN_RATIO = 0.80;
    private static final double THIRD_BIGGEST_MIN_RATIO = 0.70;

    public static char[][] labelCoralPixels(Mat bgrImage, char[][] labels) {
        // Convert once to Lab so brown-vs-red/yellow separation is easier.
        Mat lab = new Mat();
        Imgproc.cvtColor(bgrImage, lab, Imgproc.COLOR_BGR2Lab);

        int rows = lab.rows();
        int cols = lab.cols();

        // Store a score for each candidate pixel so we can rank them later.
        double[][] brownScores = new double[rows][cols];
        boolean[][] candidateMask = new boolean[rows][cols];
        List<Double> candidateScores = new ArrayList<>();

        // Read the Lab image into one flat byte array for faster looping.
        byte[] data = new byte[(int) (lab.total() * lab.channels())];
        lab.get(0, 0, data);
        lab.release();

        int index = 0;

        // PASS 1:
        // find coral candidates and mark them as weak coral 'c'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int l = data[index] & 0xFF;
                int a = data[index + 1] & 0xFF;
                int b = data[index + 2] & 0xFF;
                index += 3;

                brownScores[r][c] = Double.NEGATIVE_INFINITY;
                candidateMask[r][c] = false;

                // Skip anything outside the brown coral gate.
                if (!passesBrownGate(l, a, b)) {
                    continue;
                }

                double score = brownScore(l, a, b);

                brownScores[r][c] = score;
                candidateMask[r][c] = true;
                candidateScores.add(score);

                labels[r][c] = 'c';
            }
        }

        // No coral-like pixels found.
        if (candidateScores.isEmpty()) {
            return labels;
        }

        // Sort high to low and keep only the strongest part as coral seeds.
        Collections.sort(candidateScores, Collections.reverseOrder());
        int topCount = Math.max(1, (int) Math.ceil(candidateScores.size() * TOP_BROWN_FRACTION));
        double cutoffScore = candidateScores.get(topCount - 1);

        // PASS 2:
        // strongest brown candidates become strong coral 'C'
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (candidateMask[r][c] && brownScores[r][c] >= cutoffScore) {
                    labels[r][c] = 'C';
                }
            }
        }

        // PASS 3:
        // grow outward from strong coral through connected weak coral
        growConnectedCoral(labels);

        // PASS 4:
        // clean up by keeping only the main coral component(s)
        keepMainCoralComponents(labels);

        return labels;
    }

    private static boolean passesBrownGate(int l, int a, int b) {
        // Coral should be on the darker side.
        if (l > MAX_L) {
            return false;
        }

        // Needs enough warmth to be brown,
        // but not so much that it drifts into pink/red rim territory.
        if (a < MIN_A || a > MAX_A) {
            return false;
        }

        // Needs some yellow support to be brown,
        // but not too much or it starts looking like algae.
        if (b < MIN_B || b > MAX_B) {
            return false;
        }

        // Extra rim rejection:
        // if a is much higher than b, it is getting too reddish.
        if ((a - b) > 18) {
            return false;
        }

        return true;
    }

    private static double brownScore(int l, int a, int b) {
        double score = 0.0;

        // Darker pixels are favored for coral.
        score += 1.10 * (128.0 - l);

        // More warmth helps, as long as it stayed inside the gate above.
        score += 1.00 * (a - 128.0);

        // Best coral is near a middle brown-yellow,
        // not too dull and not too yellow.
        score -= 0.65 * Math.abs(b - 134.0);

        return score;
    }

    private static void growConnectedCoral(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        ArrayDeque<int[]> queue = new ArrayDeque<>();

        // Start BFS from every strong coral pixel.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (labels[r][c] == 'C') {
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

                // Any weak coral touching strong coral gets promoted.
                if (labels[nr][nc] == 'c') {
                    labels[nr][nc] = 'C';
                    queue.add(new int[] { nr, nc });
                }
            }
        }
    }

    private static void keepMainCoralComponents(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        boolean[][] visited = new boolean[rows][cols];
        List<List<int[]>> components = new ArrayList<>();

        int[] dr = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dc = { -1, 0, 1, -1, 1, -1, 0, 1 };

        // Find connected components made of strong coral pixels only.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (labels[r][c] != 'C' || visited[r][c]) {
                    continue;
                }

                List<int[]> component = new ArrayList<>();
                ArrayDeque<int[]> queue = new ArrayDeque<>();
                queue.add(new int[] { r, c });
                visited[r][c] = true;

                while (!queue.isEmpty()) {
                    int[] cur = queue.removeFirst();
                    int cr = cur[0];
                    int cc = cur[1];
                    component.add(cur);

                    for (int k = 0; k < 8; k++) {
                        int nr = cr + dr[k];
                        int nc = cc + dc[k];

                        if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) {
                            continue;
                        }

                        if (visited[nr][nc]) {
                            continue;
                        }

                        if (labels[nr][nc] != 'C') {
                            continue;
                        }

                        visited[nr][nc] = true;
                        queue.add(new int[] { nr, nc });
                    }
                }

                components.add(component);
            }
        }

        // If no strong coral survived, wipe coral labels and stop.
        if (components.isEmpty()) {
            clearAllCoralLabels(labels);
            return;
        }

        // Sort biggest component first.
        components.sort((a, b) -> Integer.compare(b.size(), a.size()));

        int biggestSize = components.get(0).size();
        boolean keepThree = false;

        // Only keep the next two if they are close enough in size to the biggest one.
        if (components.size() >= 3) {
            int secondSize = components.get(1).size();
            int thirdSize = components.get(2).size();

            keepThree = secondSize >= Math.ceil(biggestSize * SECOND_BIGGEST_MIN_RATIO) &&
                    thirdSize >= Math.ceil(biggestSize * THIRD_BIGGEST_MIN_RATIO);
        }

        // Clear out all coral labels before writing back the kept components.
        clearAllCoralLabels(labels);

        // Keep either just the main component, or the top three.
        int componentsToKeep = keepThree ? 3 : 1;

        for (int i = 0; i < componentsToKeep && i < components.size(); i++) {
            for (int[] pixel : components.get(i)) {
                labels[pixel[0]][pixel[1]] = 'C';
            }
        }
    }

    private static void clearAllCoralLabels(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        // Reset both weak and strong coral labels back to empty.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (labels[r][c] == 'C' || labels[r][c] == 'c') {
                    labels[r][c] = '.';
                }
            }
        }
    }
}