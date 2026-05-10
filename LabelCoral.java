import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

public class LabelCoral {

    // Only connected coral blobs with at least this many pixels are kept.
    private static final int MIN_COMPONENT_PIXELS = 9000;

    // ============================================================
    // ACCEPTABLE CORAL COLOR TUNING
    // Tuning was Ai generated
    // ============================================================

    // Light / medium brown coral.
    private static final int WARM_L_MAX = 195;
    private static final int WARM_V_MAX = 238;
    private static final int WARM_A_MIN = 129;
    private static final int WARM_A_MAX = 166;
    private static final int WARM_B_MIN = 122;
    private static final int WARM_B_MAX = 170;
    private static final int WARM_A_MINUS_B_MAX = 38;
    private static final int WARM_S_MIN = 12;

    // Dark brown coral.
    private static final int DARK_L_MAX = 140;
    private static final int DARK_V_MAX = 170;
    private static final int DARK_A_MIN = 124;
    private static final int DARK_B_MIN = 108;
    private static final int DARK_B_MAX = 162;
    private static final int DARK_A_MINUS_B_MAX = 38;
    private static final int DARK_A_PLUS_B_MIN = 232;

    // Black-brown coral.
    // Lower these if very black coral is still being missed.
    private static final int BLACK_BROWN_L_MAX = 95;
    private static final int BLACK_BROWN_V_MAX = 125;
    private static final int BLACK_BROWN_A_MIN = 119;
    private static final int BLACK_BROWN_B_MIN = 102;
    private static final int BLACK_BROWN_B_MAX = 158;
    private static final int BLACK_BROWN_A_MINUS_B_MAX = 42;
    private static final int BLACK_BROWN_A_PLUS_B_MIN = 224;

    // Extra BGR support for black-brown coral.
    private static final int BLACK_BROWN_RED_MINUS_BLUE_MIN = -8;
    private static final int BLACK_BROWN_GREEN_MINUS_BLUE_MIN = -22;
    private static final int BLACK_BROWN_RED_MINUS_GREEN_MIN = -22;

    // Brown hue support.
    private static final int BROWN_H_LOW_MAX = 36;
    private static final int BROWN_H_HIGH_MIN = 165;
    private static final int BROWN_LOW_S_MAX = 28;

    // BGR check
    private static final int BGR_BROWN_RED_MINUS_GREEN_MIN = -14;
    private static final int BGR_BROWN_RED_MINUS_BLUE_MIN = -2;
    private static final int BGR_BROWN_GREEN_MINUS_BLUE_MIN = -26;

    // ============================================================
    // REJECTION TUNING
    // ============================================================

    // Green / olive algae rejection.
    private static final int GREEN_DARK_L_MAX = 90;
    private static final int GREEN_DARK_V_MAX = 115;

    private static final int LAB_GREEN_A_MAX = 124;
    private static final int LAB_GREEN_S_MIN = 12;
    private static final int LAB_GREEN_L_MIN = 35;

    private static final int HSV_GREEN_H_MIN = 35;
    private static final int HSV_GREEN_H_MAX = 105;
    private static final int HSV_GREEN_S_MIN = 16;

    private static final int BGR_GREEN_GREEN_MINUS_RED_MIN = 7;
    private static final int BGR_GREEN_GREEN_MINUS_BLUE_MIN = 4;

    private static final int OLIVE_A_MAX = 127;
    private static final int OLIVE_B_MIN = 124;
    private static final int OLIVE_H_MIN = 32;
    private static final int OLIVE_H_MAX = 95;
    private static final int OLIVE_S_MIN = 12;

    // Bright white tray / tag rejection.
    private static final int TRAY_L_MIN = 210;
    private static final int TRAY_V_MIN = 225;
    private static final int TRAY_S_MAX = 60;

    // Bright red/orange junk rejection, such as ruler / tags.
    private static final int JUNK_L_MIN = 165;
    private static final int JUNK_V_MIN = 235;
    private static final int JUNK_A_MIN = 148;
    private static final int JUNK_B_MIN = 140;
    private static final int JUNK_H_LOW_MAX = 24;
    private static final int JUNK_H_HIGH_MIN = 170;
    private static final int JUNK_S_MIN = 70;

    public static char[][] labelCoralPixels(Mat bgrImage, char[][] labels) {

        Mat lab = new Mat();
        Imgproc.cvtColor(bgrImage, lab, Imgproc.COLOR_BGR2Lab);

        Mat hsv = new Mat();
        Imgproc.cvtColor(bgrImage, hsv, Imgproc.COLOR_BGR2HSV);

        int rows = bgrImage.rows();
        int cols = bgrImage.cols();
        int bgrChannels = bgrImage.channels();

        byte[] bgrData = new byte[(int) (bgrImage.total() * bgrChannels)];
        byte[] labData = new byte[(int) (lab.total() * lab.channels())];
        byte[] hsvData = new byte[(int) (hsv.total() * hsv.channels())];

        bgrImage.get(0, 0, bgrData);
        lab.get(0, 0, labData);
        hsv.get(0, 0, hsvData);

        lab.release();
        hsv.release();

        int bgrIndex = 0;
        int labIndex = 0;
        int hsvIndex = 0;

        // Directly label strong coral.
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {

                int blue = bgrData[bgrIndex] & 0xFF;
                int green = bgrData[bgrIndex + 1] & 0xFF;
                int red = bgrData[bgrIndex + 2] & 0xFF;

                int l = labData[labIndex] & 0xFF;
                int a = labData[labIndex + 1] & 0xFF;
                int b = labData[labIndex + 2] & 0xFF;

                int h = hsvData[hsvIndex] & 0xFF;
                int s = hsvData[hsvIndex + 1] & 0xFF;
                int v = hsvData[hsvIndex + 2] & 0xFF;

                if (isCoralBrown(l, a, b, h, s, v, blue, green, red)) {
                    labels[r][c] = 'C';
                }

                bgrIndex += bgrChannels;
                labIndex += 3;
                hsvIndex += 3;
            }
        }

        // PASS 2:
        // Remove tiny false-positive coral components.
        filterSmallCoralComponents(labels);

        return labels;
    }

    private static boolean isCoralBrown(
            int l, int a, int b,
            int h, int s, int v,
            int blue, int green, int red) {

        if (isGreenishAlgaeLike(l, a, b, h, s, v, blue, green, red)) {
            return false;
        }

        if (isBrightTrayLike(l, s, v)) {
            return false;
        }

        if (isBrightRedOrangeJunk(l, a, b, h, s, v)) {
            return false;
        }

        boolean warmBrownLab = l <= WARM_L_MAX &&
                v <= WARM_V_MAX &&
                a >= WARM_A_MIN &&
                a <= WARM_A_MAX &&
                b >= WARM_B_MIN &&
                b <= WARM_B_MAX &&
                (a - b) <= WARM_A_MINUS_B_MAX &&
                s >= WARM_S_MIN;

        boolean darkBrownLab = l <= DARK_L_MAX &&
                v <= DARK_V_MAX &&
                a >= DARK_A_MIN &&
                b >= DARK_B_MIN &&
                b <= DARK_B_MAX &&
                (a - b) <= DARK_A_MINUS_B_MAX &&
                (a + b) >= DARK_A_PLUS_B_MIN;

        boolean blackBrownLab = l <= BLACK_BROWN_L_MAX &&
                v <= BLACK_BROWN_V_MAX &&
                a >= BLACK_BROWN_A_MIN &&
                b >= BLACK_BROWN_B_MIN &&
                b <= BLACK_BROWN_B_MAX &&
                (a - b) <= BLACK_BROWN_A_MINUS_B_MAX &&
                (a + b) >= BLACK_BROWN_A_PLUS_B_MIN;

        boolean blackBrownBgr = v <= BLACK_BROWN_V_MAX &&
                (red - blue) >= BLACK_BROWN_RED_MINUS_BLUE_MIN &&
                (green - blue) >= BLACK_BROWN_GREEN_MINUS_BLUE_MIN &&
                (red - green) >= BLACK_BROWN_RED_MINUS_GREEN_MIN;

        boolean brownHue = h <= BROWN_H_LOW_MAX ||
                h >= BROWN_H_HIGH_MIN ||
                s < BROWN_LOW_S_MAX;

        boolean bgrBrown = (red - green) >= BGR_BROWN_RED_MINUS_GREEN_MIN &&
                (red - blue) >= BGR_BROWN_RED_MINUS_BLUE_MIN &&
                (green - blue) >= BGR_BROWN_GREEN_MINUS_BLUE_MIN;

        return ((warmBrownLab || darkBrownLab) && (brownHue || bgrBrown))
                || (blackBrownLab && blackBrownBgr);
    }

    private static boolean isGreenishAlgaeLike(
            int l, int a, int b,
            int h, int s, int v,
            int blue, int green, int red) {

        boolean darkPixel = l <= GREEN_DARK_L_MAX || v <= GREEN_DARK_V_MAX;

        boolean labGreen = !darkPixel &&
                a <= LAB_GREEN_A_MAX &&
                s >= LAB_GREEN_S_MIN &&
                l >= LAB_GREEN_L_MIN;

        boolean hsvGreen = !darkPixel &&
                h >= HSV_GREEN_H_MIN &&
                h <= HSV_GREEN_H_MAX &&
                s >= HSV_GREEN_S_MIN;

        boolean bgrGreen = (green - red) > BGR_GREEN_GREEN_MINUS_RED_MIN &&
                (green - blue) > BGR_GREEN_GREEN_MINUS_BLUE_MIN;

        boolean oliveGreen = !darkPixel &&
                a <= OLIVE_A_MAX &&
                b >= OLIVE_B_MIN &&
                h >= OLIVE_H_MIN &&
                h <= OLIVE_H_MAX &&
                s >= OLIVE_S_MIN;

        return labGreen || hsvGreen || bgrGreen || oliveGreen;
    }

    private static boolean isBrightTrayLike(int l, int s, int v) {
        return l >= TRAY_L_MIN &&
                v >= TRAY_V_MIN &&
                s <= TRAY_S_MAX;
    }

    private static boolean isBrightRedOrangeJunk(int l, int a, int b, int h, int s, int v) {
        boolean bright = l >= JUNK_L_MIN ||
                v >= JUNK_V_MIN;

        boolean veryWarm = a >= JUNK_A_MIN &&
                b >= JUNK_B_MIN;

        boolean redOrangeHue = h <= JUNK_H_LOW_MAX ||
                h >= JUNK_H_HIGH_MIN;

        boolean saturated = s >= JUNK_S_MIN;

        return bright && veryWarm && redOrangeHue && saturated;
    }

    private static void filterSmallCoralComponents(char[][] labels) {
        int rows = labels.length;
        int cols = labels[0].length;

        boolean[][] visited = new boolean[rows][cols];

        int[] dr = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dc = { -1, 0, 1, -1, 1, -1, 0, 1 };

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

                if (component.size() < MIN_COMPONENT_PIXELS) {
                    for (int[] px : component) {
                        labels[px[0]][px[1]] = '.';
                    }
                }
            }
        }
    }
}
