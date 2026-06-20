import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.nio.FloatBuffer;
import java.util.Collections;

/**
 * Runs the PyTorch-trained card-segmentation model (exported to ONNX) directly inside
 * Java via ONNX Runtime. No Python at runtime.
 *
 * <p>This is a drop-in replacement for {@code CardFinder.findCards(Mat)}: it takes a BGR
 * tray image and returns a {@code CV_8UC1} mask (255 = card, 0 = not), exactly the same
 * contract, so the rest of the pipeline (rendering, coral/algae labeling) is unchanged.</p>
 *
 * <p><b>Important:</b> feed it the SAME kind of image the model was trained on — the RAW
 * photo, not the CLAHE/white-balanced one.</p>
 *
 * <p><b>Setup:</b> add the ONNX Runtime jar to your classpath (like you did for OpenCV):
 * Maven coordinates {@code com.microsoft.onnxruntime:onnxruntime:1.26.0}, or download the
 * jar from Maven Central. The jar bundles native libraries for Windows/macOS/Linux and
 * extracts them at runtime, so usually just having it on the classpath is enough.</p>
 *
 * <p><b>Usage:</b> create ONE instance at GUI startup (loading the model is the slow part)
 * and reuse it for every photo:</p>
 * <pre>
 *   CardFinderOnnx finder = new CardFinderOnnx("card_seg.onnx");
 *   Mat cardMask = finder.findCards(rawBgr);
 *   Mat overlay  = CardFinder.renderCardOverlay(rawBgr, cardMask);
 *   // ... when the app closes: finder.close();
 * </pre>
 */
public class CardFinderOnnx implements AutoCloseable {

    /** Model input size. Must match img_width / img_height in the Python config.yaml. */
    private static final int IN_W = 1024;
    private static final int IN_H = 768;

    /** ImageNet normalization — MUST match dataset.py (mean/std are in R, G, B order). */
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD  = {0.229f, 0.224f, 0.225f};

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    /**
     * @param onnxModelPath path to the exported model (card_seg.onnx)
     * @throws OrtException if the model cannot be loaded
     */
    public CardFinderOnnx(String onnxModelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(onnxModelPath, new OrtSession.SessionOptions());
        inputName = session.getInputNames().iterator().next();
    }

    /**
     * Predict the card mask for one BGR image.
     *
     * @param bgr raw tray image (OpenCV BGR)
     * @return CV_8UC1 mask at the input image's resolution (255 = card, 0 = not)
     * @throws OrtException if inference fails
     */
    public Mat findCards(Mat bgr) throws OrtException {
        int origW = bgr.cols();
        int origH = bgr.rows();

        // 1) Resize to the model's input size (matches Python A.Resize(height, width)).
        Mat resized = new Mat();
        Imgproc.resize(bgr, resized, new Size(IN_W, IN_H));

        // 2) Build a normalized CHW float tensor in RGB order
        //    (matches BGR->RGB + Normalize + ToTensorV2 in dataset.py).
        byte[] px = new byte[(int) (resized.total() * resized.channels())];
        resized.get(0, 0, px);
        resized.release();

        float[] chw = new float[3 * IN_H * IN_W];
        int plane = IN_H * IN_W;
        int p = 0;
        for (int y = 0; y < IN_H; y++) {
            for (int x = 0; x < IN_W; x++) {
                int b = px[p]     & 0xFF;
                int g = px[p + 1] & 0xFF;
                int r = px[p + 2] & 0xFF;
                p += 3;
                int idx = y * IN_W + x;
                chw[idx]             = ((r / 255f) - MEAN[0]) / STD[0]; // R plane
                chw[plane + idx]     = ((g / 255f) - MEAN[1]) / STD[1]; // G plane
                chw[2 * plane + idx] = ((b / 255f) - MEAN[2]) / STD[2]; // B plane
            }
        }

        // 3) Run the model and threshold the logits.
        //    sigmoid(logit) > 0.5  is the same as  logit > 0, so we skip the sigmoid.
        long[] shape = {1, 3, IN_H, IN_W};
        byte[] maskBytes = new byte[IN_H * IN_W];
        try (OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(chw), shape);
             OrtSession.Result result = session.run(Collections.singletonMap(inputName, input))) {
            float[][][][] logits = (float[][][][]) result.get(0).getValue(); // [1][1][H][W]
            float[][] m = logits[0][0];
            int k = 0;
            for (int y = 0; y < IN_H; y++) {
                for (int x = 0; x < IN_W; x++) {
                    maskBytes[k++] = (byte) (m[y][x] > 0f ? 255 : 0);
                }
            }
        }

        // 4) Upscale the mask back to the original resolution (nearest keeps it binary).
        Mat small = new Mat(IN_H, IN_W, CvType.CV_8UC1);
        small.put(0, 0, maskBytes);
        Mat full = new Mat();
        Imgproc.resize(small, full, new Size(origW, origH), 0, 0, Imgproc.INTER_NEAREST);
        small.release();
        return full; // CV_8UC1, 255 = card  — same contract as CardFinder.findCards
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
