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
 * Runs the coral-segmentation model (PyTorch U-Net exported to ONNX) inside Java via ONNX
 * Runtime. Same shape of class as {@link CardFinderOnnx}, but for coral.
 *
 * <p><b>Two differences from the card finder, both important:</b></p>
 * <ul>
 *   <li>It is meant to be run on a <b>single card crop of the CLAHE image</b>, not the whole
 *       tray. {@link TrayCleaner} crops each card the card model found and calls this per crop,
 *       so the coral fills more of the model's input and its fine texture survives the resize.</li>
 *   <li>It returns a {@code CV_8UC1} mask where {@code 255 = coral} (at the crop's resolution).</li>
 * </ul>
 *
 * <p><b>Input size MUST match the coral model's training config.</b> When you export the coral
 * model, whatever {@code A.Resize(height, width)} you trained with has to be reflected in
 * {@link #IN_W} / {@link #IN_H} below, exactly like the card model. The defaults here are a
 * square-ish size that suits a single card; change them to match your config.yaml.</p>
 *
 * <p><b>Usage</b> (create ONCE at startup, reuse for every crop):</p>
 * <pre>
 *   CoralFinderOnnx coral = new CoralFinderOnnx("coral_seg.onnx");
 *   Mat coralMaskForCrop = coral.findCoral(claheCardCrop);   // 255 = coral
 *   // ... at shutdown: coral.close();
 * </pre>
 */
public class CoralFinderOnnx implements AutoCloseable {

    /**
     * Model input size. MUST match img_width / img_height the coral model trained with
     * (Python A.Resize(height, width)). These are placeholders sized for a single card crop;
     * set them to your real coral config before trusting the output.
     */
    private static final int IN_W = 1024;
    private static final int IN_H = 1024;

    /** ImageNet normalization — MUST match the coral model's dataset.py (mean/std in R,G,B). */
    private static final float[] MEAN = {0.485f, 0.456f, 0.406f};
    private static final float[] STD  = {0.229f, 0.224f, 0.225f};

    private final OrtEnvironment env;
    private final OrtSession session;
    private final String inputName;

    /**
     * @param onnxModelPath path to the exported coral model (e.g. coral_seg.onnx)
     * @throws OrtException if the model cannot be loaded
     */
    public CoralFinderOnnx(String onnxModelPath) throws OrtException {
        env = OrtEnvironment.getEnvironment();
        session = env.createSession(onnxModelPath, new OrtSession.SessionOptions());
        inputName = session.getInputNames().iterator().next();
    }

    /**
     * Predict the coral mask for one BGR image (intended to be a CLAHE card crop).
     *
     * @param bgr BGR image (a single card crop of the normalized/CLAHE tray)
     * @return CV_8UC1 mask at the input image's resolution (255 = coral, 0 = not)
     * @throws OrtException if inference fails
     */
    public Mat findCoral(Mat bgr) throws OrtException {
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

        // 4) Upscale the mask back to the crop's resolution (nearest keeps it binary).
        Mat small = new Mat(IN_H, IN_W, CvType.CV_8UC1);
        small.put(0, 0, maskBytes);
        Mat full = new Mat();
        Imgproc.resize(small, full, new Size(origW, origH), 0, 0, Imgproc.INTER_NEAREST);
        small.release();
        return full; // CV_8UC1, 255 = coral
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}