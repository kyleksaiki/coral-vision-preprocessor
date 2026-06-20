"""Export the trained model to ONNX so it can run without PyTorch (e.g. in a Java
service via onnxruntime, or anywhere portable).

Run:  python export_onnx.py --ckpt checkpoints/best.pt --out card_seg.onnx"""

import argparse
import time

import torch

from model import load_model


def run(ckpt, out_path):
    model, cfg = load_model(ckpt, "cpu")
    h, w = cfg["train"]["img_height"], cfg["train"]["img_width"]
    dummy = torch.randn(1, 3, h, w)

    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX to {out_path}  (input 1x3x{h}x{w})")

    # Rough CPU latency, averaged over a few runs.
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        t0 = time.time()
        runs = 10
        for _ in range(runs):
            model(dummy)
        ms = (time.time() - t0) / runs * 1000
    print(f"Approx PyTorch CPU latency: {ms:.0f} ms/image "
          f"(GPU is typically 10-30x faster; onnxruntime is usually faster than this on CPU).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="card_seg.onnx")
    args = ap.parse_args()
    run(args.ckpt, args.out)
