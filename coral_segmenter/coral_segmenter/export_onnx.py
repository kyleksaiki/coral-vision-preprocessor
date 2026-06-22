"""Export the trained coral model to ONNX for the Java app (CoralFinderOnnx.java).

Run:  python export_onnx.py --ckpt checkpoints/best.pt --out coral_seg.onnx

The exported input size is whatever you trained with (config img_height/img_width).
It MUST match IN_W / IN_H in CoralFinderOnnx.java. With the default config that's
1 x 3 x 768 x 768."""

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
    print(f"-> set CoralFinderOnnx.java IN_W={w}, IN_H={h} to match.")

    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
        t0 = time.time()
        runs = 10
        for _ in range(runs):
            model(dummy)
        ms = (time.time() - t0) / runs * 1000
    print(f"Approx PyTorch CPU latency: {ms:.0f} ms/crop (onnxruntime is usually faster).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--out", default="coral_seg.onnx")
    args = ap.parse_args()
    run(args.ckpt, args.out)
