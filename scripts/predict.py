#!/usr/bin/env python
"""Classify one or more images with the exported ONNX model (mirrors what the browser does).

    python scripts/predict.py photo.jpg [more.jpg ...] [--model export/pinniped.onnx]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EXPORT_DIR  # noqa: E402


def preprocess(im: Image.Image, meta: dict) -> np.ndarray:
    """Pure-numpy/PIL implementation of model_meta.json's preprocessing, independent of torch."""
    im = im.convert(meta["input"]["color_space"])
    w, h = im.size
    s = meta["resize_shorter_side"] / min(w, h)
    im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    c = meta["crop_size"]
    left, top = (im.width - c) // 2, (im.height - c) // 2
    im = im.crop((left, top, left + c, top + c))
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array(meta["mean"], dtype=np.float32)) / np.array(meta["std"], dtype=np.float32)
    return x.transpose(2, 0, 1)[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    meta = json.loads((EXPORT_DIR / "model_meta.json").read_text())
    path = args.model or str(EXPORT_DIR / meta["recommended_file"])
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for p in args.images:
        with Image.open(p) as im:
            x = preprocess(im, meta)
        logits = sess.run(None, {meta["input"]["name"]: x})[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = int(probs.argmax())
        label = meta["classes"][top]
        if label in meta["pinniped_classes"] and probs[top] < meta["threshold"]:
            label = "unsure"
        print(f"{p}: {label} ({probs[top]:.1%})  " + " ".join(f"{c}={v:.3f}" for c, v in zip(meta["classes"], probs)))


if __name__ == "__main__":
    main()
