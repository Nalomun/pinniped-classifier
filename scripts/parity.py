#!/usr/bin/env python
"""Parity test: PyTorch vs ONNX Runtime on the same test images.

Picks a few test images per class (deterministic), copies them to reports/parity_samples/
(they are CC0/CC-BY; attribution is written alongside), runs them through
  * the PyTorch checkpoint
  * export/pinniped.onnx (fp32)
  * export/pinniped_int8.onnx (if it exists)
using exactly common.eval_transform(), and reports the max |Δprob| and argmax agreement.
reports/parity.json stores the per-image probabilities so the browser implementation can be
checked against the same images (Phase 2).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CLASSES, EXPORT_DIR, IMAGES_DIR, OUTPUTS_DIR, REPORTS_DIR, eval_transform, load_manifest  # noqa: E402
from eval import load_checkpoint  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUTPUTS_DIR / "best.pt"))
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--atol", type=float, default=1e-3, help="max allowed |Δprob| torch vs fp32 onnx")
    args = ap.parse_args()

    sample_dir = REPORTS_DIR / "parity_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    test_df = load_manifest("test")
    rows = (test_df.sort_values("photo_id").groupby("class", sort=True).head(args.per_class)).to_dict("records")

    model, _ = load_checkpoint(Path(args.checkpoint), "cpu")
    tf = eval_transform()
    sessions = {"onnx_fp32": ort.InferenceSession(str(EXPORT_DIR / "pinniped.onnx"), providers=["CPUExecutionProvider"])}
    if (EXPORT_DIR / "pinniped_int8.onnx").exists():
        sessions["onnx_int8"] = ort.InferenceSession(str(EXPORT_DIR / "pinniped_int8.onnx"), providers=["CPUExecutionProvider"])

    out, worst = [], {k: 0.0 for k in sessions}
    for r in rows:
        src = IMAGES_DIR / r["local_path"]
        dst = sample_dir / f"{r['class']}_{r['photo_id']}{src.suffix}"
        shutil.copy(src, dst)
        with Image.open(src) as im:
            x = tf(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            p_torch = torch.softmax(model(x), 1)[0].numpy()
        rec = {"file": dst.name, "true_class": r["class"], "species": r["taxon_name"], "attribution": r["attribution"],
               "observation_url": r["observation_url"], "input_tensor_stats": {"mean": float(x.mean()), "std": float(x.std()),
               "first5": [float(v) for v in x.flatten()[:5]]}, "torch": p_torch.round(6).tolist()}
        for k, s in sessions.items():
            p = torch.softmax(torch.from_numpy(s.run(None, {"input": x.numpy()})[0]), 1)[0].numpy()
            rec[k] = p.round(6).tolist()
            rec[f"{k}_max_abs_diff"] = float(np.abs(p - p_torch).max())
            rec[f"{k}_argmax_match"] = bool(p.argmax() == p_torch.argmax())
            worst[k] = max(worst[k], rec[f"{k}_max_abs_diff"])
        rec["pred"] = CLASSES[int(p_torch.argmax())]
        out.append(rec)
        print(f"{dst.name:34s} true={r['class']:12s} torch={rec['pred']:12s} p={p_torch.max():.3f} "
              + " ".join(f"{k}Δ={rec[f'{k}_max_abs_diff']:.2e}" for k in sessions))

    summary = {"n_images": len(out), "classes": CLASSES, "max_abs_prob_diff": worst,
               "argmax_agreement": {k: all(o[f"{k}_argmax_match"] for o in out) for k in sessions},
               "fp32_parity_pass": worst["onnx_fp32"] <= args.atol, "atol": args.atol, "images": out}
    (REPORTS_DIR / "parity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "images"}, indent=1))
    if not summary["fp32_parity_pass"]:
        sys.exit("PARITY FAILED: fp32 ONNX differs from PyTorch by more than atol")


if __name__ == "__main__":
    main()
