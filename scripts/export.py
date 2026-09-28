#!/usr/bin/env python
"""Export the best checkpoint to ONNX, optionally quantize to int8, write model_meta.json.

Steps
  1. torch -> ONNX (opset 17, dynamic batch axis), checked with onnx.checker.
  2. Score the fp32 ONNX model on the test set with onnxruntime using the SAME preprocessing
     and metric code as eval.py.
  3. Static int8 quantization (QDQ format, per-channel weights, calibrated on val images).
     Kept only if pinniped accuracy AND overall accuracy drop by < --max-drop points
     (default 0.5). Both results are reported either way.
  4. export/model_meta.json: class order, input spec, preprocessing steps, mean/std,
     threshold (from reports/metrics.json), file hashes, and headline metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASSES,
    EXPORT_DIR,
    IMG_SIZE,
    MANIFEST_CSV,
    OUTPUTS_DIR,
    PINNIPED_CLASSES,
    REPORTS_DIR,
    ManifestDataset,
    eval_transform,
    load_manifest,
    preprocess_spec,
    resize_for,
)
from eval import compute_metrics, load_checkpoint  # noqa: E402

OPSET = 17


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_onnx(model, path: Path, img_size: int):
    model.eval()
    dummy = torch.randn(1, 3, img_size, img_size)
    try:
        torch.onnx.export(model, dummy, str(path), input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=OPSET, dynamo=False)
        how = "torchscript exporter"
    except Exception as e:  # noqa: BLE001
        print(f"[export] legacy exporter failed ({e}); trying dynamo exporter")
        torch.onnx.export(model, (dummy,), str(path), input_names=["input"], output_names=["logits"],
                          dynamic_shapes={"x": {0: "batch"}}, opset_version=OPSET, dynamo=True)
        how = "dynamo exporter"
    onnx.checker.check_model(str(path))
    print(f"[export] wrote {path} ({path.stat().st_size/1e6:.1f} MB) via {how}")


def onnx_probs(path: Path, df: pd.DataFrame, mean, std, img_size: int, batch_size: int = 64) -> np.ndarray:
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    ds = ManifestDataset(df, eval_transform(mean, std, img_size))
    out = []
    for i in range(0, len(ds), batch_size):
        x = torch.stack([ds[j][0] for j in range(i, min(i + batch_size, len(ds)))]).numpy()
        logits = sess.run(None, {name: x})[0]
        out.append(torch.softmax(torch.from_numpy(logits), 1).numpy())
    return np.concatenate(out)


class CalibReader:
    """Feeds val images to the static quantizer."""

    def __init__(self, df: pd.DataFrame, input_name: str, mean, std, img_size: int, batch_size: int = 16):
        self.ds = ManifestDataset(df, eval_transform(mean, std, img_size))
        self.name, self.bs, self.i = input_name, batch_size, 0

    def get_next(self):
        if self.i >= len(self.ds):
            return None
        x = torch.stack([self.ds[j][0] for j in range(self.i, min(self.i + self.bs, len(self.ds)))]).numpy()
        self.i += self.bs
        return {self.name: x}


def quantize(fp32_path: Path, int8_path: Path, calib_df: pd.DataFrame, mean, std, img_size: int, method: str = "percentile"):
    """Static int8 quantization (QDQ, per-channel weights). MobileNetV3's hardswish + SE blocks
    produce activation outliers, so plain MinMax calibration can wreck accuracy; percentile /
    entropy calibration clip those outliers. main() tries several and keeps the best on val."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    methods = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile, "entropy": CalibrationMethod.Entropy}
    pre = fp32_path.with_suffix(".pre.onnx")
    quant_pre_process(str(fp32_path), str(pre))
    sess = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"])
    reader = CalibReader(calib_df, sess.get_inputs()[0].name, mean, std, img_size)
    quantize_static(str(pre), str(int8_path), reader, quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=methods[method], extra_options={"CalibMovingAverage": True} if method == "minmax" else {})
    pre.unlink(missing_ok=True)
    print(f"[quant:{method}] wrote {int8_path} ({int8_path.stat().st_size/1e6:.1f} MB)")


def weight_only_int8(fp32_path: Path, out_path: Path, min_elements: int = 20_000, mse_search: bool = True) -> int:
    """Store Conv/Gemm/MatMul weights as per-channel symmetric int8 with a DequantizeLinear in
    front of each consumer. Compute stays fp32, so no calibration data is needed and ORT-web
    dequantizes once at session load. Returns the number of tensors quantized.

    Two details that took the accuracy drop at 352px from -1.5 pts to 0.0 (for ~0.6 MB):
      * tensors with fewer than `min_elements` weights (first conv, squeeze-excite, depthwise
        convs of the early blocks) stay fp32; they are tiny but quantization-sensitive.
      * each channel's scale is chosen by a small MSE search over clipping ratios instead of
        plain max-abs, which wastes int8 range on a single outlier weight."""
    from onnx import helper, numpy_helper

    model = onnx.load(str(fp32_path))
    g = model.graph
    consumers: dict[str, list] = {}
    for n in g.node:
        for inp in n.input:
            consumers.setdefault(inp, []).append(n)
    new_nodes, n_q = [], 0
    for init in list(g.initializer):
        users = consumers.get(init.name, [])
        W = numpy_helper.to_array(init)
        if W.dtype != np.float32 or W.ndim < 2 or not users or not all(u.op_type in ("Conv", "Gemm", "MatMul") for u in users):
            continue
        if W.size < min_elements:
            continue
        axis = 1 if users[0].op_type == "MatMul" else 0  # output-channel axis
        Wr = W.reshape(W.shape[0], -1) if axis == 0 else W.T.reshape(W.shape[1], -1)
        amax = np.abs(Wr).max(1)
        if mse_search:
            best, best_err = amax.copy(), np.full(len(amax), np.inf)
            for r in np.linspace(0.85, 1.0, 16):
                sc = np.maximum(amax * r / 127.0, 1e-8)
                err = ((np.clip(np.round(Wr / sc[:, None]), -127, 127) * sc[:, None] - Wr) ** 2).sum(1)
                better = err < best_err
                best[better], best_err[better] = amax[better] * r, err[better]
            amax = best
        scale = np.maximum(amax / 127.0, 1e-8).astype(np.float32)
        bshape = [-1] + [1] * (W.ndim - 1) if axis == 0 else [1, -1] + [1] * (W.ndim - 2)
        Wq = np.clip(np.round(W / scale.reshape(bshape)), -127, 127).astype(np.int8)
        g.initializer.remove(init)
        g.initializer.extend([numpy_helper.from_array(Wq, init.name + "_q"), numpy_helper.from_array(scale, init.name + "_scale"),
                              numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), init.name + "_zp")])
        new_nodes.append(helper.make_node("DequantizeLinear", [init.name + "_q", init.name + "_scale", init.name + "_zp"], [init.name],
                                          axis=axis, name="dq_" + init.name))
        n_q += 1
    nodes = list(g.node)
    del g.node[:]
    g.node.extend(new_nodes + nodes)  # DequantizeLinear nodes only depend on initializers, so first is a valid order
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    print(f"[w8] quantized {n_q} weight tensors -> {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return n_q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUTPUTS_DIR / "best.pt"))
    ap.add_argument("--max-drop", type=float, default=0.5, help="max accuracy drop (points) to keep int8")
    ap.add_argument("--calib-images", type=int, default=300)
    ap.add_argument("--skip-quant", action="store_true", help="skip static (activation) int8; weight-only int8 is always tried")
    ap.add_argument("--img-size", type=int, default=0, help="inference resolution baked into the ONNX input shape; 0 = reports/metrics.json's img_size")
    args = ap.parse_args()

    EXPORT_DIR.mkdir(exist_ok=True)
    model, ck = load_checkpoint(Path(args.checkpoint), "cpu")
    mean, std = ck["mean"], ck["std"]
    metrics = json.loads((REPORTS_DIR / "metrics.json").read_text())
    img_size = args.img_size or metrics.get("img_size") or ck.get("img_size", IMG_SIZE)
    print(f"[export] inference size {img_size} (trained at {ck.get('img_size', IMG_SIZE)}); threshold from metrics.json was tuned at {metrics.get('img_size')}")
    threshold = metrics["threshold"]
    test_df, val_df = load_manifest("test"), load_manifest("val")

    fp32_path = EXPORT_DIR / "pinniped.onnx"
    export_onnx(model, fp32_path, img_size)
    fp32_m = compute_metrics(onnx_probs(fp32_path, test_df, mean, std, img_size), test_df, threshold)
    torch_m = metrics["test"]
    print(f"[fp32 onnx] test acc={fp32_m['accuracy_4way']:.4f} pinniped={fp32_m['pinniped_acc_strict']:.4f} "
          f"(torch: {torch_m['accuracy_4way']:.4f} / {torch_m['pinniped_acc_strict']:.4f})")

    results = {"fp32": {"file": fp32_path.name, "size_mb": round(fp32_path.stat().st_size / 1e6, 2), "sha256": sha256(fp32_path),
                        "test_accuracy_4way": fp32_m["accuracy_4way"], "test_pinniped_acc_strict": fp32_m["pinniped_acc_strict"],
                        "test_fur_seal_acc": fp32_m["fur_seal_acc"]}}
    recommended = "fp32"

    # Weight-only int8: Conv/Gemm weights stored as int8 + per-channel DequantizeLinear, all
    # activations and compute stay fp32. ~4x smaller download, no calibration needed, and
    # ORT-web just dequantizes the weights once at session load. Kept under the same rule.
    w8_path = EXPORT_DIR / "pinniped_w8.onnx"
    weight_only_int8(fp32_path, w8_path)
    w8_m = compute_metrics(onnx_probs(w8_path, test_df, mean, std, img_size), test_df, threshold)
    w8_drop_acc = 100 * (fp32_m["accuracy_4way"] - w8_m["accuracy_4way"])
    w8_drop_pin = 100 * (fp32_m["pinniped_acc_strict"] - w8_m["pinniped_acc_strict"])
    w8_keep = w8_drop_acc < args.max_drop and w8_drop_pin < args.max_drop
    print(f"[w8 onnx] {w8_path.stat().st_size/1e6:.1f} MB test acc={w8_m['accuracy_4way']:.4f} (drop {w8_drop_acc:+.2f} pts) "
          f"pinniped={w8_m['pinniped_acc_strict']:.4f} (drop {w8_drop_pin:+.2f} pts) -> {'KEEP' if w8_keep else 'DISCARD'}")
    results["weight_only_int8"] = {"file": w8_path.name if w8_keep else None, "size_mb": round(w8_path.stat().st_size / 1e6, 2),
                                   "sha256": sha256(w8_path), "test_accuracy_4way": w8_m["accuracy_4way"],
                                   "test_pinniped_acc_strict": w8_m["pinniped_acc_strict"], "test_fur_seal_acc": w8_m["fur_seal_acc"],
                                   "drop_accuracy_points": round(w8_drop_acc, 3), "drop_pinniped_points": round(w8_drop_pin, 3), "kept": w8_keep}
    if w8_keep:
        recommended = "weight_only_int8"
    else:
        w8_path.unlink(missing_ok=True)

    int8_path = EXPORT_DIR / "pinniped_int8.onnx"
    if not args.skip_quant:
        calib_df = val_df.sample(n=min(args.calib_images, len(val_df)), random_state=0)
        # pick the calibration method on VAL, then report the winner on TEST
        val_scores = {}
        for method in ["percentile", "entropy", "minmax"]:
            cand = int8_path.with_name(f"pinniped_int8_{method}.onnx")
            try:
                quantize(fp32_path, cand, calib_df, mean, std, img_size, method)
                vm = compute_metrics(onnx_probs(cand, val_df, mean, std, img_size), val_df)
            except Exception as e:  # noqa: BLE001  (ORT calibrators occasionally crash on some graphs)
                print(f"[quant:{method}] FAILED: {type(e).__name__}: {str(e)[:120]}")
                cand.unlink(missing_ok=True)
                fp32_path.with_suffix(".pre.onnx").unlink(missing_ok=True)
                continue
            val_scores[method] = vm["pinniped_acc_strict"]
            print(f"[quant:{method}] val pinniped acc={vm['pinniped_acc_strict']:.4f} acc={vm['accuracy_4way']:.4f}")
        best_method = max(val_scores, key=val_scores.get)
        for method in val_scores:
            cand = int8_path.with_name(f"pinniped_int8_{method}.onnx")
            if method == best_method:
                cand.replace(int8_path)
            else:
                cand.unlink(missing_ok=True)
        print(f"[quant] best calibration on val: {best_method}")
        int8_m = compute_metrics(onnx_probs(int8_path, test_df, mean, std, img_size), test_df, threshold)
        drop_acc = 100 * (fp32_m["accuracy_4way"] - int8_m["accuracy_4way"])
        drop_pin = 100 * (fp32_m["pinniped_acc_strict"] - int8_m["pinniped_acc_strict"])
        keep = drop_acc < args.max_drop and drop_pin < args.max_drop
        print(f"[int8 onnx] test acc={int8_m['accuracy_4way']:.4f} (drop {drop_acc:+.2f} pts) "
              f"pinniped={int8_m['pinniped_acc_strict']:.4f} (drop {drop_pin:+.2f} pts) -> {'KEEP' if keep else 'DISCARD'}")
        results["int8"] = {"file": int8_path.name, "size_mb": round(int8_path.stat().st_size / 1e6, 2), "sha256": sha256(int8_path),
                           "test_accuracy_4way": int8_m["accuracy_4way"], "test_pinniped_acc_strict": int8_m["pinniped_acc_strict"],
                           "test_fur_seal_acc": int8_m["fur_seal_acc"], "drop_accuracy_points": round(drop_acc, 3),
                           "drop_pinniped_points": round(drop_pin, 3), "kept": keep, "calibration": best_method,
                           "val_pinniped_acc_by_calibration": val_scores}
        if keep:
            recommended = "int8"
        else:
            int8_path.unlink(missing_ok=True)
            results["int8"]["file"] = None

    manifest = pd.read_csv(MANIFEST_CSV)
    meta = {
        "name": "Seal or Sea Lion? pinniped classifier",
        "version": date.today().isoformat(),
        "architecture": ck["model"],
        "pretrained_tag": ck["tag"],
        "onnx_opset": OPSET,
        "recommended_file": results[recommended]["file"],
        "files": results,
        "classes": CLASSES,
        "pinniped_classes": PINNIPED_CLASSES,
        "class_descriptions": {
            "true_seal": "Phocidae (earless/true seals)",
            "eared_seal": "Otariidae (sea lions and fur seals)",
            "walrus": "Odobenidae",
            "not_pinniped": "anything else",
        },
        "input": {"name": "input", "shape": [1, 3, img_size, img_size], "dtype": "float32", "layout": "NCHW", "color_space": "RGB"},
        "output": {"name": "logits", "shape": [1, len(CLASSES)], "note": "raw logits; apply softmax to get probabilities"},
        "preprocessing": preprocess_spec(mean, std, img_size),
        "resize_shorter_side": resize_for(img_size),
        "crop_size": img_size,
        "train_img_size": ck.get("img_size", IMG_SIZE),
        "inference_size_note": ("Inference runs above the training resolution on purpose (FixRes effect); "
                                "the threshold was re-tuned on validation at this size."),
        "mean": mean,
        "std": std,
        "threshold": threshold,
        "decision_rule": ("argmax over the 4 probabilities. If the argmax is a pinniped class and its probability is below "
                          "`threshold`, show the 'unsure' response instead. If the argmax is not_pinniped, show the not-a-seal response."),
        "metrics_test": {k: torch_m[k] for k in ["accuracy_4way", "pinniped_acc_strict", "pinniped_acc_3way", "fur_seal_acc",
                                                  "not_pinniped_precision", "not_pinniped_recall"]},
        "training_data": {"source": "iNaturalist research-grade observations, CC0 / CC-BY photos only",
                          "n_images": int(len(manifest)), "per_class": manifest["class"].value_counts().to_dict(),
                          "attribution": "see data/manifest.csv"},
    }
    (EXPORT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[meta] wrote {EXPORT_DIR/'model_meta.json'} (recommended: {meta['recommended_file']})")


if __name__ == "__main__":
    main()
