#!/usr/bin/env python
"""Evaluate a checkpoint on val/test and produce the README artefacts.

Outputs (reports/):
  metrics.json                 all numbers below
  confusion_matrix.png         test-set confusion matrix (counts + row-normalised)
  worst_misclassifications.png grid of the most confident wrong answers on test
  per_species.csv              accuracy per species on test
  threshold_sweep.csv          how the "unsure" threshold trades coverage vs. caught errors (val)

Metrics reported separately, as the brief asks:
  * pinniped_acc_strict : on test images that ARE pinnipeds, 4-way argmax == truth
                          (a pinniped mistaken for not_pinniped counts as wrong). Ship bar: >= 0.95
  * pinniped_acc_3way   : same images, argmax restricted to the three pinniped logits
  * fur_seal_acc        : fur-seal test images called eared_seal (4-way)
  * not_pinniped P/R    : precision/recall of the not_pinniped class (4-way)
  * threshold           : tuned on VAL, applied to test; see tune_threshold()

Also importable: predict_probs(), compute_metrics() are reused by export.py to score the
ONNX / int8 models with identical code.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASSES,
    IMAGES_DIR,
    OUTPUTS_DIR,
    PINNIPED_CLASSES,
    REPORTS_DIR,
    ManifestDataset,
    eval_transform,
    get_device,
    load_manifest,
)

PIN_IDX = [CLASSES.index(c) for c in PINNIPED_CLASSES]
NP_IDX = CLASSES.index("not_pinniped")


def load_checkpoint(path: Path, device: str):
    from train import build_model

    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["model"], pretrained=False)
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_probs(model, df: pd.DataFrame, device: str, mean, std, img_size: int = 224, batch_size: int = 128, workers: int = 4) -> np.ndarray:
    ds = ManifestDataset(df, eval_transform(mean, std, img_size))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    out = []
    for x, _ in loader:
        logits = model(x.to(device)).float()
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------
def apply_threshold(probs: np.ndarray, t: float) -> np.ndarray:
    """Decision rule shared with the browser: argmax; if that is a pinniped class with
    probability < t, answer 'unsure' (-1)."""
    pred = probs.argmax(1)
    top_p = probs[np.arange(len(probs)), pred]
    unsure = np.isin(pred, PIN_IDX) & (top_p < t)
    return np.where(unsure, -1, pred)


def tune_threshold(probs: np.ndarray, y: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Pick t on the validation set.

    Among predictions whose argmax is a pinniped class, some are right and some are wrong
    (either a mislabelled pinniped or a negative that slipped through). We want the threshold
    that flags as many of the wrong ones as possible while flagging few right ones:
    maximise J = (fraction of wrong flagged) - (fraction of right flagged), Youden's index.
    """
    pred = probs.argmax(1)
    top_p = probs[np.arange(len(probs)), pred]
    m = np.isin(pred, PIN_IDX)
    right, wrong = m & (pred == y), m & (pred != y)
    rows = []
    for t in np.round(np.arange(0.40, 0.991, 0.01), 2):
        fl = top_p < t
        rows.append(
            {
                "threshold": t,
                "wrong_flagged": float((fl & wrong).sum() / max(1, wrong.sum())),
                "right_flagged": float((fl & right).sum() / max(1, right.sum())),
                "n_wrong": int(wrong.sum()),
                "n_right": int(right.sum()),
            }
        )
    sweep = pd.DataFrame(rows)
    sweep["youden_j"] = sweep["wrong_flagged"] - sweep["right_flagged"]
    best = sweep.loc[sweep["youden_j"].idxmax()]
    return float(best["threshold"]), sweep


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(probs: np.ndarray, df: pd.DataFrame, threshold: float | None = None) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    y = df["class"].map(CLASSES.index).to_numpy()
    pred = probs.argmax(1)
    is_pin = np.isin(y, PIN_IDX)
    pred3 = np.array(PIN_IDX)[probs[:, PIN_IDX].argmax(1)]
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=list(range(len(CLASSES))), zero_division=0)
    fur = (df["subgroup"] == "fur_seal").to_numpy()
    sea = (df["subgroup"] == "sea_lion").to_numpy()
    m = {
        "n": int(len(y)),
        "accuracy_4way": float((pred == y).mean()),
        "pinniped_acc_strict": float((pred[is_pin] == y[is_pin]).mean()),
        "pinniped_acc_3way": float((pred3[is_pin] == y[is_pin]).mean()),
        "pinniped_n": int(is_pin.sum()),
        "fur_seal_acc": float((pred[fur] == y[fur]).mean()) if fur.any() else None,
        "fur_seal_n": int(fur.sum()),
        "sea_lion_acc": float((pred[sea] == y[sea]).mean()) if sea.any() else None,
        "sea_lion_n": int(sea.sum()),
        "per_class": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "not_pinniped_precision": float(p[NP_IDX]),
        "not_pinniped_recall": float(r[NP_IDX]),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(CLASSES)))).tolist(),
    }
    # negatives by bucket (lookalike vs other) so we can see where the roast fails
    for b in sorted(df.loc[y == NP_IDX, "subgroup"].dropna().unique()):
        mask = (df["subgroup"] == b).to_numpy() & (y == NP_IDX)
        m[f"not_pinniped_recall_{b}"] = float((pred[mask] == NP_IDX).mean())
    if threshold is not None:
        pt = apply_threshold(probs, threshold)
        answered = pt != -1
        m["threshold"] = threshold
        m["with_threshold"] = {
            "pinniped_coverage": float(answered[is_pin].mean()),
            "pinniped_acc_on_answered": float((pt[is_pin & answered] == y[is_pin & answered]).mean()),
            "pinniped_unsure_rate": float((~answered[is_pin]).mean()),
            "negatives_wrongly_called_pinniped_before": int(np.isin(pred[~is_pin], PIN_IDX).sum()),
            "negatives_wrongly_called_pinniped_after": int(np.isin(pt[~is_pin], PIN_IDX).sum()),
            "pinniped_errors_before": int((pred[is_pin] != y[is_pin]).sum()),
            "pinniped_errors_after": int(((pt[is_pin] != y[is_pin]) & answered[is_pin]).sum()),
        }
    return m


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_confusion(cm: list[list[int]], path: Path, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=150)
    ax.imshow(norm, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]}\n{norm[i, j]:.1%}", ha="center", va="center", fontsize=9,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_worst(df: pd.DataFrame, probs: np.ndarray, path: Path, n: int = 20):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    y = df["class"].map(CLASSES.index).to_numpy()
    pred = probs.argmax(1)
    conf = probs[np.arange(len(probs)), pred]
    wrong = np.where(pred != y)[0]
    worst = wrong[np.argsort(-conf[wrong])][:n]
    cols = 5
    rows = int(np.ceil(len(worst) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.4), dpi=110)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, worst):
        r = df.iloc[i]
        with Image.open(IMAGES_DIR / r["local_path"]) as im:
            ax.imshow(im.convert("RGB"))
        sp = str(r.get("taxon_name") or "")[:28]
        ax.set_title(f"true: {r['class']}\npred: {CLASSES[pred[i]]} ({conf[i]:.0%})\n{sp}", fontsize=8)
    fig.suptitle(f"{len(worst)} most confident test errors (of {len(wrong)} total)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return [{"photo_id": int(df.iloc[i]["photo_id"]), "true": df.iloc[i]["class"], "pred": CLASSES[pred[i]],
             "confidence": float(conf[i]), "species": df.iloc[i].get("taxon_name"), "url": df.iloc[i]["observation_url"]} for i in worst]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(OUTPUTS_DIR / "best.pt"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=0,
                    help="inference resolution; 0 = the checkpoint's training size. Testing somewhat ABOVE the training "
                         "size helps (the 'FixRes' effect: random-resized-crop training shows objects larger than a "
                         "centre crop does). For the 288px model, 352 gave +1.2 pts.")
    args = ap.parse_args()

    device = get_device(args.device)
    REPORTS_DIR.mkdir(exist_ok=True)
    model, ck = load_checkpoint(Path(args.checkpoint), device)
    print(f"loaded {args.checkpoint} ({ck['model']}, best val epoch {ck.get('val')})")

    val_df, test_df = load_manifest("val"), load_manifest("test")
    img_size = args.img_size or ck.get("img_size", 224)
    print(f"inference size {img_size} (trained at {ck.get('img_size', 224)})")
    val_probs = predict_probs(model, val_df, device, ck["mean"], ck["std"], img_size, workers=args.workers)
    test_probs = predict_probs(model, test_df, device, ck["mean"], ck["std"], img_size, workers=args.workers)
    np.savez(OUTPUTS_DIR / "probs.npz", val=val_probs, test=test_probs)

    threshold, sweep = tune_threshold(val_probs, val_df["class"].map(CLASSES.index).to_numpy())
    sweep.to_csv(REPORTS_DIR / "threshold_sweep.csv", index=False)
    val_m = compute_metrics(val_probs, val_df, threshold)
    test_m = compute_metrics(test_probs, test_df, threshold)

    plot_confusion(test_m["confusion_matrix"], REPORTS_DIR / "confusion_matrix.png", f"Test confusion matrix (n={test_m['n']})")
    worst = plot_worst(test_df, test_probs, REPORTS_DIR / "worst_misclassifications.png")

    # per-species accuracy on test (pinnipeds) + per-bucket for negatives
    pred = test_probs.argmax(1)
    per = test_df.assign(correct=(pred == test_df["class"].map(CLASSES.index).to_numpy()))
    per_species = (per.groupby(["class", "subgroup", "query_taxon"], dropna=False)["correct"]
                   .agg(n="size", acc="mean").reset_index().sort_values(["class", "acc"]))
    per_species.to_csv(REPORTS_DIR / "per_species.csv", index=False)

    metrics = {"model": ck["model"], "tag": ck["tag"], "img_size": img_size, "train_img_size": ck.get("img_size", 224),
               "best_epoch": ck.get("val"), "threshold": threshold,
               "val": val_m, "test": test_m, "worst_test_errors": worst}
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\n==== TEST ====")
    for k in ["accuracy_4way", "pinniped_acc_strict", "pinniped_acc_3way", "fur_seal_acc", "sea_lion_acc", "not_pinniped_precision", "not_pinniped_recall"]:
        print(f"{k:28s} {test_m[k]:.4f}")
    print("per-class:")
    for c, v in test_m["per_class"].items():
        print(f"  {c:14s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
    print(f"threshold (tuned on val) = {threshold}")
    print("with threshold:", json.dumps(test_m["with_threshold"], indent=1))
    print("\nper-species (lowest first):\n", per_species.head(12).to_string(index=False))
    bar = test_m["pinniped_acc_strict"] >= 0.95
    print(f"\nSHIP BAR (>=95% pinniped test acc): {'PASS' if bar else 'FAIL'} ({test_m['pinniped_acc_strict']:.2%})")


if __name__ == "__main__":
    main()
