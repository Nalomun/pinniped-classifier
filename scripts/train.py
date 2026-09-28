#!/usr/bin/env python
"""Fine-tune a small ImageNet-pretrained timm model on the 4-class manifest.

Default backbone: mobilenetv3_large_100 (timm tag mobilenetv3_large_100.miil_in21k_ft_in1k).
Why this one:
  * ~4.2M params once the 1000-way ImageNet head is swapped for our 4-way head
    -> ~17 MB fp32 ONNX, ~4.5 MB int8. Fits the <= 20 MB browser budget.
  * ~0.22 GFLOPs at 224px: roughly 2x cheaper than efficientnet_b0 (0.39 GFLOPs), which
    matters for WASM inference on phones.
  * Ops (hardswish/hardsigmoid, SE blocks) all export cleanly to ONNX opset 17 and run on
    onnxruntime-web's WASM backend.
  * The miil_in21k_ft_in1k weights were pretrained on ImageNet-21k, which contains far more
    animal categories than 1k, so the features transfer better to fine-grained mammals.
efficientnet_b0 is supported via --model as the fallback if MobileNetV3 misses the bar.

Recipe: AdamW, cosine LR with warmup, label smoothing, mixed precision on CUDA.
Whole network is fine-tuned (not just the head) with a lower LR on the backbone.
The best checkpoint by validation *pinniped* accuracy is kept.

Runs on Colab GPU in a few minutes; on an 8-core laptop CPU expect ~4-5 min/epoch.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CLASSES,
    OUTPUTS_DIR,
    PINNIPED_CLASSES,
    ManifestDataset,
    eval_transform,
    get_device,
    load_manifest,
    seed_everything,
    train_transform,
)

MODEL_TAGS = {
    "mobilenetv3_large_100": "mobilenetv3_large_100.miil_in21k_ft_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
}


def build_model(name: str, pretrained: bool = True) -> nn.Module:
    tag = MODEL_TAGS.get(name, name)
    return timm.create_model(tag, pretrained=pretrained, num_classes=len(CLASSES))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, enabled=(device == "cuda")):
            out = model(x)
        logits_all.append(out.float().cpu())
        y_all.append(y)
    logits = torch.cat(logits_all)
    y = torch.cat(y_all)
    pred = logits.argmax(1)
    acc = (pred == y).float().mean().item()
    pin_idx = torch.tensor([CLASSES.index(c) for c in PINNIPED_CLASSES])
    is_pin = torch.isin(y, pin_idx)
    pin_acc = (pred[is_pin] == y[is_pin]).float().mean().item() if is_pin.any() else float("nan")
    loss = nn.functional.cross_entropy(logits, y).item()
    return {"loss": loss, "acc": acc, "pinniped_acc": pin_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mobilenetv3_large_100", choices=list(MODEL_TAGS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3, help="head LR; backbone uses lr * backbone-mult")
    ap.add_argument("--backbone-mult", type=float, default=0.25)
    ap.add_argument("--weight-decay", type=float, default=0.02)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training images")
    ap.add_argument("--out", default=str(OUTPUTS_DIR))
    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} model={args.model} epochs={args.epochs} bs={args.batch_size}")
    if device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))

    train_df = load_manifest("train")
    val_df = load_manifest("val")
    if args.limit:
        train_df = train_df.sample(n=min(args.limit, len(train_df)), random_state=args.seed)
    print(f"train={len(train_df)} val={len(val_df)}")
    print("train class counts:", train_df["class"].value_counts().to_dict())

    train_ds = ManifestDataset(train_df, train_transform())
    val_ds = ManifestDataset(val_df, eval_transform())
    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=pin, drop_last=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.workers, pin_memory=pin)

    model = build_model(args.model).to(device)
    if device == "cuda":
        model = model.to(memory_format=torch.channels_last)

    head_params = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr * args.backbone_mult}, {"params": head_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))

    history, best = [], {"pinniped_acc": -1.0, "acc": -1.0}
    for epoch in range(args.epochs):
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(y)
            n += len(y)
            if i % 20 == 0:
                print(f"  ep{epoch+1} it{i}/{steps_per_epoch} loss={loss.item():.3f} {(time.time()-t0)/(i+1):.2f}s/it", flush=True)
        val = evaluate(model, val_loader, device)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / n, **{f"val_{k}": v for k, v in val.items()}, "time_s": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec), flush=True)
        # Model selection on pinniped accuracy (the ship bar), ties broken by overall acc.
        if (val["pinniped_acc"], val["acc"]) > (best["pinniped_acc"], best["acc"]):
            best = {**val, "epoch": epoch + 1}
            torch.save({"model": args.model, "tag": MODEL_TAGS[args.model], "classes": CLASSES, "state_dict": model.state_dict(), "args": vars(args), "val": val},
                       out_dir / "best.pt")
            print(f"  saved best (val pinniped_acc={val['pinniped_acc']:.4f}, acc={val['acc']:.4f})")

    (out_dir / "history.json").write_text(json.dumps({"history": history, "best": best, "args": vars(args)}, indent=2))
    print("best:", best)


if __name__ == "__main__":
    main()
