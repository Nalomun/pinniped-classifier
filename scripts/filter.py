#!/usr/bin/env python
"""Dataset v2 step: drop pinniped photos where no animal is actually visible.

Research-grade iNaturalist observations are records of *presence*, not portraits. A large
share of pinniped photos are a speck in the water, an aerial view of a colony, empty ocean
where the animal just dived, a skull on the beach or a carcass. Those are unlearnable at
224 px and are not what a visitor uploads, so they are removed from training AND evaluation.

Method: CLIP (ViT-B/32) zero-shot. Every photo is scored against "animal clearly visible"
prompts vs "nothing usable" prompts (see PROMPTS); the score is the total softmax mass on the
positive prompts. Pinniped photos below --threshold are dropped. not_pinniped photos are
kept regardless: an empty beach or a distant bird IS "not a pinniped".

The threshold was chosen by eye from reports/filter_bands.png (grids of photos in score bands).

Outputs
  data/manifest_unfiltered.csv   everything download.py fetched, with the clip score
  data/manifest.csv              the kept rows (split.py runs on this)
  reports/filter_bands.png       what the score bands look like (README)
  reports/filter_dropped.png     a sample of what was dropped (README)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, IMAGES_DIR, MANIFEST_CSV, PINNIPED_CLASSES, REPORTS_DIR  # noqa: E402

CLIP_MODEL, CLIP_TAG = "ViT-B-32", "laion2b_s34b_b79k"
PROMPTS = {
    "visible": [
        "a photo of a seal",
        "a photo of a sea lion",
        "a photo of a fur seal",
        "a photo of a walrus",
        "a close-up photo of a seal's face",
        "a photo of a seal lying on a beach",
        "a photo of sea lions on rocks",
        "a photo of a seal swimming with its head above water",
        "a photo of a group of seals",
        "a photo of a wild animal, clearly visible",
    ],
    "unusable": [
        "a photo of the ocean with nothing in it",
        "a photo of waves",
        "a landscape photo of a coastline",
        "an aerial photo of a beach",
        "a photo of a tiny dot far away in the water",
        "a photo of an animal skull",
        "a photo of bones on a beach",
        "a photo of a dead, decomposing animal",
        "a photo of animal tracks in the sand",
        "a photo of rocks",
        "a very blurry photo",
        "a photo of a sign",
        "a photo of a person",
    ],
}


class Imgs(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.pre = paths, preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.pre(im.convert("RGB"))


@torch.no_grad()
def score(df: pd.DataFrame, batch_size: int = 64, workers: int = 2) -> np.ndarray:
    import open_clip

    model, _, pre = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_TAG)
    model.eval()
    tok = open_clip.get_tokenizer(CLIP_MODEL)
    texts = PROMPTS["visible"] + PROMPTS["unusable"]
    t = model.encode_text(tok(texts))
    t = t / t.norm(dim=-1, keepdim=True)
    n_pos = len(PROMPTS["visible"])
    dl = DataLoader(Imgs([str(IMAGES_DIR / p) for p in df["local_path"]], pre), batch_size=batch_size, num_workers=workers)
    out = []
    for i, x in enumerate(dl):
        f = model.encode_image(x)
        f = f / f.norm(dim=-1, keepdim=True)
        p = (100.0 * f @ t.T).softmax(dim=-1)
        out.append(p[:, :n_pos].sum(1))
        if i % 20 == 0:
            print(f"  scored {min((i+1)*batch_size, len(df))}/{len(df)}", flush=True)
    return torch.cat(out).numpy()


def grid(df: pd.DataFrame, path: Path, title: str, cols: int = 6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(df) / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6), dpi=90)
    for ax in np.array(axes).reshape(-1):
        ax.axis("off")
    for ax, (_, r) in zip(np.array(axes).reshape(-1), df.iterrows()):
        with Image.open(IMAGES_DIR / r["local_path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{r['clip_visible_score']:.2f} {str(r['taxon_name'])[:22]}", fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--rescore", action="store_true", help="recompute CLIP scores even if manifest_unfiltered.csv has them")
    ap.add_argument("--inspect-only", action="store_true", help="write band grids, don't filter")
    args = ap.parse_args()

    unf = DATA_DIR / "manifest_unfiltered.csv"
    src = unf if unf.exists() and not args.rescore else MANIFEST_CSV
    df = pd.read_csv(src)
    if "clip_visible_score" not in df.columns or args.rescore:
        print(f"[filter] scoring {len(df)} photos with CLIP {CLIP_MODEL}/{CLIP_TAG}")
        df["clip_visible_score"] = score(df)
        df.to_csv(unf, index=False)
        print(f"[filter] wrote {unf}")

    REPORTS_DIR.mkdir(exist_ok=True)
    pin = df[df["class"].isin(PINNIPED_CLASSES)]
    rng = np.random.default_rng(0)
    bands = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    parts = []
    for lo, hi in bands:
        b = pin[(pin.clip_visible_score >= lo) & (pin.clip_visible_score < hi)]
        parts.append(b.sample(n=min(6, len(b)), random_state=int(rng.integers(1e6))))
        print(f"  band [{lo:.1f},{hi:.1f}): {len(b)} pinniped photos ({len(b)/len(pin):.1%})")
    grid(pd.concat(parts), REPORTS_DIR / "filter_bands.png", "CLIP 'animal visible' score bands (6 random pinniped photos per band, low to high)")
    if args.inspect_only:
        return

    keep = (~df["class"].isin(PINNIPED_CLASSES)) | (df["clip_visible_score"] >= args.threshold)
    dropped = df[~keep]
    print(f"[filter] threshold={args.threshold}: dropping {len(dropped)} of {len(pin)} pinniped photos ({len(dropped)/len(pin):.1%})")
    print(dropped.groupby("class").size().to_string())
    grid(dropped.sample(n=min(24, len(dropped)), random_state=0), REPORTS_DIR / "filter_dropped.png", f"Random sample of pinniped photos dropped by the visibility filter (score < {args.threshold})")
    kept = df[keep].reset_index(drop=True)
    kept.to_csv(MANIFEST_CSV, index=False)
    print("[filter] kept photos per class:\n" + kept["class"].value_counts().to_string())
    print(f"[filter] wrote {MANIFEST_CSV}; now run scripts/split.py")


if __name__ == "__main__":
    main()
