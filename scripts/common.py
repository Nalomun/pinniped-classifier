"""Shared definitions: class order, taxa configuration, preprocessing spec, dataset.

Everything the browser needs to reproduce preprocessing lives in PREPROCESS and is
written verbatim into export/model_meta.json by scripts/export.py.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
OUTPUTS_DIR = REPO_ROOT / "outputs"
REPORTS_DIR = REPO_ROOT / "reports"
EXPORT_DIR = REPO_ROOT / "export"

# Class order is the model's output order. Never reorder without retraining.
CLASSES = ["true_seal", "eared_seal", "walrus", "not_pinniped"]
PINNIPED_CLASSES = ["true_seal", "eared_seal", "walrus"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# Otariidae genera that are fur seals (subfamily Arctocephalinae). Everything else in
# Otariidae is a sea lion. Used only for the fur-seal subgroup metric.
FUR_SEAL_GENERA = {"Arctocephalus", "Callorhinus"}

# ---------------------------------------------------------------------------
# Data sourcing configuration. Taxon IDs are resolved at runtime by name via the
# iNaturalist API (scripts/download.py), never hardcoded.
# ---------------------------------------------------------------------------
# For pinniped classes we enumerate species under the family and "water-fill" the
# observation budget across species so no single species dominates.
PINNIPED_TAXA = {
    "true_seal": {"family": "Phocidae", "target_obs": 1600, "photos_per_obs": 2},
    "eared_seal": {
        "family": "Otariidae",
        # Separate budgets so fur seals are well represented (they are the whole
        # point of the app's "seal vs sea lion" correction).
        "subgroups": {"fur_seal": 850, "sea_lion": 850},
        "photos_per_obs": 2,
    },
    "walrus": {"family": "Odobenidae", "target_obs": 10_000, "photos_per_obs": 3},
}

# Negatives: (taxon name, rank hint, observation budget, bucket). Lookalikes get
# the largest share because that is where a pinniped model gets fooled.
NEGATIVE_TAXA = [
    # --- lookalikes (marine / semi-aquatic / blubbery / lying on rocks) ---
    ("Lutrinae", "subfamily", 320, "lookalike"),        # otters incl. sea otter
    ("Sirenia", "order", 220, "lookalike"),             # manatees, dugong
    ("Delphinidae", "family", 220, "lookalike"),        # dolphins, orcas
    ("Mysticeti", "parvorder", 100, "lookalike"),       # baleen whales
    ("Spheniscidae", "family", 260, "lookalike"),       # penguins
    ("Ursidae", "family", 160, "lookalike"),            # bears (closest land relatives)
    ("Hippopotamidae", "family", 100, "lookalike"),     # hippos
    ("Cheloniidae", "family", 150, "lookalike"),        # sea turtles on beaches
    ("Castor", "genus", 100, "lookalike"),              # beavers
    ("Crocodylia", "order", 100, "lookalike"),          # crocs basking
    # --- other common animals ---
    ("Canis familiaris", "species", 160, "other_animal"),
    ("Felis catus", "species", 100, "other_animal"),
    ("Cervidae", "family", 100, "other_animal"),
    ("Bovidae", "family", 80, "other_animal"),
    ("Laridae", "family", 120, "other_animal"),         # gulls: share beaches with seals
    ("Phalacrocoracidae", "family", 100, "other_animal"),  # cormorants on the same rocks
    # --- a small slice of non-animals so "my lunch" and "a plant" also get roasted ---
    ("Plantae", "kingdom", 100, "non_animal"),
    ("Fungi", "kingdom", 50, "non_animal"),
    ("Insecta", "class", 100, "non_animal"),
]
NEGATIVE_PHOTOS_PER_OBS = 1

ALLOWED_LICENSES = {"cc0", "cc-by"}

# ---------------------------------------------------------------------------
# Preprocessing spec (shared by train/eval/export/parity and the browser).
# ---------------------------------------------------------------------------
IMG_SIZE = 224
RESIZE_SHORTER = 256
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PREPROCESS = {
    "color_space": "RGB",
    "steps": [
        {"op": "resize_shorter_side", "size": RESIZE_SHORTER, "interpolation": "bilinear"},
        {"op": "center_crop", "size": IMG_SIZE},
        {"op": "to_float", "scale": "divide_by_255"},
        {"op": "normalize", "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        {"op": "layout", "value": "NCHW"},
    ],
}


def seed_everything(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def eval_transform():
    """Deterministic transform used for val/test/export/parity. Must match PREPROCESS."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(RESIZE_SHORTER, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform():
    """Augmentation for training. Backgrounds (water/sand/ice/rock) and lighting
    vary a lot across iNat photos, so we crop aggressively and jitter colour."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.35, 1.0), ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.15, scale=(0.02, 0.12)),
        ]
    )


class ManifestDataset:
    """Minimal torch Dataset over manifest rows for one split."""

    def __init__(self, df, transform, images_dir: Path = IMAGES_DIR):
        import torch  # noqa: F401  (ensure torch importable)

        self.paths = [str(images_dir / p) for p in df["local_path"].tolist()]
        self.labels = [CLASS_TO_IDX[c] for c in df["class"].tolist()]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, self.labels[i]


def load_manifest(split: str | None = None, require_files: bool = True):
    import pandas as pd

    df = pd.read_csv(MANIFEST_CSV)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    if require_files:
        exists = df["local_path"].map(lambda p: os.path.exists(IMAGES_DIR / p))
        missing = int((~exists).sum())
        if missing:
            print(f"[manifest] warning: {missing} files missing on disk for split={split}; dropping them")
            df = df[exists].reset_index(drop=True)
    return df
