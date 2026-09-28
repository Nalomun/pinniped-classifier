"""Shared definitions: class order, taxa configuration, preprocessing spec, dataset.

Everything the browser needs to reproduce preprocessing lives in preprocess_spec() and is
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
# Budgets are ~40% above the final target because scripts/filter.py later drops photos
# where no animal is visible (distant specks, carcasses, tracks, skulls...).
PINNIPED_TAXA = {
    "true_seal": {"family": "Phocidae", "target_obs": 2300, "photos_per_obs": 2},
    "eared_seal": {
        "family": "Otariidae",
        # Separate budgets so fur seals are well represented (they are the whole
        # point of the app's "seal vs sea lion" correction).
        "subgroups": {"fur_seal": 1200, "sea_lion": 1200},
        "photos_per_obs": 2,
    },
    "walrus": {"family": "Odobenidae", "target_obs": 10_000, "photos_per_obs": 3},
}

# iNaturalist annotation (term_id, value_id) pairs that mean "this photo is not of a live,
# visible animal". Observations carrying any of these are skipped at download time.
# 17 = Alive or Dead (19 = Dead); 22 = Evidence of Presence (25 scat, 26 track, 29 bone,
# 23 feather, 35 egg, 36 hair, 37 leafmine, 38 gall, 39 molt).
EXCLUDED_ANNOTATIONS = {(17, 19), (22, 25), (22, 26), (22, 29), (22, 23), (22, 35), (22, 36), (22, 37), (22, 38), (22, 39)}
MIN_IMAGE_SIDE = 200  # px; smaller "medium" files are really thumbnails of deleted originals

# Negatives: (taxon name, rank hint, observation budget, bucket). Lookalikes get
# the largest share because that is where a pinniped model gets fooled.
NEGATIVE_TAXA = [
    # --- lookalikes (marine / semi-aquatic / blubbery / lying on rocks) ---
    ("Lutrinae", "subfamily", 420, "lookalike"),        # otters incl. sea otter
    ("Sirenia", "order", 300, "lookalike"),             # manatees, dugong
    ("Delphinidae", "family", 300, "lookalike"),        # dolphins, orcas
    ("Mysticeti", "parvorder", 140, "lookalike"),       # baleen whales
    ("Spheniscidae", "family", 340, "lookalike"),       # penguins
    ("Ursidae", "family", 220, "lookalike"),            # bears (closest land relatives)
    ("Hippopotamidae", "family", 140, "lookalike"),     # hippos
    ("Cheloniidae", "family", 200, "lookalike"),        # sea turtles on beaches
    ("Castor", "genus", 140, "lookalike"),              # beavers
    ("Crocodylia", "order", 140, "lookalike"),          # crocs basking
    # --- other common animals ---
    ("Canis familiaris", "species", 220, "other_animal"),
    ("Felis catus", "species", 140, "other_animal"),
    ("Cervidae", "family", 140, "other_animal"),
    ("Bovidae", "family", 110, "other_animal"),
    ("Laridae", "family", 160, "other_animal"),         # gulls: share beaches with seals
    ("Phalacrocoracidae", "family", 140, "other_animal"),  # cormorants on the same rocks
    # --- a small slice of non-animals so "my lunch" and "a plant" also get roasted ---
    ("Plantae", "kingdom", 140, "non_animal"),
    ("Fungi", "kingdom", 70, "non_animal"),
    ("Insecta", "class", 140, "non_animal"),
]
NEGATIVE_PHOTOS_PER_OBS = 1

ALLOWED_LICENSES = {"cc0", "cc-by"}

# ---------------------------------------------------------------------------
# Preprocessing spec (shared by train/eval/export/parity and the browser).
# ---------------------------------------------------------------------------
IMG_SIZE = 224
RESIZE_SHORTER = 256  # = IMG_SIZE / 0.875, the standard ImageNet crop fraction


def resize_for(img_size: int) -> int:
    return int(round(img_size / 0.875))
# Default normalisation. NOTE: the mean/std actually used are taken from the pretrained
# weights' config at training time and stored in the checkpoint / model_meta.json, because
# they differ between weight sets (e.g. timm's *.miil_in21k_ft_in1k weights expect raw [0,1]
# inputs: mean 0, std 1; the *.ra_in1k weights expect ImageNet mean/std).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess_spec(mean, std, img_size: int = IMG_SIZE) -> dict:
    return {
        "color_space": "RGB",
        "steps": [
            {"op": "resize_shorter_side", "size": resize_for(img_size), "interpolation": "bilinear"},
            {"op": "center_crop", "size": img_size},
            {"op": "to_float", "scale": "divide_by_255"},
            {"op": "normalize", "mean": list(mean), "std": list(std)},
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


def eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, img_size: int = IMG_SIZE):
    """Deterministic transform used for val/test/export/parity. Must match preprocess_spec()."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.Resize(resize_for(img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(list(mean), list(std)),
        ]
    )


def train_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD, img_size: int = IMG_SIZE):
    """Augmentation for training. Backgrounds (water/sand/ice/rock) and lighting
    vary a lot across iNat photos, so we crop aggressively and jitter colour."""
    from torchvision import transforms as T

    return T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.35, 1.0), ratio=(0.75, 1.333)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
            T.ToTensor(),
            T.Normalize(list(mean), list(std)),
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
