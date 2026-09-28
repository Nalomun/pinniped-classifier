#!/usr/bin/env python
"""Assign train/val/test splits BY OBSERVATION.

An iNaturalist observation can carry several photos of the same animal seconds apart.
If those landed in different splits the test set would be contaminated with near-duplicates
of training images and the accuracy would be a lie. So the unit of splitting is the
observation id, not the photo.

Splits are stratified by (class, subgroup) so fur seals / sea lions / each negative bucket
keep the same 70/15/15 ratio, and the shuffle is seeded so the split is reproducible.
The result is written back into data/manifest.csv as a `split` column and committed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import MANIFEST_CSV  # noqa: E402


def assign_splits(df: pd.DataFrame, seed: int, val_frac: float, test_frac: float) -> pd.Series:
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index, dtype=object)
    strata = df.fillna({"subgroup": ""}).groupby(["class", "subgroup"], sort=True)
    for key, g in strata:
        obs_ids = np.array(sorted(g["observation_id"].unique()))
        rng.shuffle(obs_ids)
        n = len(obs_ids)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        test_ids = set(obs_ids[:n_test])
        val_ids = set(obs_ids[n_test : n_test + n_val])
        for idx, oid in zip(g.index, g["observation_id"]):
            split[idx] = "test" if oid in test_ids else "val" if oid in val_ids else "train"
    return split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    df = pd.read_csv(MANIFEST_CSV)
    df["split"] = assign_splits(df, args.seed, args.val, args.test)

    # Leakage guard: every observation must live in exactly one split.
    per_obs = df.groupby("observation_id")["split"].nunique()
    assert (per_obs == 1).all(), "observation spans multiple splits"

    df.to_csv(MANIFEST_CSV, index=False)
    print("photos per class x split:\n", pd.crosstab(df["class"], df["split"]).to_string())
    print("\neared_seal subgroup x split:\n", pd.crosstab(df[df["class"] == "eared_seal"]["subgroup"], df[df["class"] == "eared_seal"]["split"]).to_string())
    print("\nobservations per split:", df.groupby("split")["observation_id"].nunique().to_dict())
    print(f"\nwrote {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
