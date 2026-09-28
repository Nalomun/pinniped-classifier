#!/usr/bin/env python
"""Build the dataset from iNaturalist.

Two modes:

1. Fresh build (default): resolve taxon IDs by name via the API, enumerate species under
   each pinniped family, sample observations (research grade, CC0/CC-BY photos only),
   download photos and write data/manifest.csv.

2. --from-manifest: skip the API and re-download exactly the photos listed in the
   committed data/manifest.csv. This is the reproducible path (iNat data changes daily).

Sampling policy (see common.py for budgets):
  * Pinniped classes: species are enumerated under the family and the observation budget
    is "water-filled" across species (rare species contribute everything they have, common
    species are capped) so harbor seals / California sea lions don't dominate.
  * Otariidae gets two separate budgets (fur seals vs sea lions) so the fur-seal case the
    app exists for is well represented.
  * Negatives: fixed per-taxon budgets weighted toward lookalikes.
  * Observations with more photos than allowed keep only the first N photos. The split is
    done per observation later (scripts/split.py) so sibling photos never leak across splits.

Usage:
  python scripts/download.py                      # fresh build
  python scripts/download.py --from-manifest      # reproduce committed dataset
  python scripts/download.py --workers 8 --seed 42
"""
from __future__ import annotations

import argparse
import io
import math
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    ALLOWED_LICENSES,
    DATA_DIR,
    FUR_SEAL_GENERA,
    IMAGES_DIR,
    MANIFEST_CSV,
    NEGATIVE_PHOTOS_PER_OBS,
    NEGATIVE_TAXA,
    PINNIPED_TAXA,
)

API = "https://api.inaturalist.org/v1"
UA = {"User-Agent": "pinniped-classifier/1.0 (github.com/Nalomun/pinniped-classifier; educational)"}
BASE_FILTERS = {"quality_grade": "research", "photo_license": "cc0,cc-by", "photos": "true"}
API_SLEEP = 1.0  # iNat asks for <= 60 requests/min

_session = requests.Session()
_session.headers.update(UA)
_last_call = [0.0]


def api_get(path: str, **params):
    """GET with polite rate limiting and retries."""
    for attempt in range(6):
        wait = API_SLEEP - (time.time() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.time()
        try:
            r = _session.get(f"{API}/{path}", params=params, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"{r.status_code}")
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as e:
            backoff = 2 ** attempt
            print(f"[api] {path} failed ({e}); retrying in {backoff}s")
            time.sleep(backoff)
    raise RuntimeError(f"API call failed repeatedly: {path} {params}")


# ---------------------------------------------------------------------------
# Taxon resolution (by name, never hardcoded IDs)
# ---------------------------------------------------------------------------
def resolve_taxon(name: str, rank: str | None = None) -> dict:
    params = {"q": name, "per_page": 10}
    if rank:
        params["rank"] = rank
    res = api_get("taxa", **params)["results"]
    exact = [t for t in res if t["name"].lower() == name.lower()]
    if rank:
        exact = [t for t in exact if t["rank"] == rank] or exact
    if not exact:
        raise RuntimeError(f"Could not resolve taxon {name!r} (rank={rank}); got {[t['name'] for t in res]}")
    t = exact[0]
    print(f"[taxa] {name:20s} -> id={t['id']} rank={t['rank']} common={t.get('preferred_common_name')}")
    return t


def species_under(taxon_id: int) -> list[dict]:
    """Species (with licensed research-grade obs counts) under a taxon."""
    out = []
    page = 1
    while True:
        d = api_get("observations/species_counts", taxon_id=taxon_id, per_page=200, page=page, **BASE_FILTERS)
        for r in d["results"]:
            t = r["taxon"]
            if t["rank"] == "species":
                out.append({"taxon_id": t["id"], "name": t["name"], "common_name": t.get("preferred_common_name"), "count": r["count"]})
        if page * 200 >= d["total_results"]:
            break
        page += 1
    return out


def water_fill(items: list[dict], budget: int) -> dict[int, int]:
    """Allocate `budget` observations across species: rare species keep all they have,
    common species share what is left equally."""
    alloc = {}
    remaining, left = budget, sorted(items, key=lambda s: s["count"])
    for i, s in enumerate(left):
        share = math.ceil(remaining / (len(left) - i))
        take = min(s["count"], share)
        alloc[s["taxon_id"]] = take
        remaining -= take
    return alloc


# ---------------------------------------------------------------------------
# Observation sampling
# ---------------------------------------------------------------------------
def fetch_observations(taxon_id: int, n: int, rng: random.Random) -> list[dict]:
    """Return up to n observations for taxon_id.

    If the taxon has <= n licensed observations we page deterministically with id_above and
    take all. Otherwise we ask the API for random pages and dedupe until we have n.
    """
    total = api_get("observations", taxon_id=taxon_id, per_page=0, **BASE_FILTERS)["total_results"]
    seen, out = set(), []
    if total <= n:
        id_above = 0
        while True:
            d = api_get("observations", taxon_id=taxon_id, per_page=200, order_by="id", order="asc", id_above=id_above, **BASE_FILTERS)
            res = d["results"]
            if not res:
                break
            for o in res:
                if o["id"] not in seen:
                    seen.add(o["id"])
                    out.append(o)
            id_above = res[-1]["id"]
            if len(res) < 200:
                break
    else:
        max_pages = math.ceil(n / 200) * 4 + 2
        for page in range(1, max_pages + 1):
            d = api_get("observations", taxon_id=taxon_id, per_page=200, order_by="random", page=page, **BASE_FILTERS)
            for o in d["results"]:
                if o["id"] not in seen:
                    seen.add(o["id"])
                    out.append(o)
            if len(out) >= n:
                break
        rng.shuffle(out)
        out = out[:n]
    return out


def medium_url(url: str) -> str:
    return re.sub(r"/square\.(\w+)$", r"/medium.\1", url)


def rows_from_obs(o: dict, cls: str, subgroup: str, query_taxon: str, photos_per_obs: int) -> list[dict]:
    rows = []
    taxon = o.get("taxon") or {}
    user = o.get("user") or {}
    kept = 0
    for p in o.get("photos", []):
        lic = (p.get("license_code") or "").lower()
        if lic not in ALLOWED_LICENSES or p.get("hidden"):
            continue
        ext = medium_url(p["url"]).rsplit(".", 1)[-1].lower()
        ext = "jpg" if ext in ("jpg", "jpeg") else ext
        rows.append(
            {
                "photo_id": p["id"],
                "observation_id": o["id"],
                "class": cls,
                "subgroup": subgroup,
                "query_taxon": query_taxon,
                "taxon_id": taxon.get("id"),
                "taxon_name": taxon.get("name"),
                "taxon_rank": taxon.get("rank"),
                "common_name": taxon.get("preferred_common_name"),
                "license": lic,
                "attribution": p.get("attribution"),
                "user_login": user.get("login"),
                "user_name": user.get("name"),
                "photo_url": medium_url(p["url"]),
                "observation_url": f"https://www.inaturalist.org/observations/{o['id']}",
                "observed_on": o.get("observed_on"),
                "place_guess": o.get("place_guess"),
                "local_path": f"{cls}/{p['id']}.{ext}",
            }
        )
        kept += 1
        if kept >= photos_per_obs:
            break
    return rows


def build_manifest(seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    rows: list[dict] = []

    # ---- pinniped classes -------------------------------------------------
    for cls, cfg in PINNIPED_TAXA.items():
        fam = resolve_taxon(cfg["family"], "family")
        species = species_under(fam["id"])
        print(f"[{cls}] {len(species)} species with licensed research-grade observations")
        if "subgroups" in cfg:
            groups = {"fur_seal": [], "sea_lion": []}
            for s in species:
                genus = s["name"].split()[0]
                groups["fur_seal" if genus in FUR_SEAL_GENERA else "sea_lion"].append(s)
            plan = []
            for sg, budget in cfg["subgroups"].items():
                alloc = water_fill(groups[sg], budget)
                plan += [(s, alloc[s["taxon_id"]], sg) for s in groups[sg]]
        else:
            alloc = water_fill(species, cfg["target_obs"])
            plan = [(s, alloc[s["taxon_id"]], "") for s in species]
        for s, n, sg in plan:
            if n <= 0:
                continue
            obs = fetch_observations(s["taxon_id"], n, rng)
            new = [r for o in obs for r in rows_from_obs(o, cls, sg, s["name"], cfg["photos_per_obs"])]
            print(f"  {s['name']:32s} ({s['common_name']}): {len(obs):4d} obs -> {len(new):4d} photos")
            rows += new

    # ---- negatives ----------------------------------------------------------
    for name, rank, n, bucket in NEGATIVE_TAXA:
        t = resolve_taxon(name, rank)
        obs = fetch_observations(t["id"], n, rng)
        new = [r for o in obs for r in rows_from_obs(o, "not_pinniped", bucket, name, NEGATIVE_PHOTOS_PER_OBS)]
        print(f"  {name:32s}: {len(obs):4d} obs -> {len(new):4d} photos")
        rows += new

    df = pd.DataFrame(rows).drop_duplicates("photo_id").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Photo download
# ---------------------------------------------------------------------------
def download_one(row: dict) -> tuple[int, int | None, int | None, str | None]:
    dest = IMAGES_DIR / row["local_path"]
    if dest.exists() and dest.stat().st_size > 0:
        try:
            with Image.open(dest) as im:
                return row["photo_id"], im.width, im.height, None
        except Exception:
            dest.unlink(missing_ok=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = _session.get(row["photo_url"], timeout=60)
            if r.status_code == 404:
                return row["photo_id"], None, None, "404"
            r.raise_for_status()
            im = Image.open(io.BytesIO(r.content))
            im.load()
            w, h = im.size
            dest.write_bytes(r.content)
            return row["photo_id"], w, h, None
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(1.5 * (attempt + 1))
    return row["photo_id"], None, None, err


def download_all(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(download_one, r) for r in df.to_dict("records")]
        for f in tqdm(as_completed(futs), total=len(futs), desc="downloading", unit="img"):
            pid, w, h, err = f.result()
            results[pid] = (w, h, err)
    df = df.copy()
    df["width"] = df["photo_id"].map(lambda p: results[p][0])
    df["height"] = df["photo_id"].map(lambda p: results[p][1])
    errs = df["photo_id"].map(lambda p: results[p][2])
    bad = errs.notna()
    if bad.any():
        print(f"[download] {int(bad.sum())} photos failed and were dropped:")
        print(errs[bad].value_counts().head())
    return df[~bad].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-manifest", action="store_true", help="re-download photos listed in data/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="query the API and write the manifest but don't download photos")
    args = ap.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if args.from_manifest:
        df = pd.read_csv(MANIFEST_CSV)
        print(f"[manifest] {len(df)} photos listed in {MANIFEST_CSV}")
        df2 = download_all(df, args.workers)
        if len(df2) != len(df):
            print("[manifest] some photos are no longer available; they were dropped from the in-memory frame only. "
                  "Split assignments in manifest.csv are unchanged.")
        return

    df = build_manifest(args.seed)
    print("\n[manifest] photos per class:\n", df["class"].value_counts().to_string())
    print("\n[manifest] eared_seal by subgroup:\n", df[df["class"] == "eared_seal"]["subgroup"].value_counts().to_string())
    if not args.dry_run:
        df = download_all(df, args.workers)
    df.to_csv(MANIFEST_CSV, index=False)
    print(f"\n[manifest] wrote {len(df)} rows to {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
