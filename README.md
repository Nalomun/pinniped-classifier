# Seal or Sea Lion?

A tiny image classifier that looks at a photo and tells you whether it's a **true seal**, an
**eared seal** (sea lion *or* fur seal), or a **walrus**, and roasts you gently when it's none of
those. It runs entirely in the browser: the photo never leaves your device.

**Live page:** <https://quinnlambert.com/seal> · **Training code:** this repo

<!-- RESULTS_BADGE -->

## Why three families instead of "seal vs sea lion"

Pinnipeds split into three families:

| Family | Common name | Members | The visual tell |
|---|---|---|---|
| Phocidae | true (earless) seals | harbor, grey, elephant, leopard, monk seals… | no ear flaps, short clawed front flippers, wriggles on its belly |
| Otariidae | eared seals | **sea lions and fur seals** | visible ear flaps, long front flippers, can "walk" on rotated hind flippers |
| Odobenidae | walrus | just the walrus | tusks, whiskers, sheer size |

"Seal vs sea lion" is the popular framing, but it's taxonomically wrong: **fur seals are
otariids**, i.e. they are on team sea lion, not team seal. Classifying at the family level
means the answer is always correct, and the question mark in the name lets the result
correct the premise ("your 'seal' is actually on team sea lion").

## Data: iNaturalist, CC0 / CC-BY only

All images come from [iNaturalist](https://www.inaturalist.org) via its public API
(`scripts/download.py`). Filters:

* **research-grade** observations only (community-verified IDs),
* photos licensed **CC0 or CC-BY** only (each photo's license is checked individually, not
  just the observation's),
* taxon IDs are looked up **by name at run time**, never hardcoded.

Every image's observation ID, species, license, photographer attribution and URL is recorded
in [`data/manifest.csv`](data/manifest.csv). The raw images are not committed;
`python scripts/download.py --from-manifest` re-downloads exactly this dataset.

### Sampling policy

iNaturalist is very skewed (harbor seals and California sea lions are most of the pinniped
observations). To get species and geographic variety, the observation budget for each class is
**water-filled across species**: rare species contribute everything they have, common species
are capped at an equal share of what's left. Otariidae gets two separate budgets, one for fur
seals and one for sea lions, because fur seals are the case this whole app exists for.

<!-- DATA_TABLE -->

The `not_pinniped` class is deliberately weighted toward **lookalikes**: otters, manatees and
dugongs, dolphins and whales, penguins, bears, hippos, sea turtles, beavers, crocodilians. Then
a smaller share of other common animals (dogs, cats, deer, bovids, gulls, cormorants) and a
small slice of plants, fungi and insects so a photo of your lunch also gets roasted. No people.

### Splitting by observation, not by photo

An iNaturalist observation often carries several photos of the same animal taken seconds
apart. If those were split independently, the test set would contain near-duplicates of
training images and the accuracy would be inflated. `scripts/split.py` therefore assigns
**whole observations** to train/val/test (70/15/15, stratified by class and subgroup, fixed
seed) and asserts that no observation spans two splits.

## Model

`mobilenetv3_large_100` from `timm`, initialised from the `miil_in21k_ft_in1k` weights
(ImageNet-21k pretraining, then ImageNet-1k), fine-tuned end to end at 224×224.

Why this one and not `efficientnet_b0`:

* **Size.** Once the 1000-way ImageNet head is swapped for a 4-way head it has 4.2M
  parameters → ~17 MB as fp32 ONNX, ~4.5 MB as int8. Under the 20 MB budget either way.
* **Speed in WASM.** ~0.22 GFLOPs per image vs ~0.39 for EfficientNet-B0; roughly 2× cheaper
  on a phone running `onnxruntime-web` without a GPU.
* **Exports cleanly.** Hardswish / hardsigmoid / squeeze-excite all map to plain ONNX opset-17
  ops that ORT-web's WASM backend supports.
* **Better starting features.** ImageNet-21k has far more animal categories than ImageNet-1k,
  which matters for a fine-grained mammal task with a few thousand images.

Training recipe (`scripts/train.py`): AdamW, cosine schedule with one warmup epoch, lower LR
on the backbone than the head, label smoothing 0.1, mixed precision on GPU. Augmentation is
random-resized-crop (scale 0.35–1), horizontal flip, colour jitter and light random erasing,
because water, sand, rock and ice backgrounds vary enormously across iNat photos. Model
selection is by validation **pinniped** accuracy, since that's the ship bar.

Preprocessing at inference (also written into `export/model_meta.json` for the browser):
resize the shorter side to 256 (bilinear) → centre-crop 224 → scale to [0,1] → normalise
with ImageNet mean/std → NCHW, RGB.

## Results

<!-- RESULTS -->

## Reproduce

Locally (CPU is fine for everything except training, which takes ~40 min for 12 epochs on
an 8-core laptop; use the Colab notebook for a GPU):

```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 torchvision==0.29.0
pip install -r requirements.txt

python scripts/download.py --from-manifest   # ~8k photos, ~800 MB; exact committed dataset
# python scripts/download.py && python scripts/split.py   # ...or draw a fresh sample
python scripts/train.py                      # -> outputs/best.pt
python scripts/eval.py                       # -> reports/metrics.json + figures, tunes threshold on val
python scripts/export.py                     # -> export/pinniped.onnx (+ int8 if it survives), model_meta.json
python scripts/parity.py                     # PyTorch vs ONNX Runtime on committed sample images
python scripts/predict.py some_photo.jpg     # try it
```

On Colab: open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), pick a GPU
runtime, run all cells. It clones this repo, re-downloads the dataset from the manifest,
trains, evaluates, exports and zips the artefacts.

## Repo layout

```
scripts/common.py     class order, taxa config, preprocessing spec (single source of truth)
scripts/download.py   iNaturalist sampling + download -> data/manifest.csv
scripts/split.py      observation-grouped 70/15/15 split -> manifest `split` column
scripts/train.py      timm fine-tuning
scripts/eval.py       metrics, confusion matrix, worst errors, threshold tuning
scripts/export.py     ONNX + int8 + model_meta.json
scripts/parity.py     PyTorch vs onnxruntime check on reports/parity_samples/
scripts/predict.py    CLI inference with the ONNX model (torch-free preprocessing)
notebooks/train_colab.ipynb
data/manifest.csv     every image: observation, species, license, attribution, URL, split
export/               pinniped.onnx, pinniped_int8.onnx, model_meta.json
reports/              metrics.json, figures, per_species.csv, parity.json, parity_samples/
```

## Attribution and license

Photos are by iNaturalist contributors under CC0 or CC-BY; `data/manifest.csv` lists the
photographer and license for every image, and the few sample images committed under
`reports/parity_samples/` carry their attribution in `reports/parity.json`. Thanks to the
iNaturalist community, without whom this wouldn't exist.

Code is MIT licensed (see [LICENSE](LICENSE)). The trained model weights are released under
the same terms; the training images themselves are not redistributed here.
