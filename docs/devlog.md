# Devlog: Seal or Sea Lion?

A running log of how this project is going, what I've learned, and what I'm thinking about
next. Newest entry at the bottom. The README is the polished version; this is the honest one.
I build this with Claude Code doing most of the typing and me making the calls, so "we" below
means me and the robot.

---

## 2026-09-28 — Day one: the idea, the data, and two embarrassing bugs

**The idea.** A tiny classifier on my portfolio site that looks at a photo and tells you
whether it's a true seal, an eared seal, or a walrus, and roasts you if it's none of those.
The joke that makes it worth building: everyone says "seal vs sea lion", but fur seals are
*eared* seals, so the popular framing is taxonomically wrong. If I classify at the family level
(Phocidae / Otariidae / Odobenidae) the answer is always right, and the app gets to correct the
premise: "your seal is on team sea lion." It has to run entirely in the browser, because the
whole point of a portfolio piece like this is that it costs me nothing to host and nobody's
photo goes anywhere.

**Constraints I set.** Model ≤ 20 MB so it's tolerable to download on a phone. ≥ 95 % test
accuracy on the three pinniped classes, which I picked as a ship bar before knowing anything
about the data. (Spoiler: that number was optimistic. More below.)

**Data.** iNaturalist, via their API. Research-grade observations only, CC0 or CC-BY photos
only, every photo's license checked individually. Taxon IDs are looked up by name at run time
so nothing is hardcoded from memory. The sampling is "water-filled" across species so harbor
seals and California sea lions don't drown everything else, and Otariidae gets two separate
budgets so fur seals are half the class, since they're the whole point.

One thing I care about that's easy to get wrong: iNaturalist observations often have several
photos of the same animal seconds apart. If you split by photo, near-duplicates leak from
train into test and your accuracy is a lie. We split by *observation* and assert that no
observation spans two splits.

Walrus turned out to be tiny: iNaturalist has 167 licensed research-grade walrus observations
in the entire world. That's all of them, and it's about 280 photos.

**Model.** `mobilenetv3_large_100` from timm, ImageNet-21k pretrained weights, because it's
~4 M parameters once you swap the ImageNet head (≈ 17 MB fp32), about 2× cheaper than
EfficientNet-B0 in WASM, and every op it uses exports cleanly to ONNX.

**Bug one.** The first training run started with a loss of 8 on a 4-class problem, which is
absurd (random guessing is 1.4). timm initialises a new `Linear` head with
`uniform(±1/√fan_out)`, and with `fan_out = 4` that's ±0.5, so the untrained model was emitting
logits of ±20 and blasting the pretrained backbone with enormous gradients. Fixed by starting
the head near zero. Lesson: look at your initial loss.

**Bug two.** The ImageNet-21k weights we picked expect raw [0, 1] inputs (mean 0, std 1), not
the usual ImageNet mean/std. I'd been feeding them normalised inputs. Fine-tuning can recover
from that, which is exactly why it's a sneaky bug. Now mean/std come from timm's pretrained
config and get written into the checkpoint and the model metadata, so the browser can't get
this wrong either.

**Then my laptop died.** Battery at 9 %, CPU throttled to 1.5 GHz, training interrupted at
epoch 3 with 80 % accuracy. Everything got committed and pushed before it went dark.

## 2026-09-28 — Day one, later: the data was the problem

Plugged in. Before retraining I wanted to know why 80 % looked like a ceiling, so we ran a
few diagnostics instead of tuning:

- A frozen DINOv2 ViT-S (5× the parameters) with a linear probe only reached 83 %. Our
  backbone's frozen features got 78 %. If a much stronger encoder buys 5 points, the data is
  the limit, not the model.
- Actually looking at the images explained it. Research-grade iNaturalist observations are
  records of *presence*, not portraits. A big share of the pinniped photos are a speck in the
  water, an aerial shot of a colony, empty ocean where the animal just dived, a skull, or a
  carcass. iNaturalist's own annotations mark 8 % of eared-seal observations as dead, and only
  a quarter of observations are annotated at all.

So, dataset v2:

- Skip observations the community annotated as dead / track / scat / bone / feather / egg.
  (The API can't exclude a single annotation value, so this is done from the observation JSON.)
- Drop "medium" photos under 200 px; those are thumbnails of deleted originals.
- A CLIP zero-shot "is an animal clearly visible?" score for every photo. It's strongly
  bimodal. Pinniped photos below 0.95 (15 % of them) are dropped from training **and** test.
  Negatives are kept as they are, because an empty beach *is* not a pinniped. I picked the
  threshold by staring at grids of photos in each score band. Every score is committed in
  `data/manifest_unfiltered.csv` so anyone can audit the cut.

The filter isn't perfect: it drops a couple of perfectly visible animals and keeps plenty of
"lone head in the water" shots. Those heads are where most of the remaining errors live: the
family tells are ear flaps and flippers, and they're underwater.

**Results on v2.** 224 px: 86.3 % pinniped test accuracy. 288 px: 88.5 %. Fur seals, the
case the app exists for, are the *best* subgroup at 91.3 %. Walrus is the worst at 64 % recall,
because 36 test photos is 36 test photos. Square-root class-balanced sampling took it from
58 % to 64 %; more data is the real fix and iNaturalist doesn't have it.

Accuracy tracks the CLIP visibility score almost linearly: 93.5 % on portrait-like photos,
64 % on photos that barely passed the filter. The honest reading is that a 4 M-parameter model
tops out a bit under the bar on this data, and the bar was a guess anyway.

**The "unsure" threshold.** Softmax confidence alone is a bad out-of-distribution detector (a
dog can get a confident "seal"), which is why `not_pinniped` is a trained class with lookalike
negatives: otters, manatees, dolphins, penguins, bears, hippos. The threshold is the second
line of defence. If the top class is a pinniped but under 0.85, the page says "unsure". On
test that flags a quarter of pinniped photos and lifts accuracy on the rest to 92.9 %, and it
cuts confident non-pinnipeds-called-pinniped from 40 to 9 out of 500.

**Int8, the saga.** Static int8 (weights + activations) destroyed the model: −26 points.
MobileNetV3's hardswish and squeeze-excite blocks produce activation outliers that MinMax
calibration can't handle, and the percentile/entropy calibrators that would fix it crash in
onnxruntime 1.30 with numpy 2. Shipped fp32 at 16.8 MB, under budget, moved on.

**Things that didn't happen.** I started an EfficientNet-B0 comparison run and killed it: at
288 px on my CPU it was going to take three hours, and the frozen-feature probe already said
its features were worse (76 % vs 78 %). Also learned that Firefox on this laptop holds 11 GB
of RAM and the swap was full, which is why one training run was crawling at 87 seconds per
step until I halved the batch size.

**Handoff.** Phase 2 (the actual `/seal` page) went to a separate Claude Code session with a
handoff doc that spells out the exact preprocessing spec and a browser parity check against
12 committed sample images. It's live.

## 2026-09-28 — Evening: free accuracy, and a way to fit a bigger model

Asked the obvious question: can we get the number up without wrecking the page? Measured a
few things on the existing checkpoint:

| Change | Pinniped accuracy | Cost |
|---|---|---|
| Shipped model, 288 px | 88.5 % | baseline |
| Same model, inference at 352 px | **89.7 %** | 1.5× compute, no retraining |
| Same model at 384 / 416 px | 89.5 % / 89.0 % | past the sweet spot |
| Horizontal-flip test-time averaging | 88.6 % | 2× compute for nothing |
| Ensemble of the 224 and 288 models | 88.6 % | two models for nothing |
| Weight-only int8 (int8 storage, fp32 compute) | 88.1 % | download 16.8 → 4.4 MB |

The 352 px thing is the "FixRes" effect: random-resized-crop training shows the network
objects larger than a centre crop does at test time, so testing a bit above the training
resolution helps. Free lunch, taking it.

Weight-only int8 is the more interesting one. Quantizing only the weights (per-channel, with a
DequantizeLinear in front of each conv) loses 0.4 points instead of 26, and cuts the download
4×. For this model that's nice; for the *next* model it's the enabler: a backbone with 4× the
parameters can still ship under 20 MB.

The threshold is not a lever, by the way. Above 0.85 the confidently wrong answers survive
while the moderately confident right ones get flagged, so accuracy-on-answered goes *down*.

**Update, an hour later.** Re-tuned everything at 352 px and the weight-only int8 model
lost 1.5 points there, not 0.4 like at 288. Annoying. Poked at it: the damage came from the
tiny tensors (first conv, squeeze-excite, early depthwise convs) and from max-abs scaling
wasting the int8 range on single outlier weights. Keeping every tensor under 20k elements in
fp32 (0.9 MB total) and picking each channel's scale with a small MSE search over clipping
ratios brought the drop to exactly 0.00 at 5.0 MB. That's the one shipping: 89.7 %, 5 MB.

**Decision.** Ship 352 px + weight-only int8 now (small change, site session swaps the model
and re-runs its parity check). Then, as a separate experiment on Colab, try a bigger
backbone (EfficientNet-B2 or ConvNeXt-nano) at 288 px shipped as weight-only int8. My guess is
91–93 %, at the cost of roughly 1 s per photo on a phone instead of ~200 ms. Whether that
trade is worth it depends on how the page feels, which is why the small change goes first.

<!-- next entry goes here -->
