# Handoff: build the `/seal` page on quinnlambert.com ("Seal or Sea Lion?")

You are picking up **Phase 2** of a two-phase project. Phase 1 (training an image
classifier) is finished and lives in a separate repo. Your job is the website integration
only. Do not modify the training repo except to read from it.

## The two repos

| | Path | Notes |
|---|---|---|
| Site (you edit this) | `/home/quinn/Documents/myhome` | Next.js 14, App Router (`src/app`), Tailwind 3.4, deployed on Vercel. Has its own `CLAUDE.md`: read it first. Seal mascot assets are in `public/seals/`. |
| Training repo (read only) | `/home/quinn/Documents/pinniped-classifier` | Also at <https://github.com/Nalomun/pinniped-classifier>. Read its `README.md` for the full story. |

## What the app is

An image classifier that tells the visitor whether a photo shows a **true seal**, an
**eared seal** (sea lions + fur seals), or a **walrus**, and gives a funny response when it's
none of those. It runs **entirely in the visitor's browser**; photos never leave the device.

- App name: **Seal or Sea Lion?**
- Route: `/seal`
- GitHub link for the project card: `https://github.com/Nalomun/pinniped-classifier`

**Why three families, not "seal vs sea lion":** pinnipeds split into Phocidae (true seals),
Otariidae (eared seals: sea lions *and* fur seals) and Odobenidae (walrus). "Seal vs sea lion"
is the popular framing but taxonomically wrong, because fur seals are otariids. Classifying at
family level means the answer is always correct, and the name's question mark lets the result
correct the premise ("your 'seal' is on team sea lion"). The page should explain this in a
short footnote, not just assert it.

## How to work

1. **Read before proposing.** Read the site repo's `CLAUDE.md`, then its structure and
   conventions. Report back: App Router vs Pages Router (it is App Router, confirm), styling
   approach, how the **"Things I make"** section and its project cards / status tags are
   structured, how routes are organised, and what the seal mascot assets are.
2. **Propose a plan before editing anything**, including which files you'll add or touch.
   Wait for approval.
3. Explain key design choices briefly as you go. Report results honestly; if the browser
   parity check doesn't match, say so and show the numbers.
4. Do not commit or push unless asked.

## Model artefacts (copy from the training repo)

| Training repo path | Purpose |
|---|---|
| `export/pinniped_w8.onnx` | **the model to ship: 5.0 MB**, int8 weights with fp32 compute, ONNX opset 17. sha256 in `model_meta.json` (`recommended_file`). |
| `export/pinniped.onnx` | the same model as plain fp32, 16.8 MB; fallback / reference only. |
| `export/model_meta.json` | class order, input spec, preprocessing steps, mean/std, threshold. **The single source of truth for inference.** Load it or mirror it exactly. |
| `reports/parity.json` | per-image expected probabilities for 12 sample images (from PyTorch and ONNX Runtime, which agree to 4e-7) |
| `reports/parity_samples/*.jpg` | those 12 images (CC0/CC-BY, attribution in `parity.json`) |

Put the model in the site's static assets (e.g. `public/models/pinniped_w8.onnx`). Make sure
it is served with long cache headers and is **only fetched on `/seal`**.

## Exact inference spec (mirrors `model_meta.json`)

- Input tensor name `input`, shape `[1, 3, 352, 352]`, float32, **NCHW**, **RGB**.
- Output tensor name `logits`, shape `[1, 4]`, raw logits: apply **softmax**.
- Class order (index → label): `0 true_seal`, `1 eared_seal`, `2 walrus`, `3 not_pinniped`.
- Preprocessing, in order:
  1. decode the image **respecting EXIF orientation** (phone photos are often rotated; use
     `createImageBitmap(file, { imageOrientation: "from-image" })` or equivalent),
  2. **resize so the shorter side is 402 px** (bilinear), keeping aspect ratio,
  3. **centre-crop 352×352**,
  4. convert to float and **divide by 255**,
  5. normalise with **mean `[0, 0, 0]`, std `[1, 1, 1]`** (these weights expect raw [0,1]
     inputs; this is a no-op, but implement it from the meta file so a future model with
     ImageNet mean/std keeps working),
  6. lay out as NCHW (all R values, then all G, then all B).
- **Decision rule:** `top = argmax(probs)`. If `top` is `not_pinniped` → the "this ain't no
  seal" response. Else if `probs[top] < threshold` (**threshold = 0.85**) → the "unsure"
  response. Else → the class result with `probs[top]` shown as a percentage.

Implement the preprocessing with a canvas (`drawImage` with `imageSmoothingQuality: "high"`)
and read pixels with `getImageData`. Skip the alpha channel.

### Browser parity check (part of "done")

Run the 12 images in `reports/parity_samples/` through the page's pipeline and compare with
the `torch` probabilities in `reports/parity.json`. Canvas resampling is not bit-identical
to PIL's bilinear resize, so expect small deltas: the **argmax must match on all 12** and
**max |Δ probability| should be below ~0.05**. If it isn't, the preprocessing is wrong (common
culprits: BGR/RGB order, HWC instead of CHW, forgetting the /255, resizing to 288 directly
instead of 329-then-crop, ignoring EXIF orientation). Note two of the samples are *expected*
to be model errors (`true_seal_144793.jpg` → eared_seal, `walrus_2123671.jpg` → not_pinniped);
match the model, not the true label. `parity.json` also has `input_tensor_stats` (mean, std,
first 5 values of the preprocessed tensor) to help debug a mismatch.

## Inference runtime

- `onnxruntime-web`, **WASM backend** (no WebGPU requirement; WebGPU may be offered as an
  optional accelerator but WASM must work). Use the multi-threaded WASM build if
  cross-origin isolation headers are feasible on Vercel; otherwise single-threaded is fine
  (~150–400 ms per image on a phone at 288 px).
- **Lazy-load the runtime and the model only on `/seal`** (dynamic `import("onnxruntime-web")`
  inside the page or a client component; the session is created on first use). No other page's
  bundle or load may change. Verify with the Next.js bundle analysis or by checking that no
  `ort-*` chunk is referenced from other routes.
- The `.wasm` binaries must be reachable: either copy them to `public/` and set
  `ort.env.wasm.wasmPaths`, or point `wasmPaths` at the matching-version jsDelivr/unpkg URL.
  Pin the `onnxruntime-web` version.
- Show a loading state with progress while the model downloads (fetch with a
  `ReadableStream` to get progress, then pass the `ArrayBuffer` to `InferenceSession.create`).
  Cache the session for the page's lifetime.

## Input

- File upload, drag-and-drop, paste from clipboard (`paste` event, `clipboardData.files`),
  and **mobile camera capture** (`<input type="file" accept="image/*" capture="environment">`).
- iPhones may hand over HEIC; browsers generally can't decode it via canvas. Detect the decode
  failure and show a friendly message asking for a JPEG/PNG (most phones convert
  automatically when `accept="image/*"` is used; just handle the failure gracefully).
- State plainly on the page that **the image never leaves your device**.

## Result display

- Class label and confidence %.
- **The visual tell** for the class:
  - eared seal: visible ear flaps, long front flippers, can "walk" on rotated hind flippers
  - true seal: no ear flaps, short clawed front flippers, wriggles on its belly
  - walrus: tusks, whiskers, sheer size
- **Fun copy.** Draft **several variants per case** for review before finalising:
  - eared seal: the "your seal is actually on team sea lion" angle (mention fur seals are
    otariids too)
  - true seal: a "genuine, certified seal" angle
  - walrus: a surprised "wait, it's a walrus"
  - not_pinniped: a "this ain't no seal" roast
  - unsure (pinniped but below threshold): a hedging line ("something's flippering around in
    there but I'm not calling it")
- A short **footnote on the three pinniped families** so the taxonomy is explained.
- A small honesty line somewhere on the page: trained on iNaturalist photos of wild animals;
  about 88 % accurate on a held-out test set (93 % on clear, close-up photos); heads-only-in-
  water shots and walruses are the hard cases. The training-repo README has the exact numbers
  if you want to link to it.

## Design

- Match the site's existing deep purple/black theme and components. Use the seal mascot from
  `public/seals/` if it fits naturally (e.g. as the "thinking" state while inferring).
- Responsive and mobile-first: camera capture is a main use case. Big tap targets, result
  readable without scrolling on a phone.

## Project listing

Add a card for this project to the **"Things I make"** section following the existing card
pattern, with the appropriate status tag (`live` once deployed). Link it to `/seal` and to
`https://github.com/Nalomun/pinniped-classifier`.

## Model update (2026-09-28, after the page went live)

The model file and its preprocessing spec changed. If the page already exists, this is a
swap, not a rebuild:

| | Before | **After** |
|---|---|---|
| File | `export/pinniped.onnx`, 16.8 MB fp32 | **`export/pinniped_w8.onnx`, 5.0 MB** (int8 weights, fp32 compute; `pinniped.onnx` still exists as the fp32 reference) |
| sha256 | `a9f32cdb...` | `e120a7afd84683cdaa75e0a7b939a18ea62e4a1c28a7a39e72580a3d87fca74e` |
| Input shape | `[1, 3, 288, 288]` | **`[1, 3, 352, 352]`** |
| Resize shorter side to | 329 | **402** |
| Centre crop | 288 | **352** |
| Mean / std | 0 / 1 | 0 / 1 (unchanged) |
| Threshold | 0.85 | 0.85 (re-tuned at 352, came out the same) |
| Pinniped test accuracy | 88.5 % | **89.7 %** |

Same network, run at a higher resolution than it was trained at (a known trick), stored with
int8 weights. Inference is ~1.5x the compute of before; on a phone that's roughly 200-300 ms.

What to do: copy the new `pinniped_w8.onnx` and the new `model_meta.json` (its
`recommended_file` now says `pinniped_w8.onnx`), and read **every** preprocessing number from
`model_meta.json` rather than from constants so the next swap is free. `reports/parity.json`
and `reports/parity_samples/` were regenerated at 352 px: re-run the browser parity check
against them (same rule: argmax must match on all 12, |dp| < ~0.05). `parity.json` now also
carries `img_size`, `mean` and `std`, and its `onnx_w8` entries are the expected outputs of
the int8-weight file specifically (they differ from `torch` by up to ~0.007). The weight-only
model uses `DequantizeLinear` nodes (opset 17), which `onnxruntime-web`'s WASM backend
supports; if the session fails to create, fall back to `pinniped.onnx` and report it.

## Done when

- `/seal` works locally and in a Vercel preview.
- The 12 parity images give the same argmax as `reports/parity.json` and probabilities
  within ~0.05.
- No other page's load is affected (no `onnxruntime-web` chunk outside `/seal`).
- The card appears in "Things I make".
- Copy variants have been reviewed and one set chosen.
