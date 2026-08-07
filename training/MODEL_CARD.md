# Model card — `best.onnx`

Fall / sit / stand detector used by the Human Anomaly Detection system, both
server-side (`inference.js`) and in the browser
(`frontend-new/src/lib/detect/`).

> **Accuracy has not been measured.** The training notebook's `model.val()`
> output was never saved, so there is no recorded mAP for the file that is
> actually deployed. The tables below are marked TBD and are filled in by
> running sections 3, 3a and 3b of `train_fall_detection.ipynb`. Until then,
> treat any accuracy claim about this model — including in the portfolio and
> the READMEs — as unsupported.

## What it is

| | |
| --- | --- |
| Architecture | YOLOv8n (nano) |
| Task | Object detection, 3 classes |
| Classes | `0: fall`, `1: sit`, `2: stand` |
| Input | `images`, float32 `[1, 3, 640, 640]`, RGB CHW, `/255`, letterboxed with grey `rgb(114,114,114)` |
| Output | `output0`, `[1, 7, 8400]`, channel-major, **no NMS**, sigmoid already applied |
| Export | opset 12, `simplify=True`, `nms=False`, `dynamic=False` |
| File | 11.7 MB (12,266,856 bytes) |
| Exported | 2026-07-11, ultralytics 8.4.92 |

Class order is the model's own, from its embedded ultralytics metadata. It is
restated in `CLASS_NAMES` (`render.yaml`, the Render dashboard) and in
`frontend-new/src/lib/detect/constants.js`, because ONNX custom metadata is not
readable through onnxruntime-node at runtime. **All three must agree.**

## Training data

[`swook1015-ijop5/stand-sit-fall`](https://universe.roboflow.com/swook1015-ijop5/stand-sit-fall)
on Roboflow Universe, version 2 — 8,340 images. The dataset's own published
baseline is ~81% mAP50 / 85.2% precision / 74.6% recall. That is a sanity
reference for a training run, **not** a measurement of this model.

Recipe: `yolov8n.pt`, `imgsz=640`, `epochs=100`, `patience=20`, `batch=16`.
Nano was chosen deliberately — the backend runs on a 512 MB free-tier host that
has already had to be worked around for OOM crashes and 30-second inference
(see commits `90b4fb4`, `bfa3d9d`).

## Accuracy — TBD

Roboflow **test** split, from notebook section 3:

| class | precision | recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| fall | TBD | TBD | TBD | TBD |
| sit | TBD | TBD | TBD | TBD |
| stand | TBD | TBD | TBD | TBD |
| **all** | TBD | TBD | TBD | TBD |

Webcam-domain set, from notebook section 3b:

| class | precision | recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| — | TBD | TBD | TBD | TBD |

The gap between those two tables is the number that matters most for the
deployed demo, and it needs footage recorded on the camera the demo actually
runs on. The training data is staged and framed roughly like CCTV — a whole
body at some distance. A laptop webcam sees a close-up torso at desk height,
often cropped at the waist. Detector accuracy off its training distribution is
not predictable from its accuracy on it.

## Operating point

`DETECTION_CONFIDENCE=0.4`, IoU 0.45 for NMS, applied as a single floor across
all three classes.

**This 0.4 is a guess, not a choice** — it predates any measurement. Notebook
section 3a derives per-class thresholds from the precision-recall curves, using
F2 for `fall` (a missed fall costs more than a false alarm) and F1 for `sit`
and `stand`. Since `inference.js` applies one floor to every class, set
`DETECTION_CONFIDENCE` to the lowest chosen threshold; the frontend's slider
can tighten it further per session.

## Known failure modes

The important one is structural, not a defect in the weights: **this model
classifies each frame in isolation, and a fall is not a single-frame event.**
It is a transition into a sustained state. In one frame, a person who has
fallen, a person bending to pick something up, and a person lying on a sofa
look the same — so a per-frame detector cannot separate them, however well
trained.

That is why `frontend-new/src/lib/detect/tracker.js` exists. It associates
detections into tracks, votes each track's class over a 7-result window, and
only confirms a fall after it has persisted for 1.2 s, with box aspect ratio as
a soft corroborating signal. Bare model output flickers between classes and
fires on anyone who bends over; that behaviour is a property of the framing,
and the tracker is the fix. `npm run tracker` in `frontend-new` pins down what
it is supposed to buy.

Also expected:
- Close-up torsos cropped at the waist (typical laptop webcam framing) are out
  of distribution — see the domain gap above.
- A fall directly toward or away from the lens keeps a tall bounding box, so
  the aspect-ratio corroboration does not help there. It is deliberately a
  bonus and never a gate, for exactly this reason.
- Multiple overlapping people are only separated as well as class-aware NMS at
  IoU 0.45 allows.

## Latency

| where | per frame |
| --- | --- |
| onnxruntime-node, 1 thread, Ryzen 5 5600U | 273–311 ms (median 287 ms, n=15) |
| onnxruntime-node on Render free tier | not measured; inference is throttled to one frame per 500 ms regardless |
| onnxruntime-web, WASM, 1 thread | not measured |
| onnxruntime-web, WebGPU | not measured |

Only the first row is measured. The browser figures need a real pass on desktop
Chrome (WebGPU), desktop Firefox (WASM — no WebGPU in stable) and a mid-range
Android before being quoted anywhere.

## If accuracy needs to improve

In order of expected value per unit of effort:

1. **Measure first** (sections 3 and 3b). Everything below is guesswork without
   a baseline, and the tracker may already have closed the gap that prompted
   the question.
2. **Choose thresholds from the curve** (section 3a). Free, and per-class.
3. **Fine-tune on a few hundred labelled frames from the deployment domain.**
   If the domain gap is large, this beats any architecture change.
4. **Only then consider a larger backbone.** yolov8s roughly doubles inference
   cost, which the 512 MB host cannot absorb — though it matters much less now
   that the browser can carry the live path.

Separately, **int8 dynamic quantisation** is worth evaluating for the browser
build: fp32 weights barely compress (11.7 MB → 10.1 MB gzip), and quantisation
would take the download to roughly 3 MB. It needs the baseline from section 3
to measure what accuracy it costs.
