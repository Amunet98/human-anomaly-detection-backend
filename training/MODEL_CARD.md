# Model card — `best.onnx`

Fall / sit / stand posture classification for the Human Anomaly Detection
system, used both server-side (`inference.js`) and in the browser
(`frontend-new/src/lib/detect/`).

**As of 2026-08-08 this is a two-stage system, not a posture detector.** A
COCO-pretrained pose model finds people and their joints; posture is then decided
from keypoint geometry in `posture.js`. The previous single-stage 3-class
detector is described under "Previous model" below, along with the measurements
that motivated replacing it.

## What it is

| | |
| --- | --- |
| Architecture | YOLOv8n-pose (nano), COCO-pretrained, **unmodified** — no fine-tuning |
| Task | Person detection + 17 COCO keypoints |
| Model classes | 1 (`person`). The three postures are **not** model outputs |
| Posture | Derived from geometry in `posture.js` / `frontend-new/src/lib/detect/posture.js` |
| Input | `images`, float32 `[1, 3, 640, 640]`, RGB CHW, `/255`, letterboxed with grey `rgb(114,114,114)` |
| Output | `output0`, `[1, 56, 8400]`, channel-major = 4 box + 1 person-conf + 17x3 keypoints, **no NMS** |
| Export | opset 12, `simplify=True`, `nms=False`, `dynamic=False` |
| File | 12.9 MB (13,514,574 bytes) |
| Parameters | 3,289,964 |
| Exported | 2026-08-08, ultralytics 8.4.116, from the official `yolov8n-pose.pt` release asset |

Preprocessing is byte-identical to the old model's, so `letterbox.js` and the
server's `preprocess()` are unchanged. Only the decode and everything after it
moved.

`CLASS_NAMES` in `render.yaml` / `.env` / `constants.js` is now the *posture*
vocabulary rather than the model's class list. It still has to agree across all
three, because `posture.js` emits exactly those three strings.

## Why the architecture changed

The old model was a yolov8n trained on 8,340 images to classify posture directly.
Measured on the fixture set (`frontend-new/scripts/eval-fixtures/`):

| | clean | perturbed | macro-F1 (perturbed) |
| --- | --- | --- | --- |
| old 3-class detector | 5/5 (100%) | 23/30 (**76.7%**) | 0.763 |
| **yolov8n-pose + geometry** | 5/5 (100%) | **30/30 (100%)** | **1.000** |

"Perturbed" replays every fixture through hflip, grayscale, darken-40%,
blur-3px, downscale-320w and centre-crop-80%. Both models are perfect on clean
images; only the sweep separates them.

The old model's per-class breakdown under perturbation showed the actual defect:

| class | precision | recall |
| --- | --- | --- |
| fall | 1.000 | 0.667 |
| sit | **0.545** | **1.000** |
| stand | 0.818 | 0.750 |

`sit` never missed and was wrong nearly half the time it fired — a fallback
attractor. Blur or downscale a standing subject and the answer became `sit` at
0.63–0.88 confidence. That is the signature of a model keyed on scene appearance
(bench, floor, indoor room) rather than body configuration: degrade the
appearance and it falls back to its most common training class. More data or a
bigger backbone would not have fixed it, because the shortcut was available in
the task framing itself.

Keypoint geometry does not have that failure mode. A shoulder-to-hip angle is
the same whether the frame is sharp or blurred, colour or greyscale, 1408px or
320px wide. What can still fail is *finding* the joints — but that job is done by
a model trained on COCO's ~200k person instances with heavy augmentation, a far
better-conditioned problem than 3-way posture classification from 8,340 images.

A useful side effect: the pose model finds every person in frame. The street-fall
fixture now returns the faller **plus four background pedestrians**; the old
model saw one object. `filterTinyBoxes` drops the distant ones (they are
4.6–12.4% of the largest box, under the 15% floor) and `analyzeBuffer` ranks any
`fall` above everything else when reducing to a single `top`, so a calm bystander
cannot outscore someone mid-fall.

## How posture is decided

`posture.js`, from midpoints of the shoulder / hip / knee / ankle pairs. A joint
below confidence 0.5 counts as missing.

| feature | meaning |
| --- | --- |
| `torsoAngle` | shoulder-mid to hip-mid, degrees off vertical. 0 = upright, 90 = horizontal |
| `kneeDrop` | `(kneeY - hipY)` over torso length. Large = hips well above knees |
| `kneeAngle` | interior angle at the knee |
| `thighShinRatio` | projected thigh length over projected shin length |
| `aspect` | box w/h, tiebreak only, never a gate |

Order: `torsoAngle >= 50` (or `aspect >= 1.5` with `torsoAngle >= 30`, covering a
body lying toward the lens) is a **fall**; else `thighShinRatio < 0.75` is a
**sit**; else `kneeDrop < 0.5` / `kneeAngle < 150` is a **sit**; else **stand**.

### Why thigh foreshortening is a separate rule

The femur and tibia are within ~10% of each other in real length, so their
*projected* ratio is ~1 whenever both lie in the image plane — which is what
standing with vertical legs means. A thigh much shorter than its own shin can
only mean the thigh points toward or away from the lens.

This exists because of a real miss (2026-08-08): a woman seated facing the camera
with her legs stretched forward was classified `stand 63%`. Both other leg
features genuinely read as standing — knees well below hips (`kneeDrop` 0.64) and
an almost straight leg (`kneeAngle` 172°). Only the foreshortened thigh (0.58×
shin) distinguished it. Fixture `bench-sit-frontal.jpg` and a unit test in
`posture-check.mjs` (built from the real keypoints) guard it.

The two mechanisms are complementary, not redundant:

| sit viewed | thigh in image plane? | caught by |
| --- | --- | --- |
| side-on (thigh horizontal) | yes | `kneeDrop` + `kneeAngle`; ratio stays ~1 |
| front-on (legs extended) | no | `thighShinRatio` only |

Measured: standing **1.00 / 1.01 / 1.07 / 1.08 / 1.11**, front-on seated **0.47 /
0.58**, side-on seated 1.38. The 0.75 threshold has ~0.17 of margin below and
~0.25 above.

It is applied as a **gate, not a vote**. Left as one signal among three it would
lose 2–1 to `kneeDrop` and `kneeAngle` in precisely the case it exists to catch,
since those two are what fail there. The justification for overriding them is
that this is a statement about projection geometry rather than a correlation — no
standing pose puts a thigh at 0.58 of its own shin. It is also robust to camera
pitch, being a ratio of two adjacent segments: a high or low camera foreshortens
thigh and shin together and cancels out.

It runs *after* the fall check, so it cannot affect fall detection.

Measured separation on the fixtures — upright subjects at 1.6/1.7/14.0 deg
versus falls at 57.6 and 72.3 deg; seated knee 87 deg versus standing 175-179 deg.
Both gaps are wide, which is why the thresholds are round numbers rather than
fitted ones.

**Occlusion tiers**, because two of five fixtures have no visible legs:

Joints below `KP_CONF_THRESHOLD` (0.65) count as missing. That number is
load-bearing: leg-joint confidence is sharply bimodal — genuine joints at 0.79
and 0.80–1.00, joints the model is *guessing* at (legs behind a sofa) at
0.42/0.46/0.47/0.50 and below 0.30, with the 0.50–0.79 band empty. The original
0.5 sat directly on the guessing cluster, so two near-identical photos of the
same standing woman returned `sit` and `stand`: one hidden knee scored exactly
0.50 in one shot and 0.42 in the other, which was enough to move her between
tier B and tier C. Guarded by fixtures `lodge-group-a/b.jpg`.

| tier | available | behaviour | confidence x |
| --- | --- | --- | --- |
| A | hips + knees + ankles | full rule set | 1.0 |
| B | hips + knees | `kneeDrop` only | 0.85 |
| C | hips only | **cannot tell sit from stand** — returns `stand` | 0.6 |
| D | no hips | no torso vector — returns `stand` | 0.4 |

The discounts are multiplicative and feed `tracker.js`'s confidence-weighted
vote directly, so a tier-C guess from a waist-up frame cannot outvote a clean
full-body read of the same person.

> **Known limitation — tier C is a real gap, not a solved case.** A waist-up crop
> of someone standing and a waist-up crop of someone at a desk are geometrically
> identical, and the classifier returns `stand` for both. That is the safe
> non-alarming default, not a correct answer. It matters specifically for the
> close-range laptop-webcam framing this demo runs in, which is the same
> domain-gap concern flagged for the old model below. Resolving it needs a signal
> geometry does not carry — scene context, or the transition into the pose.

## Measured accuracy — top-1 posture

From `frontend-new/scripts/eval-check.mjs` (`npm run eval:robust`), against the
labelled fixture set in `frontend-new/scripts/eval-fixtures/`. Measured
2026-08-08.

| | clean | perturbed |
| --- | --- | --- |
| accuracy | **8/8 (100%)** | **48/48 (100%)** |
| macro-F1 | 1.000 | 1.000 |

Per-class, under perturbation: `fall` P=1.000 R=1.000, `sit` P=1.000 R=1.000,
`stand` P=1.000 R=1.000. Survival is 5/5 for every one of the six perturbations
individually.

**Read this with the caveat it deserves.** The fixture set is eight images. A
perfect score on eight images through six perturbations is 48 trials, not 48
independent samples, and it does not mean the system is perfect — it means the
fixture set no longer discriminates and has to grow before it can say anything
more. What the number does support is the *comparison*: the same trials that
the old model failed 7 of, this one passes, and the failures it fixed were the
systematic kind (every `stand` collapsing to `sit` under blur) rather than
scattered noise.

The harness now also runs two **pass/fail** checks that top-1 accuracy cannot
express: `expectedAll` verifies every person in a multi-subject frame, and
`consistentWith` asserts that a pair of near-identical photos does not disagree.
Both were added because a real bug hid behind top-1 — the standing woman flipped
between `sit` and `stand` across two shots while top-1 stayed `sit` on both.

Note fixture resolution is deliberate: `lodge-group-a/b.jpg` are stored at native
2048px because downscaling them to 960px moved the offending keypoint confidence
off the boundary and made them stop reproducing the bug entirely.

Grow the set to 15-20 images per class — deliberately including desk-webcam
framing, which the tier-C limitation above predicts will be the weak spot —
before quoting a figure anywhere load-bearing.

What this measurement does **not** cover, unchanged from before: box
localisation quality, multi-person scenes scored per-person rather than top-1,
and any video/temporal behaviour (that is `npm run tracker`'s job).

## Previous model (superseded 2026-08-08)

Kept because the comparison above depends on it, and because the deployed file
carried these properties until this date.

| | |
| --- | --- |
| Architecture | YOLOv8n, 3 classes `0: fall`, `1: sit`, `2: stand` |
| Output | `output0`, `[1, 7, 8400]` |
| File | 11.7 MB (12,266,856 bytes) |
| Exported | 2026-07-11, ultralytics 8.4.92 |
| Training data | [`swook1015-ijop5/stand-sit-fall`](https://universe.roboflow.com/swook1015-ijop5/stand-sit-fall) v2, 8,340 images |
| Recipe | `yolov8n.pt`, `imgsz=640`, `epochs=100`, `patience=20`, `batch=16` |
| Detection mAP | never measured — `model.val()` output was never saved |

The dataset's own published baseline was ~81% mAP50 / 85.2% precision / 74.6%
recall. That was always a sanity reference for a training run, **not** a
measurement of the deployed file, and any portfolio or README figure quoting it
as one was unsupported.

Its decoder survives as `decodeYolov8` in `postprocess.js`. It is unused by the
application and kept only so `eval-check.mjs` can still score the old weights for
comparison; delete both together if that comparison stops being interesting.

`train_fall_detection.ipynb` trains this superseded model. Note its cell 3 says
`VERSION = 1` while the shipped weights' ONNX metadata says
`stand-sit-fall-2/data.yaml`, so re-running it as-is would not even reproduce
what was deployed. The current model needs no training notebook at all — it is
the official COCO release weights, exported unmodified.

## Operating point

`DETECTION_CONFIDENCE=0.4` is now the **person-detection** floor: it gates the
pose model's `personConf` before any posture is computed. It no longer has
per-class meaning, because the model has one class.

Posture confidence is a separate, downstream quantity — `personConf` times the
geometric tier and margin factors in `posture.js` — and it is what
`tracker.js`'s `FALL_ENTER_CONF=0.55` / `FALL_EXIT_CONF=0.35` gate against. The
frontend's slider still tightens the person floor per session.

IoU 0.45 for NMS, class-agnostic (trivially so now — one class), plus
`MIN_BOX_AREA_RATIO=0.15` relative to the largest box in frame.

The per-class threshold work sketched in the old notebook section 3a no longer
applies as written: there are no per-class scores to threshold. The equivalent
knob is now the geometric boundaries in `posture.js`, pinned by
`npm run posture`.

## Known failure modes

The important one is structural, not a defect in the weights: **this model
classifies each frame in isolation, and a fall is not a single-frame event.**
It is a transition into a sustained state. In one frame, a person who has
fallen, a person bending to pick something up, and a person lying on a sofa
look the same — so a per-frame detector cannot separate them, however well
trained.

That is why `frontend-new/src/lib/detect/tracker.js` exists. It associates
detections into tracks, votes each track's class over a 7-result window, and
only confirms a fall after it has persisted for 1.2 s. Bare per-frame output
flickers between classes and
fires on anyone who bends over; that behaviour is a property of the framing,
and the tracker is the fix. `npm run tracker` in `frontend-new` pins down what
it is supposed to buy.

Also expected:
- **Close-up torsos cropped at the waist** (typical laptop webcam framing) hit
  tier C, where sit and stand are geometrically indistinguishable and the
  classifier returns `stand`. This is the single most likely failure in the
  demo's actual deployment domain.
- A fall directly toward or away from the lens foreshortens the torso, so
  `torsoAngle` understates it. The `aspect >= 1.5` clause exists for that case,
  but a fall that is *both* foreshortened and narrow-boxed will still be missed.
- Overlapping people are only separated as well as NMS at IoU 0.45 allows. The
  pose model does detect far more people per frame than the old one, so this
  path now gets exercised where before it did not.
- `MIN_BOX_AREA_RATIO=0.15` suppresses a genuinely small second subject standing
  well behind a close one. Acceptable for a single-subject demo, wrong for a
  crowd.

## Latency

| where | per frame |
| --- | --- |
| onnxruntime-node, 1 thread, Ryzen 5 5600U | 273–311 ms (median 287 ms, n=15) — **measured on the superseded 3-class model**; the pose model is a similar size and parameter count but has not been re-timed |
| onnxruntime-node on Render free tier | not measured; inference is throttled to one frame per 500 ms regardless |
| onnxruntime-web, WASM, 1 thread | not measured |
| onnxruntime-web, WebGPU | not measured |

Only the first row is measured. The browser figures need a real pass on desktop
Chrome (WebGPU), desktop Firefox (WASM — no WebGPU in stable) and a mid-range
Android before being quoted anywhere.

## If accuracy needs to improve

In order of expected value per unit of effort:

1. **Grow the fixture set** (`frontend-new/scripts/eval-fixtures/`). At five
   images the harness can no longer tell the current system apart from a better
   one. Prioritise desk-webcam framing, which the tier-C limitation predicts is
   the weak spot, and multi-person frames.
2. **Tune the geometric boundaries against that set**, with `npm run posture` as
   the guard. The thresholds in `posture.js` are calibrated to wide measured
   gaps, not fitted — there is room, but no evidence yet on which direction.
3. **Solve tier C properly** if desk framing matters. Geometry alone cannot;
   it needs the transition into the pose (temporal) or scene context.
4. **Only then consider yolov8s-pose.** Roughly doubles inference cost, which
   the 512 MB host cannot absorb — though it matters much less now that the
   browser can carry the live path. Keypoint localisation is not currently the
   bottleneck, so expect little.

Fine-tuning the pose model is deliberately *not* on this list. It is COCO
weights doing a task COCO covers well; the value is in the geometry layer and
the fixtures, not in the backbone.

Separately, **int8 dynamic quantisation** is worth evaluating for the browser
build. fp32 weights are high-entropy and barely compress. Quantisation would cut
the model to roughly 3 MB. Now that the eval harness exists, the accuracy cost is
finally measurable — which was the blocker before.
