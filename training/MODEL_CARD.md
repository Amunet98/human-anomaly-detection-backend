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
| Model classes | 1 (`person`). The four postures are **not** model outputs |
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
three, because `posture.js` emits exactly those strings — four of them since
2026-08-12, when `squat` was added.

One trap that comes with that: `postprocess.js`'s `decodeYolov8`, kept only so
`eval-check.mjs` can still score the archived 3-class weights, indexes a class
list by the old model's head order. `squat` entered `CLASS_NAMES` at index 2,
which is where those weights emit `stand`. It therefore has its own frozen
`LEGACY_CLASS_NAMES` and must not be pointed back at `CLASS_NAMES`.

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
below `KP_CONF_THRESHOLD` (0.65) counts as missing - see the occlusion-tier
section below for why that number is load-bearing.

| feature | meaning |
| --- | --- |
| `torsoAngle` | shoulder-mid to hip-mid, degrees off vertical. 0 = upright, 90 = horizontal |
| `kneeDrop` | `(kneeY - hipY)` over torso length. Large = hips well above knees |
| `kneeAngle` | interior angle at the knee |
| `thighShinRatio` | projected thigh length over projected shin length |
| `hipAnkleDrop` | `(ankleY - hipY)` over torso length. **Signed** - negative means ankles above hips |
| `stanceOffset` | horizontal hip-to-ankle distance over torso length |
| `aspect` | box w/h. Since 2026-08-12 a **gate in its own right** - `>= 1.5` is a fall on its own |

Order:

1. `torsoAngle >= 50` (or `aspect >= 1.5` alone, covering a body lying toward
   the lens) is a **fall**
2. `kneeDrop <= -0.25` is a **fall** - the knees are above the hips, which is a
   body on its back or sprawled, never a seated posture
3. `thighShinRatio >= 2.5` is a **fall** - the shin points at the lens, i.e.
   kneeling or on all fours
4. `kneeAngle < 130` **and** `0.3 <= hipAnkleDrop < 1.0` **and**
   `stanceOffset < 0.5` is a **squat**
5. `thighShinRatio < 0.75` is a **sit**
6. `kneeDrop < 0.5` / `kneeAngle < 150` is a **sit**
7. else **stand**

Steps 3 and 4 were added 2026-08-12 and are calibrated against 3,106 measured
detections rather than the fixture set - see "Calibration at scale" below.

### Steps 1 and 2, added 2026-08-12 after a dojo video

Both were found the same way the thigh gate was - by replaying real footage
rather than reasoning about the thresholds. A 51-second aikido clip produced a
frame of a man **flat on his back** returning `sit` at 0.77.

**Why it escaped.** His torso read 25deg (under the 50 gate), his box aspect was
1.96, and the wide-box hatch then required `torsoAngle >= 30` - so it missed by
4.6deg. Falling through, `kneeDrop` was **-1.11**: knees a full torso-length
above the hips. A negative kneeDrop is trivially under `STAND_KNEE_DROP`, so the
sit branch claimed him.

**The asymmetry is the real lesson.** `SQUAT_HIP_ANKLE_DROP_MIN` was added
earlier for exactly this shape, on the stated reasoning that "a fall relabelled
`squat` is a missed alarm". That guard was applied to the squat gate and nowhere
else, so the leak did not close - it moved to `sit`. A guard on one branch of a
decision chain is not a guard on the property.

Measured over the corpus, among detections predicted `sit`:

| kneeDrop | ground truth fall : sit |
| --- | --- |
| below -1.00 | **36 : 0** |
| -1.00 to -0.50 | 109 : 2 |
| -0.50 to -0.25 | 109 : 7 |

254 recovered falls against 9 sits, ~28:1. The floor is -0.25 and not 0 because
`STAND_KNEE_DROP`'s own comment records a genuine bench-sit at -0.16, legs drawn
up; a floor at 0 would have reclassified that real sit as a fall.

**The wide-box condition was self-defeating.** `aspect >= 1.5` existed for bodies
foreshortened along the view axis - which have a *low* torso angle by
construction - and then required `torsoAngle >= 30`. The two halves worked
against each other. Measured: 27 corpus detections have a wide box and were not
called fall; of the 18 carrying a label, **18 are falls, zero sit, zero stand**,
in every torsoAngle band below 30. Removing the angle condition cost nothing
measurable. Residual risk, stated because absence of evidence is what it rests
on: a standing person with arms fully outstretched can exceed 1.5. None appears
in 9,331 detections, but the corpus is fall-heavy.

Both are pinned by `npm run posture`.

### Why squat sits where it does in the order

Ahead of the thigh gate, not after it. A deep crouch foreshortens the thigh too,
so `thighShinRatio < 0.75` claims 40% of crouches first if the order is reversed;
measured, moving squat ahead of it takes the class from 27 to 51 caught at
identical cost to the other three.

Behind the kneeling gate, not ahead of it. A kneel and a deep crouch share the
leg geometry, and the kneel is the more alarming reading - taking it first costs
nothing on the squat class (17 crouches read `fall` either way) and recovers 6
more falls.

`hipAnkleDrop` needs the ankle, so **squat is a tier-A-only class**. At tier B a
squat and a chair-sit are geometrically identical and the classifier answers
`sit`. That is deliberate, and it is the same honesty as the tier-C note below.

The **floor** on `hipAnkleDrop` is not symmetry for its own sake. The feature is
signed: negative means the ankles sit above the hips, which is a sprawled or
inverted body rather than a crouch. Without it the gate accepted them - found in
the wild on a corpus image returning `knee 128deg, hips -2.36 over ankles`. A
fall relabelled `squat` is a missed alarm, so 0.3 buys 42 fewer such leaks for
3 genuine crouches, and still sits below the squat class's own p10 of 0.41.

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
>
> **The UI now discloses this rather than hiding it.** `readout.js`'s
> `postureReadout` renders tier C as a grey `LEGS HIDDEN` and tier D as `NO READ`
> instead of a confident green `STAND n%`, and the live view coaches the viewer
> to step back once the top track holds an indeterminate tier for 1.5 s. The
> classifier is unchanged — it still emits `stand`, because the tracker needs a
> class to vote on and the tier discount already caps its weight. Only the
> presentation moved. Note this fires more often than it might appear it should:
> `balcony-stand.jpg`, an eval fixture whose expected label is `stand` and which
> passes, is itself tier C, so the demo hedges on it. That is correct — his knees
> are behind a railing — but it means a viewer will meet `LEGS HIDDEN` on images
> that look unambiguous to a human. Pinned by `npm run posture`.

## Measured accuracy — top-1 posture

From `frontend-new/scripts/eval-check.mjs` (`npm run eval:robust`), against the
labelled fixture set in `frontend-new/scripts/eval-fixtures/`. Measured
2026-08-12.

**Superseded 2026-08-12.** The fixture set grew from 8 to 13 and the numbers
below replace the previous 8/8 and 48/48. Both were real; neither meant what a
reader would assume, because the set could not yet fail.

| | clean | perturbed |
| --- | --- | --- |
| accuracy | **16/19 (84.2%)** | **98/114 (86.0%)** |
| macro-F1 | 0.838 | 0.859 |

Per-class under perturbation: `fall` P=1.000 R=0.750, `sit` P=1.000 R=1.000,
`stand` P=0.750 R=1.000, `squat` P=0.769 R=0.833.

**macro-F1 now spans all four classes**, where every previous figure in this
document spanned three. 0.859 is therefore *not* a regression from 0.905 - it is
a different average, over a set that finally includes the class that had no
coverage. Comparisons across that line are meaningless.

**Updated 2026-08-12** from 63/78 (80.8%) and macro-F1 0.841, after the inverted
and wide-box gates below. `sit` precision went 0.889 to 1.000 - it no longer
fires on anyone - and fall recall 0.643 to 0.714.

Three fixtures are labelled KNOWN GAP and expected to fail: two falls that 2D
geometry cannot express, and `squat-ceiling-gap.jpg`, which sits 0.05 the wrong
side of `SQUAT_HIP_ANKLE_DROP`. Between the 2026-08-12 gates and the squat
fixtures, nothing else flips under any perturbation - `court-fall-overhead.jpg`
used to flip to `sit` under three of them, so the kneeling gate had been holding
it by a thread on the clean image only. It no longer depends on that gate.

**`squat` gained single-subject fixtures on 2026-08-12** and is now scored like
any other class. Three deep bodyweight squats - front, back and profile - pass
top-1 at tier A, and a fourth is kept as a deliberate failure. The class went
from appearing in the confusion matrix only as a false positive to P=0.769
R=0.833 under perturbation.

The three passes are chosen to exercise the gate with the thigh both in and out
of the image plane: the front and back views foreshorten it (the back view
extremely so, thigh 0.17x its own shin) while the profile does not. The back
view is also what pins the *gate ordering* - its thigh ratio is far under
`SIT_THIGH_FORESHORTEN`, so reversing the squat and thigh gates turns it red.

The claim previously made here — that the corpus "cannot supply one: every image
in it is an accident scene" — was too strong, and was checked on 2026-08-12
rather than assumed. The corpus holds **168 images with a `squat` box, 11 of
which are annotated `squat`-only**. Running all 11 through the pipeline is what
settles it: none yields a clean single-subject squat. They are crowd scenes
whose other subjects went unannotated, and in every case either the top-1 goes
to a higher-confidence `sit` from an unlabelled bystander, or the pose model
finds a fallen person the 2023 labels missed. So the conclusion holds — clean
squats must come from outside this corpus — but for the more mundane reason that
the labels are incomplete, not because such a frame cannot exist.

**The class does now have eval coverage, via `expectedAll`.** Both croucher
fixtures carry a correct tier-A `squat` that top-1 could never assert, because
the fall correctly outranks it:

| fixture | asserted squat | features |
| --- | --- | --- |
| `rail-fall-with-crouchers.jpg` | soldier, conf 0.62 | knee 78deg, hips 0.81 over ankles |
| `street-fall-with-crouchers.jpg` | officer, conf 0.82 | knee 58deg, hips 0.56 over ankles |

That is a genuine regression guard — delete the squat gate and both fixtures go
red — but it is not a substitute for a single-subject fixture, because neither
exercises the top-1 path and both would still pass if `squat` were only ever
reachable behind a `fall`. The remaining unit-test coverage in
`posture-check.mjs` is unchanged.

**Vetting new fixtures.** `npm run fixture -- <image> --expect <class>` runs a
candidate through the real pipeline and the same six perturbations the harness
scores against, and refuses to stage anything that does not hold — including a
`squat` candidate that lands below tier A, which would be testing the sit
fallback rather than the squat gate. It exists because `lodge-group-a/b` were
once downscaled to 960px and silently stopped reproducing the bug they guard: a
fixture nobody verified is worse than no fixture, because the harness reports
green either way.

### Calibration at scale (2026-08-12)

The thresholds above were originally justified by five to eight measured values.
They have now been replayed over **3,106 detections** from 4,924 deduplicated
images of the 2023 project's Roboflow corpus, staged by
`frontend-new/scripts/calibration/build-corpus.py` into an unversioned
`corpus-2023/` beside the repos, measured by
`frontend-new/scripts/feature-dump.mjs`, and analysed by the two scripts beside
the builder. The corpus itself is ~250 MB and deliberately not committed; the
scripts that regenerate it are.

The existing gates hold. Fraction of each class falling below each threshold:

| gate | stand | fall | sit | squat |
| --- | --- | --- | --- | --- |
| `kneeDrop < 0.5` | **3.7%** | 78.3% | 67.5% | 64.6% |
| `kneeAngle < 150` | **7.6%** | 66.5% | 69.5% | 71.3% |
| `thighShinRatio < 0.75` | **2.3%** | 34.0% | 33.9% | 40.7% |

That is the result worth keeping: three thresholds calibrated on a handful of
images survive 400x more data essentially unchanged.

But it also shows what they do **not** do. Those gates separate `stand` from
not-`stand`. They do not separate fall from sit from squat - on
`thighShinRatio` the three medians are 0.89 / 0.87 / 0.90, and on `kneeAngle`
129 / 116 / 95. That is the mechanism behind the recall figure below.

Medians used to place the squat gate:

| | kneeAngle | hipAnkleDrop | stanceOffset |
| --- | --- | --- | --- |
| squat | 95 | 0.75 | 0.21 |
| sit | 116 | 1.15 | 0.50 |
| stand | 175 | 1.47 | 0.08 |

Note `stand` also has a low `stanceOffset` - feet under hips is true of standing
too - so that feature cannot gate alone. `kneeAngle` and `hipAnkleDrop` are what
exclude standing.

The **disagreement fallback was re-tested and kept**. When `kneeDrop` and
`kneeAngle` conflict (527 of 3,106 detections) the code defers to `kneeDrop`;
deferring to `kneeAngle` instead trades 25 correct `stand` calls for 3 correct
`sit` calls with no change to fall recall. It stays as it was.

### Fall recall against the 2023 corpus

Of 1,970 detections overlapping a 2023 `fall` box, the classifier called 731 a
fall after the kneeling gate (706 before it). **Do not read that as 36%
accuracy.** The corpus labels are box-level annotations made for a detector, its
`fall` class marks "person is down" including seated-on-the-ground, and its
`squat` class is crouching bystanders. The structure of the misses is the
trustworthy part:

| share of misses | bucket |
| --- | --- |
| 38% | torso upright, knees high - reads as seated |
| 16% | tier D, no torso vector |
| 16% | torso 30-50deg, below the fall gate |
| 10% | tier C, no knees |
| 10% | torso upright, legs extended - reads as standing |
| 5% | kneeDrop/kneeAngle disagree |
| 2% | shin foreshortened - now caught by the kneeling gate |

The dominant cause is **camera elevation**. When the camera looks down, a person
lying on the floor projects to nearly the same 2D skeleton as an upright person
seen from the front, because `torsoAngle` measures rotation *in* the image plane
and cannot see rotation *toward* it. The previous fixture set was almost entirely
eye-level and side-on, which is why none of this was visible. It matters because
CCTV is mounted high.

**Read this with the caveat it deserves.** Thirteen images through six
perturbations is 78 trials, not 78 independent samples. The set is still far too
small to quote anywhere load-bearing, and it is now deliberately unbalanced
toward hard cases: five of the fifteen came from the 2023 corpus specifically
because they were failure candidates. An 86.7% measured on a set selected that
way is not comparable to 86.7% on a representative sample, and reading it as a
deployment accuracy would be wrong in both directions.

What it is good for is regression detection, which the previous set had stopped
providing. At 8/8 and 48/48 the harness could not tell the current system from a
better or a worse one; it can now.

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
before quoting a figure anywhere load-bearing. The priorities now, in order:

1. **More `squat` variety.** The class is no longer uncovered, but all four of
   its fixtures are the same subject in the same gym from one shoot, so they
   measure viewpoint rather than population. A second subject, a different
   body type and a non-gym setting would say much more than a fifth angle of
   this one. Vet each with `npm run fixture` before adopting it.
2. **`sit`**, which is data-poor everywhere: the entire 2023 corpus holds 120
   sit boxes against 4,765 fall boxes.
3. **High-angle and overhead framing** for every class, which the recall
   analysis above identifies as the systematic weak spot and which the original
   eight fixtures did not contain a single example of.

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
  **This was confirmed, not merely predicted** (2026-08-12): fixture
  `prone-fall-view-axis.jpg` is a man face-down on a floor returning `stand` at
  0.70, with `torsoAngle` 2.5deg, `kneeDrop` 1.14, `kneeAngle` 168deg and
  `thighShinRatio` 1.29 — every feature reading as a standing person, and a box
  aspect of 0.45 that misses the escape hatch. It is kept as a deliberately
  failing fixture. Nothing in `posture.js` fixes it: two-dimensional keypoints
  do not carry rotation toward the image plane, which puts it in the same
  category as tier C. Resolving it needs a signal geometry does not have —
  the transition into the pose, or a scene/ground reference.
- **Camera elevation is the systematic version of the above**, and against the
  2023 corpus it is the single largest source of missed falls. A high-mounted
  camera projects a person on the floor into nearly the same skeleton as an
  upright person seen from the front. CCTV is mounted high, so this is a
  deployment-domain problem rather than a corner case.
- **The fallen subject is sometimes not detected at all.** Fixture
  `bus-fall-obscured.jpg` has three people crouched over a person on the ground;
  the pose model returns the crouchers and the bystanders and never the subject.
  That is a detection miss upstream of posture, and no geometry change reaches
  it. Kept as a second failing fixture precisely to keep it distinguishable from
  the view-axis case.
- **The kneeling gate is no longer what holds `court-fall-overhead.jpg`.** It
  used to recover that fixture on the clean image only, with the fixture still
  flipping to `sit` under hflip, grayscale and downscale-320. Since the inverted
  gate (`kneeDrop <= -0.25`) it survives all six perturbations - she is on her
  back with her knees drawn up, which is that gate's exact shape. The kneeling
  gate remains a genuine improvement but is no longer load-bearing here, and its
  own marginality is now untested by any fixture.
- **`torsoAngle` is unsigned with respect to inversion, and nothing else covers
  it.** The feature is the angle between shoulder-mid and hip-mid off vertical -
  it says how far from upright the torso is, never *which end is up*. A person
  standing on their head measures the same 0deg as a person standing on their
  feet. Found 2026-08-12 in the dojo clip: a man mid-air in a breakfall, head
  down and legs overhead, returned `sit` at **0.84** (torsoAngle 23deg,
  kneeDrop -0.09, kneeAngle 85deg, aspect 0.69). Every gate misses it - the
  torso reads upright, the box is narrow so the wide-box hatch does not fire,
  and kneeDrop -0.09 is well inside the -0.25 inverted floor. Not fixed, and
  deliberately not fixed by lowering that floor: -0.09 is ordinary for a genuine
  seated posture (the measured bench-sit is -0.16), so a floor that catches this
  would reclassify real sits as falls. The tractable fix is a *signed* torso
  feature - compare shoulder-mid y against hip-mid y directly - which is a new
  feature rather than a threshold change, and would need its own corpus pass.
  In the meantime the tracker's 1.2s sustain is what stops a single such frame
  mattering, and a real fall passes through this pose only briefly.
- **`squat` is tier-A only**, so a waist-up crouch returns `sit` by
  construction - the gate needs ankles and cannot run without them.
- **The squat ceiling misses genuine deep squats, and is kept anyway.**
  `SQUAT_HIP_ANKLE_DROP = 1.0` is an upper bound, and a real three-quarter-view
  squat measuring 1.05 falls through it to `sit`
  (`squat-ceiling-gap.jpg`). Raising it is measurably worse: 1.0 -> 1.05 newly
  claims 27 corpus detections of which only 3 are squats and **7 are falls**.
  Trading 7 missed alarms for 3 correct crouches is the wrong direction for an
  anomaly detector, so the miss stays. The fixture flips on a 0.05 margin -
  failing clean, grayscale, dark and downscale but passing hflip, blur and crop -
  which is what a threshold boundary honestly looks like.
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

1. **Grow the fixture set** (`frontend-new/scripts/eval-fixtures/`). Thirteen
   images is still far too few, and the set has no `squat` at all. Priorities
   are listed at the end of the accuracy section: a clean squat from outside the
   2023 corpus, more `sit`, and high-angle framing for everything.
2. **Attack camera elevation**, now identified as the largest systematic source
   of missed falls. Geometry alone cannot solve it — `torsoAngle` measures
   rotation in the image plane and a fall toward the lens is rotation out of it.
   The tractable routes are temporal (the *transition* into the pose is visible
   even when the pose is not) or a homography/ground-plane reference if the
   camera is fixed, which for CCTV it is.
3. **Tune the geometric boundaries**, with `npm run posture` as the guard. This
   moved down the list: the boundaries have now been replayed over 3,106
   detections and the existing ones hold. The evidence says the gates are not
   where the remaining error is.
4. **Solve tier C properly** if desk framing matters. Geometry alone cannot;
   it needs the transition into the pose (temporal) or scene context.
5. **Only then consider yolov8s-pose.** Roughly doubles inference cost, which
   the 512 MB host cannot absorb — though it matters much less now that the
   browser can carry the live path. Keypoint localisation is *mostly* not the
   bottleneck — though `bus-fall-obscured.jpg` is a case where it is, the
   fallen subject going undetected entirely, so a larger backbone is no longer
   a pure no-op here.

Fine-tuning the pose model is deliberately *not* on this list. It is COCO
weights doing a task COCO covers well; the value is in the geometry layer and
the fixtures, not in the backbone.

Separately, **int8 dynamic quantisation** is worth evaluating for the browser
build. fp32 weights are high-entropy and barely compress. Quantisation would cut
the model to roughly 3 MB. Now that the eval harness exists, the accuracy cost is
finally measurable — which was the blocker before.
