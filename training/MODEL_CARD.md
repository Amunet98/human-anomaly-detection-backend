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

`squat`'s R=0.833 is the most misleading figure in this table: measured against
the corpus its true recall is **44.3%** and its precision on labelled rows is
**35.1%**. See "`squat` is low-recall AND low-precision" below.

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

> **Do not read that R=0.833 as the class's recall.** All four fixtures are one
> person doing textbook gym squats in even light, so they measure the easy
> centre of the distribution rather than the class. **Measured against the
> corpus it predicts `squat` for 47 of 106 tier-A ground-truth squats - 44.3%,
> at 35.1% precision.** The fixture figure is flattering by roughly 39 points and
> exists to catch regressions, not to describe field behaviour.

### `squat` is low-recall AND low-precision

**Corrected 2026-08-12, same day, after re-running `feature-dump.mjs` end to
end.** The first version of this section said the class was *precise* and that
widening its gate cost fall recall at roughly 9:1. Both halves were wrong, and
the reasoning error is worth preserving because it is easy to repeat.

Measured end to end over the corpus, on the 106 tier-A ground-truth squats:

| | |
| --- | --- |
| satisfy the squat conditions | 51/106 = 48.1% |
| **actually predicted `squat`** | **47/106 = 44.3%** |
| satisfied but claimed earlier | 4, all by fall gates |

So true recall is 44.3%, not the 48.1% that reading the thresholds alone
suggests - the fall gates take 4 of them first, correctly.

**Precision is the worse number.** Of 327 detections predicted `squat`, 134
carry a ground-truth label: 47 are squats, **78 are falls**, 7 stand, 2 sit.
That is **35.1% precision on labelled rows** - the class is wrong more often
than it is right, and its dominant error is calling a fall a crouch. (Read with
the caveat that 193 of the 327 are unlabelled, and the 2023 annotators labelled
accident participants rather than every person, so the labelled subset is
biased toward the dramatic.)

**Why "9 falls per squat" was wrong.** The squat gate is **step 4** - it runs
after all three fall gates. A detection that reaches it is one the fall gates
already declined, so widening it *cannot* take a correctly-detected fall.
Verified rather than assumed: 77 detections currently predicted `fall` satisfy
the current squat conditions and stay `fall`, because ordering wins. The earlier
sweep counted "ground-truth falls whose geometry satisfies the squat
conditions" and silently treated them as losable. They were not losable; most
were already being missed as `sit`.

The real cost of widening, counting only rows the fall gates did not claim:

| widened to | rows moved | genuinely squat | already-missed falls (sit->squat) | real sit/stand broken |
| --- | --- | --- | --- | --- |
| 130 / 1.2 / 0.5 | 48 | 4 | 4 | 7 |
| 150 / 1.2 / 0.5 | 113 | 6 | 19 | 14 |
| 150 / 1.5 / 0.7 | 322 | 12 | 73 | 38 |

**The recommendation survives the correction, for a different reason.** Widening
costs roughly three correct `sit`/`stand` calls per genuine squat gained, and
buys nothing in alarm behaviour - the falls it absorbs were already missed, so
they merely change which wrong label they carry. `SQUAT_HIP_ANKLE_DROP` stays at
1.0.

The cause is the one already recorded under "Calibration at scale": these
features separate `stand` from not-`stand` and do not separate fall from sit
from squat. Medians for fall / sit / squat are 0.89 / 0.87 / 0.90 on
`thighShinRatio` and 129 / 116 / 95 on `kneeAngle`. **A person crouching and a
person down on the ground are not distinguishable here.**

The honest one-line description is therefore: *`squat` catches under half the
crouches and is wrong most of the times it fires, but it costs nothing in fall
recall, because it only ever relabels detections the fall gates already
declined.* Treat a `squat` output as "not standing, probably low" rather than as
a reliable class.

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

### Per-class, end to end, before and after 2026-08-12

The question anyone reviewing a new class asks first is whether it broke the
existing ones. Measured over all 9,331 corpus detections, re-dumped end to end
rather than reasoned about:

| class | | P | R | F1 | tp / fp / fn |
| --- | --- | --- | --- | --- | --- |
| **fall** | before | 0.943 | 0.358 | 0.519 | 705 / 43 / 1265 |
| | **after** | 0.941 | **0.495** | **0.649** | 976 / 61 / 994 |
| **sit** | before | 0.056 | 0.562 | 0.102 | 50 / 845 / 39 |
| | **after** | 0.080 | 0.438 | **0.135** | 39 / 449 / 50 |
| **stand** | before | 0.561 | 0.936 | 0.702 | 821 / 642 / 56 |
| | **after** | 0.567 | 0.936 | **0.707** | 821 / 626 / 56 |
| **squat** | before | - | - | 0.000 | 0 / 0 / 170 |
| | **after** | 0.351 | 0.276 | **0.309** | 47 / 87 / 123 |

**Nothing regressed.** Exactly **2 detections** out of 9,331 that were correct
before became `squat`, both of them `sit`. Every class's F1 is flat or better.

Reading each row:

- **`fall`** gained 13.7 points of recall at a precision cost of 0.002. This is
  the whole point of the day's work.
- **`sit`** lost recall, and that is an improvement. Its precision was **0.056** -
  it fired 895 times on labelled rows and was right 50 times, a garbage-collector
  class in the same shape as the old detector's `sit` attractor documented at the
  top of this card. Shedding 305 predictions to `squat` and 382 to `fall` raised
  both its precision and its F1.
- **`stand`** is untouched: identical 821 true positives, F1 +0.005.
- **`squat`** went from nothing to F1 0.309.

Two caveats that matter for reading this table:

1. **"Before" predates all four of the 2026-08-12 gates** - kneeling, squat,
   inverted and wide-box - so this is the whole day's work, not the squat gate
   in isolation.
2. **`sit` precision is depressed by the corpus's own convention.** Its `fall`
   class marks "person is down" including seated-on-the-ground, so a correct
   `sit` on someone sitting on the floor is scored as a `sit` false positive.
   The before/after *comparison* is still valid; the absolute value is not.

`squat`'s R=0.276 here is over all 170 ground-truth squats. The 44.3% quoted
earlier is over the 106 that are tier A. The gate cannot run without ankles, so
tier A is the fair denominator for judging the gate and 27.6% is the fair one
for judging what a user actually experiences.

### Fall recall against the 2023 corpus

Of 1,970 detections overlapping a 2023 `fall` box, the classifier now calls
**976 a fall - 49.5%**, re-measured end to end on 2026-08-12 after the inverted
and wide-box gates. The same dump before those gates (and before the kneeling
and squat gates) called 705, so recall went **35.8% -> 49.5%, +271 detections,
with zero regressions**: not one detection that was already correctly called
`fall` moved to anything else. The 271 came from `sit` (266) and `stand` (5) -
that is, from exactly the two labels this section identifies as absorbing missed
falls.

A related check, since `squat` was added the same day and absorbs 78
ground-truth falls: every one of those 78 was already wrong before, predicted
`sit` (68) or `stand` (10). The squat gate moved them between two non-alarming
labels and cost no alarms.

**Do not read 49.5% as accuracy.** The corpus labels are box-level annotations made for a detector, its
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

1. **More `squat` variety - but do not expect it to move recall.** All four
   fixtures are the same subject in the same gym from one shoot, so they measure
   viewpoint rather than population; a second subject, a different body type and
   a non-gym setting would say much more than a fifth angle of this one. What
   they will do is make the fixture figure *honest* rather than raise it. The
   44.3% corpus recall is a property of the feature space, not of the fixture
   set, and no amount of fixtures changes it - see "`squat` is precise and
   low-recall on purpose". Vet each with `npm run fixture` before adopting it.
2. **`sit`**, which is data-poor everywhere: the entire 2023 corpus holds 120
   sit boxes against 4,765 fall boxes.
3. **High-angle and overhead framing** for every class, which the recall
   analysis above identifies as the systematic weak spot and which the original
   eight fixtures did not contain a single example of.

What this measurement does **not** cover, unchanged from before: box
localisation quality, multi-person scenes scored per-person rather than top-1,
and any video/temporal behaviour (that is `npm run tracker`'s job).

### False-alarm rate, measured for the first time (2026-08-12)

Every accuracy figure in this document until now was measured on people who had
fallen. The 2023 corpus is entirely accident scenes, so there was no population
of confirmed *non*-fallen people to count false alarms against - the one number
that decides whether an anomaly detector is usable in a room where nothing is
wrong.

The POLAR posture dataset supplies it. Staged via
`build-corpus.py --dataset polar` into `corpus-polar/`, it holds 5,784 matched
people across `sit` / `squat` / `stand` and **contains no falls at all**, so
every `fall` this system emits on it is wrong by construction.

| | |
| --- | --- |
| people | 5,784 |
| called `fall` | **666 = 11.5%** |
| above `FALL_ENTER_CONF` 0.55, which the tracker acts on | **470 = 8.1%** |

By true posture: **434 squatting**, 224 sitting, 8 standing. By tier: 601 tier A,
65 tier B.

**Crouching is two-thirds of it**, which is the predicted result rather than a
surprise: `KNEEL_SHIN_FORESHORTEN` deliberately emits `fall` for kneeling, and
the note under "Known failure modes" already called that the acknowledged
weakest link. It now has a number instead of an acknowledgement.

**Read it as a stills figure, not a deployment figure.** `tracker.js` requires
`FALL_CONFIRM_MS` 1,200 ms of sustained fall before confirming, so a crouch that
flickers to `fall` for one frame raises nothing. What 8.1% bounds is how often a
single frame of an ordinary person is actionable - the rate the tracker has to
suppress, not the rate a user sees. Measuring the post-tracker rate needs video
of people not falling, which this project still does not have.

**A second result from the same run, worth as much as the first.** POLAR's
tier-A `squat` agrees with this classifier **44.3%** of the time - the same
44.3% measured independently against the 2023 corpus. Two unrelated datasets,
labelled by different people for different purposes, producing the same recall
is strong evidence that the number is a property of the classifier rather than
of either label set. It also clears POLAR's labels: `stand` agrees **98.7%**, so
the two vocabularies do match and the disagreement on `squat` is this system's,
not POLAR's.

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
- **Crouching people read as `fall` 434 times in 5,784.** Measured on POLAR,
  which contains no falls: 11.5% of ordinary people are called a fall on a
  single frame, 8.1% at a confidence the tracker would act on, and two-thirds of
  those are squatting. This is the deliberate `KNEEL_SHIN_FORESHORTEN` trade
  quantified - see "False-alarm rate" above. The 1.2s sustain in `tracker.js` is
  the only thing standing between that rate and a user-visible false alarm,
  which makes it load-bearing rather than cosmetic.
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
| onnxruntime-web, WASM, 1 thread, mid-range Android (Chrome) | **919-1026 ms, 1-2 FPS** — measured 2026-08-12 from the demo's own on-screen readout, n=2 observations, not a benchmark |
| onnxruntime-web, WebGPU | not measured |

The Android row comes from the demo's own overlay during the smoke test below,
not from a timing harness - two glances at a live readout. It is enough to
establish the order of magnitude (~1 s per frame, roughly 3x the desktop
figure) and not enough to quote as a benchmark.

**It reports `wasm`, not `webgpu`, and that is correct behaviour.** Diagnosed via
`chrome://gpu` on the device (Nothing Phone 3, Adreno, Chrome 151, Android 16).
The first reading was wrong and is worth recording as such: `Vulkan: Disabled`
looked like the cause, but enabling `#enable-vulkan` left WebGPU disabled, which
disproved it.

The actual entry under "Problems Detected" is:

    Disable webgpu on vk via gl interop: crbug 442791440, 475935650
    Disabled Features: webgpu_on_vk_via_gl_interop

Chrome's compositor on this device runs on **GL** - `use_virtualized_gl_contexts`
appears in the applied workarounds - while WebGPU renders through **Vulkan**. The
hand-off between them is the "vk via gl interop" path, and Chrome blocklists it
on this hardware. Vulkan was never the blocker; the bridge out of it is. Nothing
in `session.js` is involved and nothing needs fixing.

The consequence is worth stating plainly: **WASM is the mobile deployment
reality, not a degraded path.** Visitors will not toggle `chrome://flags`, so
WebGPU is unavailable to them on Android Chrome regardless of what the JSEP build
supports. Any mobile optimisation has to target WASM, or accept 1-2 FPS - which
for a detector whose `FALL_CONFIRM_MS` is 1,200 ms is not obviously a problem.

The WebGPU row therefore stays unmeasured on mobile for a structural reason
rather than an untested one. Desktop Chrome remains the place to measure it.

The remaining figures need a real pass on desktop Chrome (WebGPU) and desktop
Firefox (WASM - no WebGPU in stable) before being quoted anywhere.

## Device smoke test (2026-08-12)

The first time any of this was run against a real camera rather than a still
image. Recorded because "untested on a device" and "smoke-tested once" are
different states, and the difference is annoying to reconstruct later.

**Setup:** the live demo in a phone browser, subject standing back far enough to
read the screen in a mirror. **Result:** the owner reports it working
"far better than yesterday", i.e. than the same demo before the inverted and
wide-box gates, and separately confirmed the tier-C `LEGS HIDDEN` readout
appears at close range.

**This is a smoke test, not QA.** One device, one browser, one person, one
lighting condition, no matrix, and no recorded output to re-examine. It is
evidence the system is not broken on real hardware. It is not evidence of an
accuracy figure, and nothing here should be quoted as one.

**What that setup does and does not exercise.** Standing back to see a mirror
puts the whole body in frame, which is **tier A** - precisely where the day's
changes land, since the inverted gate, the wide-box gate and the recovered falls
all need the full leg chain or a horizontal torso. So it tests the case that
improved most and leaves the two that did not:

- **Tier C** was confirmed in the same session: the close-range waist-up framing
  a phone at arm's length produces does show the grey `LEGS HIDDEN` readout
  rather than a confident green `STAND`. That is the honesty fix working on real
  hardware, in the framing most visitors will actually hit. Still one device and
  one observer, so it carries the same caveat as everything else here.
- **A full-frame crouch reads `squat`, not `fall`** - confirmed on the device at
  52% confidence, browser engine, tracked as `#3`. This was the prediction most
  likely to be wrong: crouching is two-thirds of the 11.5% false-alarm rate, so
  a `fall` was the expected reading and the tracker's sustain was expected to be
  what suppressed it. It never had to.

  Read it as the good case working, not as a revision of the 11.5%. The subject
  was in the squat gate's sweet spot - torso upright, hips low but above the
  ankles, feet under the centre of mass, so `hipAnkleDrop` sat comfortably inside
  the 0.3-1.0 window. The POLAR crouches that read `fall` are the ones that lean
  forward or clear that ceiling. One clean squat passing moves nothing.

- **The browser engine runs real-time on a phone**, which the latency table above
  still lists as "not measured". Not a number - no frame timing was recorded -
  but onnxruntime-web is evidently viable on mid-range Android rather than
  merely theoretically supported.

- Worth noting for its own sake: the pose model tracked shoulders, hips, knees
  and ankles correctly **through a mirror, in a dim bedroom, at an angle**. No
  fixture in the set covers any of those conditions.

- **A second, wider crouch also read `squat`**, at 69%. Still not the provocation
  the false-alarm path needs: photographing yourself keeps the torso upright,
  because the phone has to stay pointed at the mirror, so `torsoAngle` never
  approaches the 50deg gate. The case that matters is reaching toward the floor
  with the torso near-horizontal, which is close to unphotographable one-handed.
  A second person or a timer is what that test needs.

- **Tier C and tier A were observed back to back on the same posture** - one
  frame with the legs in shot reading `SQUAT 69%`, the next with the frame cut at
  the knees reading `LEGS HIDDEN` with no percentage. That is the tier system's
  whole intent visible in two screenshots: not a degraded answer, a different
  one.

- **The false-alarm path**, which remains the open one. Whether FALL CONFIRMED
  appears on a genuine forward-leaning crouch, or whether `FALL_CONFIRM_MS` eats
  it, is still unestablished - and remains the only claim in this document
  resting on the tracker rather than on measurement. A crouch in full frame should read `fall` - that is
  the 11.5% measured under "False-alarm rate" - and whether `tracker.js`'s 1.2s
  sustain actually suppresses it before the FALL CONFIRMED badge appears is the
  single most useful thing a next test could establish. It is the only claim in
  this document resting on the tracker rather than on measurement.

Still open: the perturbation matrix across browsers, the latency table above, and
the post-tracker false-alarm rate.

## A trained keypoint classifier was tried and is NOT shipped (2026-08-12)

`training/train_posture_keypoints.py` trains a classifier on the 17 joint
positions instead of deciding posture with hand-tuned gates. It is the obvious
upgrade, it works in-distribution, and it is not deployed. This section exists so
the next person to have that idea finds the measurement rather than repeating it.

**Setup.** 8,781 labelled detections - `corpus-2023` (falls) plus `corpus-polar`
(sit/squat/stand), minus the 108 disputed fall labels. Input is each joint
normalised to the person's own box plus the box aspect, so image position and
scale are not learnable. Split by *image*, so multi-person frames never span
train and test. The geometric classifier's answer travels in each row as `pred`,
so the baseline is measured on the identical split.

**In-distribution it wins clearly**, on tier A where the gates can compete:

| | accuracy | false alarms on POLAR |
| --- | --- | --- |
| `posture.js` gates | 0.693 | 0.181 |
| **GradientBoosting** | **0.830** | **0.043** |
| MLP (128,64) | 0.816 | 0.051 |

Per class the gains land where the gates are weakest: `squat` recall 0.281 ->
0.824, `sit` 0.422 -> 0.667, `fall` 0.504 -> 0.762. Better accuracy *and* four
times fewer false alarms, which is the combination that matters.

**It is not riding the occlusion shortcut.** The two sources differ in visibility
(corpus-2023 is 76% tier A, POLAR 59%), and confidences alone reach 0.621 against
0.357 chance - so the leak is real and available. What clears the model is that
coordinates alone score 0.830, *identical* to the full feature set. The ablation
is retained in the script: if a future change makes "everything" beat
"coordinates only" by more than noise, that gap is the leak being exploited.

**Then it loses on the held-out fixtures**, which come from neither corpus:

| | fixtures |
| --- | --- |
| `posture.js` gates | **16/19** |
| trained model | 15/19 |

One fixture of difference on n=19 is statistically nothing. The *kind* of
disagreement is not:

- `street-fall-with-crouchers.jpg` - **a missed fall.** The woman on the ground
  reads `fall` 0.65 from the gates (torso 89deg) and `squat` 0.72 from the model.
  With no `fall` in the frame the top-1 reduction then reports a bystander's
  `stand`.
- `lodge-group-b.jpg` - **a confident false alarm.** A seated woman reads `sit`
  0.45 from the gates and `fall` **0.80** from the model. It gets `lodge-group-a`
  right, so it flips across the near-identical pair those two fixtures exist to
  catch.
- `squat-ceiling-gap.jpg` - the model wins, correctly returning `squat` where the
  gates hit the `SQUAT_HIP_ANKLE_DROP` ceiling. Learning the boundary does beat
  hand-placing it.

**The diagnosis is domain shift.** Training data is accident scenes plus
gym/stock photography. The lodge fixtures are domestic interiors with a sofa - a
domain neither source covers. **Geometry is domain-invariant by construction**: a
shoulder-to-hip angle means the same thing in any room, whereas a learned model
inherits its training distribution. That is an uncomfortable echo of the 2023
detector documented at the top of this card - not the same failure, since this
input contains no background at all, but the same shape: strong in-distribution,
unreliable outside it.

**What would change the verdict:** training data from more domains, domestic
interiors first; and a fixture set large enough that 15-vs-16 carries
information. The approach is sound - the squat result proves it - but it is not
yet better than what ships.

The ONNX export is gitignored and nothing is wired into `inference.js`.

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
5. **Not a trained keypoint classifier - that was tried on 2026-08-12 and lost
   on the held-out fixtures.** It beats the gates in-distribution (0.830 vs
   0.693, four times fewer false alarms) and loses out of it, with a missed fall
   and a confident false alarm on a seated woman. See "A trained keypoint
   classifier was tried and is NOT shipped". Revisit it with training data from
   more domains, not with better hyperparameters.
6. **Only then consider yolov8s-pose.** Roughly doubles inference cost, which
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
