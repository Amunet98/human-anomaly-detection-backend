// Hand-mirrored port of frontend-new/src/lib/detect/posture.js.
//
// The two repos share no package, so this file and its browser twin are kept in
// step by hand and by frontend-new's `npm run parity`, which runs both against
// the same ONNX session and asserts they agree. Structured line-for-line with
// the browser version - keep it diff-able. Only the module system and the
// KP_CONF_THRESHOLD import differ.

// Decides fall / sit / stand from COCO keypoint geometry.
//
// This replaces a learned 3-class detector head. The reason is measured: the old
// yolov8n classifier scored 100% on clean fixtures but 76.7% once they were
// blurred, greyed, darkened, downscaled or cropped, and `sit` came out at
// precision 0.545 / recall 1.000 - it never missed a sit and labelled nearly
// half its non-sits as one. It had learned scene appearance (bench, floor,
// indoor room) rather than body configuration, so any degradation pushed it
// toward its most common training class.
//
// Geometry does not have that failure mode. The angle between a subject's
// shoulders and hips is the same whether the frame is sharp or blurred, colour
// or greyscale, 1408px or 320px wide. What varies is whether the *keypoints* can
// be found, and that job is done by a COCO-pretrained pose model trained on
// ~200k person instances with heavy augmentation - a far better-conditioned
// problem than 3-way posture classification from 8,340 images.
//
// Thresholds below are calibrated against measured fixture values, quoted inline
// so a future reader can see the margin rather than guess at it. They are not
// fitted to a large sample - see MODEL_CARD.md on the fixture set's size.

// Mirrors KP_CONF_THRESHOLD in the browser's constants.js.
const KP_CONF_THRESHOLD = 0.65;
const { SQUAT_PROBE_THRESHOLD, probeFeatures, squatProbability } = require('./squat-probe.js');

// --- tuning ---------------------------------------------------------------

// Torso angle off vertical, in degrees, above which the subject reads as
// horizontal. Measured: upright fixtures sit at 1.6/1.7/14.0 deg, the two falls
// at 57.6 and 72.3. 50 leaves ~36 deg of margin below and ~8 deg above, and is
// deliberately nearer the falls: a person bending to pick something up passes
// through 40-60 deg, and the tracker's FALL_CONFIRM_MS=1200 sustain requirement -
// not this threshold - is what stops that registering as a fall.
const FALL_TORSO_ANGLE = 50;

// A person lying directly toward or away from the camera has a foreshortened
// torso, so the angle understates how horizontal they are. A wide box covers
// that case. Measured: the street fall's box aspect is 2.28, every upright
// fixture is between 0.30 and 0.84.
//
// This used to additionally require torsoAngle >= 30, which made the escape
// hatch unreachable in exactly the geometry it exists for - a body foreshortened
// along the view axis has a LOW torso angle by construction, so the two
// conditions worked against each other. Measured over the corpus (2026-08-12):
// 27 detections have aspect >= 1.5 and were not called fall; of the 18 that
// carry a ground-truth label, 18 are falls and none is a sit or a stand, in
// every torsoAngle band below 30. The angle condition cost recall and bought
// nothing measurable, so it is gone.
//
// Residual risk, stated because the corpus cannot rule it out: a standing person
// with arms fully outstretched can exceed 1.5 while upright. None appears in
// 9,331 detections, but the corpus is fall-heavy, so this is an argument from
// absence. The tracker's 1.2s sustain is what stops a single such frame alarming.
const FALL_ASPECT = 1.5;

// Below FALL_TORSO_ANGLE but above this, the torso is neither clearly upright
// nor clearly horizontal (bending, crouching, reaching down). We do not flip the
// class here - we keep the sit/stand answer and discount its confidence, letting
// the tracker's confidence-weighted vote settle it over its 7-frame window.
const AMBIGUOUS_TORSO_ANGLE = 30;

// Vertical hip-to-knee separation, normalised by torso length. Measured: seated
// on a bench -0.16 (knees above hips, legs drawn up), standing 0.74 to 0.83.
const STAND_KNEE_DROP = 0.5;

// Knees this far ABOVE the hips is not a seated posture - it is a body on its
// back, sprawled, or inverted. Without this the sit branch swallows them: a
// negative kneeDrop is trivially < STAND_KNEE_DROP, so `sit` was the answer for
// anyone lying down whose torso angle stayed under the fall gate.
//
// This is the same guard the squat gate already had. SQUAT_HIP_ANKLE_DROP_MIN
// was added because inverted bodies were leaking into `squat`, on the reasoning
// that "a fall relabelled `squat` is a missed alarm" - but the sit branch below
// never got the equivalent, so the leak simply moved from `squat` to `sit`.
//
// Placed at -0.25 rather than 0 deliberately: STAND_KNEE_DROP's own comment
// records a genuine bench-sit measured at -0.16, legs drawn up. A floor at 0
// would reclassify that real sit as a fall. -0.25 sits below it with margin.
//
// Measured over the corpus (2026-08-12), among detections predicted `sit`:
//   kneeDrop < -1.00   36 fall :  0 sit
//   -1.00 to -0.50    109 fall :  2 sit
//   -0.50 to -0.25    109 fall :  7 sit
// 254 recovered falls against 9 sits turned into false alarms, ~28:1. Read the
// `fall` label with the caveat MODEL_CARD gives it - the 2023 class marks
// "person is down", which includes seated-on-the-ground - but a person on the
// ground with knees above their hips is sprawled, and "down" is the alarm an
// anomaly detector should raise either way.
const INVERTED_KNEE_DROP = -0.25;

// Interior angle at the knee. Measured: seated 87 deg, standing 175-179 deg.
// 150 sits in the wide empty gap between them.
const STAND_KNEE_ANGLE = 150;

// Projected thigh length over projected shin length.
//
// The femur and tibia are within ~10% of each other in real length, so when
// both lie in the image plane - which is what standing with vertical legs
// means - their projected ratio is ~1. A thigh much shorter than its own shin
// can only mean the thigh points toward or away from the lens.
//
// This catches the case both other features are blind to: someone sitting
// facing the camera with their legs stretched out forward. Their knees sit
// well below their hips (kneeDrop 0.64, reads as standing) and their leg is
// nearly straight (kneeAngle 172deg, reads as standing), but the thigh is
// foreshortened to 0.58 of its shin, which no standing pose produces.
//
// The two mechanisms are complementary rather than redundant: a side-on sit
// has the thigh in-plane and horizontal, so kneeDrop and kneeAngle catch it
// while the ratio stays ~1; a front-on sit foreshortens the thigh, so only the
// ratio catches it. Measured - standing 1.00 / 1.01 / 1.07 / 1.08 / 1.11,
// front-on seated 0.47 / 0.58, side-on seated 1.38 (caught by the other two).
//
// Robust to camera pitch because it is a ratio of two adjacent segments: a
// high or low camera foreshortens thigh and shin together and cancels out.
const SIT_THIGH_FORESHORTEN = 0.75;

// The same ratio read from the other end. SIT_THIGH_FORESHORTEN catches a thigh
// short relative to its shin; this catches a *shin* short relative to its thigh,
// which means the shin points at the lens - the signature of kneeling or being
// down on all fours. It was an unused half of an existing signal: the rule below
// only ever tested the low side.
//
// Measured over 3,106 corpus detections, the separation is total: no `stand`
// reaches 1.81 and no `sit` reaches 1.64, while 40 of 1,382 `fall` boxes clear
// 2.5 (up to 12.3). 2.5 is therefore well clear of every upright posture rather
// than fitted to the falls.
//
// Emitting `fall` for this is a deliberate choice and the weakest link in the
// gate: kneeling is not falling. It is justified by where the cases came from -
// people on all fours immediately after going down, which the torso angle misses
// because their torso is still upright - and it is guarded downstream by
// tracker.js's 1.2s sustain, which a deliberate kneel outlives but a stumble
// does not. Three of 108 crouching-bystander boxes trip it; that is the measured
// false-alarm cost.
const KNEEL_SHIN_FORESHORTEN = 2.5;

// --- squat ----------------------------------------------------------------
//
// Squat is a deep knee bend with the body folded down over the feet. It is
// carved out of what the rules below would otherwise call `sit`, and all three
// conditions must hold at once - each alone is far too broad.
//
// Measured medians across the corpus, which is what places these:
//
//            kneeAngle   hipAnkleDrop   stanceOffset
//   squat        95           0.75          0.21
//   sit         116           1.15          0.50
//   stand       175           1.47          0.08
//
// The ordering squat < sit < stand holds on both leg features, which is the
// whole basis for the class. Note `stand` also has a low stanceOffset - feet
// under hips is true of standing too - so that feature cannot gate alone; it is
// kneeAngle and hipAnkleDrop that exclude standing.
const SQUAT_KNEE_ANGLE = 130; // squat median 95, stand p10 155
const SQUAT_HIP_ANKLE_DROP = 1.0; // squat median 0.75, sit median 1.15
const SQUAT_STANCE_OFFSET = 0.5; // squat p75 0.36, fall median 0.81

// hipAnkleDrop also needs a *floor*, which is not symmetry for its own sake. The
// feature is signed: negative means the ankles sit above the hips, which is not
// a deep crouch but an inverted or sprawled body - someone on the ground with
// their legs up. Without this bound the gate accepted them, and a fall relabelled
// `squat` is a missed alarm rather than a cosmetic error.
//
// Found by running the gate over the corpus and reading the emitted reasons: a
// detection came back `knee 128deg, hips -2.36 over ankles`. Measured, 0.3 drops
// 42 such fall-to-squat leaks and costs 3 genuine crouches, and still sits below
// the squat class's own p10 of 0.41.
const SQUAT_HIP_ANKLE_DROP_MIN = 0.3;

// Confidence multipliers by how much information was actually available.
const TIER_FULL = 1.0; // hips + knees (+ ankles)
const TIER_NO_ANKLE = 0.85; // hips + knees, knee angle unavailable
const TIER_NO_LEGS = 0.6; // hips only - sit/stand is a guess, see below
const TIER_NO_HIPS = 0.4; // no torso vector at all
const DISAGREEMENT = 0.85; // kneeDrop and kneeAngle point different ways

// --- geometry helpers -----------------------------------------------------

const toDeg = (rad) => (rad * 180) / Math.PI;

// Midpoint of a left/right pair. Falls back to whichever single side is visible
// - a subject in profile legitimately shows only one hip - and returns null when
// neither is. A one-sided midpoint is slightly off-centre but the angles we take
// from it are not sensitive at that scale.
function midpoint(keypoints, leftIndex, rightIndex) {
  const l = keypoints[leftIndex];
  const r = keypoints[rightIndex];
  const lv = l && l.confidence >= KP_CONF_THRESHOLD;
  const rv = r && r.confidence >= KP_CONF_THRESHOLD;
  if (lv && rv) return { x: (l.x + r.x) / 2, y: (l.y + r.y) / 2 };
  if (lv) return { x: l.x, y: l.y };
  if (rv) return { x: r.x, y: r.y };
  return null;
}

// Interior angle at `vertex`, between the rays to `a` and `b`.
function angleAt(vertex, a, b) {
  const ax = a.x - vertex.x;
  const ay = a.y - vertex.y;
  const bx = b.x - vertex.x;
  const by = b.y - vertex.y;
  const la = Math.hypot(ax, ay);
  const lb = Math.hypot(bx, by);
  if (!la || !lb) return null;
  const cos = Math.max(-1, Math.min(1, (ax * bx + ay * by) / (la * lb)));
  return toDeg(Math.acos(cos));
}

// Scales confidence by how far the deciding measurement cleared its boundary, so
// a torso at 85 deg reads more confident than one that just crossed 50. Floored
// at 0.6 - a boundary call is uncertain, not worthless. This number has real
// behavioural effect: tracker.js weights its majority vote by it.
function margin(value, boundary, range) {
  return Math.max(0.6, Math.min(1, 0.6 + (0.4 * Math.abs(value - boundary)) / range));
}

// --- the squat second opinion ---------------------------------------------
//
// Consulted at exactly two points below, both of which were about to return
// `sit` at tier A. It is the one learned component in this file, and its scope
// is deliberately the narrowest thing that was measured to work.
//
// The full four-class classifier this is carved out of was rejected on
// 2026-08-12: `squat` transferred across domains (recall 0.809 against these
// gates' 0.427) but `fall` collapsed, the model calling 61% of POLAR's
// non-fallen people a fall where the gates call 11.5%. Taking only the squat
// half, and only where the gates have already declined to raise an alarm, keeps
// the result that held and discards the one that did not.
//
// Three invariants, and they are structural rather than promised - they follow
// from *where* this is called, so no threshold or model change can break them:
//
//   1. It never runs on a row the fall gates claimed. Those return above. So it
//      cannot suppress an alarm.
//   2. It never runs on a row that would answer `stand`. Stand precision was
//      the criterion the full replacement failed.
//   3. Its only possible output is `squat` in place of `sit` - two non-alarming
//      classes. Its worst case is a cosmetic relabel.
//
// Measured on corpus-2023, a domain it never saw in training or in threshold
// selection: squat recall 0.427 -> 0.536, sit recall 0.542 -> 0.475, `fall` and
// `stand` bit-identical. 12 genuine squats recovered against 4 correct `sit`
// calls broken, an exchange rate of 3:1 that held at 3.5:1 on the held-out
// POLAR split. Compare the alternative of simply widening SQUAT_HIP_ANKLE_DROP,
// measured in MODEL_CARD.md at 4 gained per 7 broken - a losing trade.
//
// It relaxes the squat gate's CEILING; it does not replace the gate's
// definition. `stanceOffset` stays a hard precondition at the same
// SQUAT_STANCE_OFFSET the gate uses, so the probe can only ever fire on a body
// whose feet are under it. That is not caution for its own sake - it was found
// by posture-check.mjs, which caught the unguarded probe calling a canonical
// chair-sit (knees level with the hips, feet projected 0.88 torso-lengths
// forward) a squat at 0.57. Feet forward of the hips is the one signature that
// separates a chair-sit from a crouch, the gate already encodes it, and the
// probe does not reliably learn it from raw joints.
//
// Measured cost of the guard on corpus-2023: 12 recovered squats become 10, and
// the exchange rate improves from 3.00:1 to 3.33:1. So it is cheap, and it buys
// back the property that the geometry defines the class while the model only
// decides how far inside it to reach.
//
// Requiring stanceOffset to be non-null also enforces tier A structurally - the
// feature needs the ankle - rather than by the caller remembering to.
//
// Returns a squat result, or null to leave the caller's `sit` alone.
function squatSecondOpinion(keypoints, box, f, personConf, leanPenalty) {
  if (f.stanceOffset === null || f.stanceOffset >= SQUAT_STANCE_OFFSET) return null;
  const p = squatProbability(probeFeatures(keypoints, box, KP_CONF_THRESHOLD));
  if (p < SQUAT_PROBE_THRESHOLD) return null;
  return {
    className: 'squat',
    confidence:
      personConf * TIER_FULL * leanPenalty * margin(p, SQUAT_PROBE_THRESHOLD, 1 - SQUAT_PROBE_THRESHOLD),
    tier: 'A',
    reason: `squat probe ${p.toFixed(2)} (geometry said sit)`,
  };
}

// --- feature extraction ---------------------------------------------------

function postureFeatures(keypoints, box) {
  const shoulder = midpoint(keypoints, 5, 6);
  const hip = midpoint(keypoints, 11, 12);
  const knee = midpoint(keypoints, 13, 14);
  const ankle = midpoint(keypoints, 15, 16);

  const width = Math.max(0, box.x2 - box.x1);
  const height = Math.max(0, box.y2 - box.y1);
  const f = {
    shoulder,
    hip,
    knee,
    ankle,
    aspect: height > 0 ? width / height : 0,
    torsoAngle: null,
    torsoLength: null,
    kneeDrop: null,
    kneeAngle: null,
    thighShinRatio: null,
    hipAnkleDrop: null,
    stanceOffset: null,
  };

  if (shoulder && hip) {
    const dx = Math.abs(hip.x - shoulder.x);
    const dy = Math.abs(hip.y - shoulder.y);
    f.torsoLength = Math.hypot(dx, dy);
    // atan2(|dx|, |dy|): 0 deg when the torso is vertical, 90 when horizontal.
    // Absolute values because upside-down is as horizontal-relevant as upright -
    // we care about the axis, not which end is up.
    f.torsoAngle = toDeg(Math.atan2(dx, dy));
  }
  if (hip && knee && f.torsoLength > 0) {
    f.kneeDrop = (knee.y - hip.y) / f.torsoLength;
  }
  if (hip && knee && ankle) {
    f.kneeAngle = angleAt(knee, hip, ankle);
    const thigh = Math.hypot(knee.x - hip.x, knee.y - hip.y);
    const shin = Math.hypot(ankle.x - knee.x, ankle.y - knee.y);
    if (shin > 0) f.thighShinRatio = thigh / shin;
  }
  // Squat features. Both need the ankle, which is what confines the squat class
  // to tier A - see SQUAT_* below.
  if (hip && ankle && f.torsoLength > 0) {
    // How far below the hips the ankles sit, in torso lengths. A chair-sit puts
    // a full vertical shin below the hip; a squat folds the body so the heels
    // come up toward it.
    f.hipAnkleDrop = (ankle.y - hip.y) / f.torsoLength;
    // How far the ankles sit horizontally from the hips. A chair-sit projects
    // the feet forward; a squat keeps them under the centre of mass.
    f.stanceOffset = Math.abs(ankle.x - hip.x) / f.torsoLength;
  }
  return f;
}

// --- classifier -----------------------------------------------------------

// Returns { className, confidence, tier, reason } for one person.
// `personConf` is the pose model's own detection score; the returned confidence
// is that scaled by how much geometric evidence supported the call.
function classifyPosture(keypoints, box, personConf) {
  const f = postureFeatures(keypoints, box);

  // Tier D - no torso vector at all. Nothing can be concluded, so emit the
  // safest non-alarming label at a weight low enough that any frame with real
  // evidence outvotes it in the tracker.
  if (f.torsoAngle === null) {
    return {
      className: 'stand',
      confidence: personConf * TIER_NO_HIPS,
      tier: 'D',
      reason: 'no shoulder/hip pair visible',
    };
  }

  // Fall: torso reads horizontal, either directly or via a wide box that implies
  // a foreshortened one.
  const horizontal = f.torsoAngle >= FALL_TORSO_ANGLE;
  const foreshortened = f.aspect >= FALL_ASPECT;
  if (horizontal || foreshortened) {
    return {
      className: 'fall',
      confidence: personConf * margin(f.torsoAngle, FALL_TORSO_ANGLE, 40),
      tier: 'A',
      reason: horizontal
        ? `torso ${f.torsoAngle.toFixed(0)}deg off vertical`
        : `box aspect ${f.aspect.toFixed(2)} with torso ${f.torsoAngle.toFixed(0)}deg`,
    };
  }

  // Not a fall. A torso between AMBIGUOUS_TORSO_ANGLE and FALL_TORSO_ANGLE is
  // mid-motion - keep the sit/stand answer but discount it.
  const leaning = f.torsoAngle >= AMBIGUOUS_TORSO_ANGLE;
  const leanPenalty = leaning ? 0.7 : 1;

  // Tier C - hips found but no knees. Sit and stand are genuinely
  // indistinguishable here: a waist-up crop of someone standing and a waist-up
  // crop of someone at a desk produce the same torso. We return `stand` because
  // it is the non-alarming default, at a confidence low enough to be overridden
  // by any frame that does see legs. This is a real limitation for close-range
  // webcam framing, not a solved case - see MODEL_CARD.md.
  if (f.kneeDrop === null) {
    return {
      className: 'stand',
      confidence: personConf * TIER_NO_LEGS * leanPenalty,
      tier: 'C',
      reason: 'knees not visible; sit/stand indeterminate',
    };
  }

  // Knees above the hips. Checked here, immediately after tier C, so it covers
  // tier B as well - the tier-B branch below has only kneeDrop to work with and
  // would otherwise answer `sit` for a body on its back with no ankles visible,
  // which is the exact frame that found this (a man flat on a dojo mat, torso
  // 25deg, aspect 1.96, kneeDrop -1.11, returned `sit` at 0.77).
  //
  // Emitted as a fall rather than as a discount, for the same reason the thigh
  // gate is decisive: this is a statement about geometry, not a correlation. No
  // seated posture puts the knees a quarter of a torso-length above the hips.
  if (f.kneeDrop !== null && f.kneeDrop <= INVERTED_KNEE_DROP) {
    return {
      className: 'fall',
      confidence:
        personConf *
        (f.kneeAngle === null ? TIER_NO_ANKLE : TIER_FULL) *
        margin(f.kneeDrop, INVERTED_KNEE_DROP, 1),
      tier: f.kneeAngle === null ? 'B' : 'A',
      reason: `knees ${(-f.kneeDrop).toFixed(2)} above hips (inverted)`,
    };
  }

  // Shin foreshortened past anything an in-plane leg can produce: the subject is
  // kneeling or on all fours. Checked before the squat gate because a kneel and
  // a deep crouch share the leg geometry, and the kneel is the more alarming
  // reading of the two - measured, taking it first costs nothing on the squat
  // class (17 crouches read `fall` either way) and recovers 6 more falls than
  // letting squat claim them.
  if (f.thighShinRatio !== null && f.thighShinRatio >= KNEEL_SHIN_FORESHORTEN) {
    return {
      className: 'fall',
      confidence: personConf * TIER_FULL * margin(f.thighShinRatio, KNEEL_SHIN_FORESHORTEN, 2),
      tier: 'A',
      reason: `shin ${(1 / f.thighShinRatio).toFixed(2)}x thigh (kneeling)`,
    };
  }

  // Squat. Placed *before* the front-on sit gate, which is not cosmetic: a deep
  // crouch foreshortens the thigh as well, so SIT_THIGH_FORESHORTEN claims 40%
  // of crouches first if the order is reversed. Measured, moving this ahead of
  // it takes the class from 27 to 51 caught at identical cost.
  //
  // All three features require the ankle, so this branch is unreachable without
  // the full leg chain. That is the intended limitation, not an oversight: at
  // tier B a squat and a chair-sit are geometrically identical, and the code
  // below correctly answers `sit` there rather than guessing.
  if (
    f.kneeAngle !== null &&
    f.hipAnkleDrop !== null &&
    f.stanceOffset !== null &&
    f.kneeAngle < SQUAT_KNEE_ANGLE &&
    f.hipAnkleDrop >= SQUAT_HIP_ANKLE_DROP_MIN &&
    f.hipAnkleDrop < SQUAT_HIP_ANKLE_DROP &&
    f.stanceOffset < SQUAT_STANCE_OFFSET
  ) {
    return {
      className: 'squat',
      confidence:
        personConf *
        TIER_FULL *
        leanPenalty *
        margin(f.hipAnkleDrop, SQUAT_HIP_ANKLE_DROP, 0.5),
      tier: 'A',
      reason: `knee ${f.kneeAngle.toFixed(0)}deg, hips ${f.hipAnkleDrop.toFixed(2)} over ankles`,
    };
  }

  // A foreshortened thigh is treated as decisive rather than as one vote among
  // three, because it is a statement about projection geometry rather than a
  // correlation: no standing pose puts a thigh at 0.58 of its own shin. Left as
  // a vote it would lose 2-1 to kneeDrop and kneeAngle in exactly the case it
  // exists to catch, since those two are what fail there.
  if (f.thighShinRatio !== null && f.thighShinRatio < SIT_THIGH_FORESHORTEN) {
    // A deep crouch foreshortens the thigh too, which is why the squat gate runs
    // ahead of this one. What reaches here is what that gate's thresholds
    // declined, and it is the larger of the two places the probe earns its keep.
    const probe = squatSecondOpinion(keypoints, box, f, personConf, leanPenalty);
    if (probe) return probe;
    return {
      className: 'sit',
      confidence:
        personConf *
        TIER_FULL *
        leanPenalty *
        margin(f.thighShinRatio, SIT_THIGH_FORESHORTEN, 0.4),
      tier: 'A',
      reason: `thigh ${f.thighShinRatio.toFixed(2)}x shin (foreshortened)`,
    };
  }

  const dropSaysSit = f.kneeDrop < STAND_KNEE_DROP;

  // Tier B - knees but no ankles, so no knee angle. kneeDrop alone decides.
  if (f.kneeAngle === null) {
    return {
      className: dropSaysSit ? 'sit' : 'stand',
      confidence:
        personConf *
        TIER_NO_ANKLE *
        leanPenalty *
        margin(f.kneeDrop, STAND_KNEE_DROP, 0.5),
      tier: 'B',
      reason: `kneeDrop ${f.kneeDrop.toFixed(2)}, ankles not visible`,
    };
  }

  // Tier A - full leg chain. Two independent signals; agreement is the common
  // case, and disagreement falls back to kneeDrop, which does not depend on the
  // ankle being correctly placed.
  //
  // The fallback was re-tested against the corpus rather than left on intuition:
  // they disagree on 527 of 3,106 detections, and deferring to kneeAngle instead
  // trades 25 correct `stand` calls for 3 correct `sit` calls with no change to
  // fall recall. kneeDrop stays.
  const angleSaysSit = f.kneeAngle < STAND_KNEE_ANGLE;
  const agree = dropSaysSit === angleSaysSit;
  // Only on the `sit` side. A `stand` answer here is never second-guessed -
  // see invariant 2 on squatSecondOpinion.
  if (dropSaysSit) {
    const probe = squatSecondOpinion(keypoints, box, f, personConf, leanPenalty);
    if (probe) return probe;
  }
  return {
    className: dropSaysSit ? 'sit' : 'stand',
    confidence:
      personConf *
      TIER_FULL *
      leanPenalty *
      (agree ? 1 : DISAGREEMENT) *
      margin(f.kneeDrop, STAND_KNEE_DROP, 0.5),
    tier: 'A',
    reason: `kneeDrop ${f.kneeDrop.toFixed(2)}, knee ${f.kneeAngle.toFixed(0)}deg${agree ? '' : ' (disagree)'}`,
  };
}

module.exports = { classifyPosture, postureFeatures };
