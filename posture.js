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
const KP_CONF_THRESHOLD = 0.5;

// --- tuning ---------------------------------------------------------------

// Torso angle off vertical, in degrees, above which the subject reads as
// horizontal. Measured: upright fixtures sit at 1.6/1.7/14.0 deg, the two falls
// at 57.6 and 72.3. 50 leaves ~36 deg of margin below and ~8 deg above, and is
// deliberately nearer the falls: a person bending to pick something up passes
// through 40-60 deg, and the tracker's FALL_CONFIRM_MS=1200 sustain requirement -
// not this threshold - is what stops that registering as a fall.
const FALL_TORSO_ANGLE = 50;

// A person lying directly toward or away from the camera has a foreshortened
// torso, so the angle understates how horizontal they are. A wide box plus a
// partly-tilted torso covers that case. Measured: the street fall's box aspect
// is 2.28, every upright fixture is between 0.30 and 0.84.
const FALL_ASPECT = 1.5;
const FALL_ASPECT_TORSO_ANGLE = 30;

// Below FALL_TORSO_ANGLE but above this, the torso is neither clearly upright
// nor clearly horizontal (bending, crouching, reaching down). We do not flip the
// class here - we keep the sit/stand answer and discount its confidence, letting
// the tracker's confidence-weighted vote settle it over its 7-frame window.
const AMBIGUOUS_TORSO_ANGLE = 30;

// Vertical hip-to-knee separation, normalised by torso length. Measured: seated
// on a bench -0.16 (knees above hips, legs drawn up), standing 0.74 to 0.83.
const STAND_KNEE_DROP = 0.5;

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
  const foreshortened = f.aspect >= FALL_ASPECT && f.torsoAngle >= FALL_ASPECT_TORSO_ANGLE;
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

  // A foreshortened thigh is treated as decisive rather than as one vote among
  // three, because it is a statement about projection geometry rather than a
  // correlation: no standing pose puts a thigh at 0.58 of its own shin. Left as
  // a vote it would lose 2-1 to kneeDrop and kneeAngle in exactly the case it
  // exists to catch, since those two are what fail there.
  if (f.thighShinRatio !== null && f.thighShinRatio < SIT_THIGH_FORESHORTEN) {
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
  const angleSaysSit = f.kneeAngle < STAND_KNEE_ANGLE;
  const agree = dropSaysSit === angleSaysSit;
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
