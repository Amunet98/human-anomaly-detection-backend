#!/usr/bin/env python3
"""Train the `squat` second opinion and emit it as a plain JS module.

Run with the project venv:
    ../../.venv/bin/python training/train_squat_probe.py

WHAT THIS IS, AND WHY IT IS NOT THE CLASSIFIER NEXT DOOR
--------------------------------------------------------
`train_posture_keypoints.py` trains a full four-class replacement for
posture.js. It was measured cross-domain on 2026-08-12 and rejected: `squat`
transferred (recall 0.809 against the gates' 0.427) but `fall` collapsed, the
model calling 61% of POLAR's non-fallen people a fall against the gates' 11.5%.
See MODEL_CARD.md, "Cross-domain transfer, measured".

This script takes the half that worked. It trains **one binary probe, squat vs
not-squat**, and that probe is consulted in exactly one place: when the
geometric gates have already decided `sit`, at tier A. It can turn a `sit` into
a `squat` and it can do nothing else.

Three properties follow from that placement, and they are the whole argument for
shipping this when the full replacement was rejected:

  1. **It cannot cost an alarm.** The fall gates are steps 1-3 of posture.js and
     run first. Any row the probe sees is one they already declined, so no
     `fall` can be suppressed - not by a bad probe, not by a mis-set threshold,
     not by a domain it has never seen.
  2. **It cannot touch `stand`.** Stand precision falling from 0.876 to 0.744 is
     the criterion that failed the full replacement. The probe never runs on a
     row the gates called `stand`.
  3. **Its worst case is a relabel between two non-alarming classes.** On the
     2023 corpus most `sit` rows are in fact missed falls; moving one to `squat`
     changes which wrong label it carries and nothing else.

THE FEATURE ENCODING IS A TRAP - READ THIS BEFORE CHANGING EITHER SIDE
-----------------------------------------------------------------------
Training rows come from `feature-dump.mjs --keypoints`, which records the
**wire** form of a detection, not the internal one. `keypointsForWire()` in
inference.js nulls the coordinates of any joint below KP_CONF_THRESHOLD (0.65)
and rounds what survives to whole pixels; the box is rounded too. `featurise()`
then turns a nulled joint into `(0, 0, 0)` - the confidence is discarded along
with the coordinates, so `c == 0` is the "this joint is not here" marker.

Inside posture.js keypoints are full precision and never nulled. A JS feature
builder that reads them directly would hand this model coordinates for occluded
joints where every training row had zeros - a systematic shift on precisely the
joints that decide occlusion tier. The generated module therefore reproduces the
wire transform, and `squat-probe-check.mjs` asserts the two agree numerically on
real corpus rows. If you change the encoding here, that check is what tells you
the JS half did not follow.

WHY PLAIN JS AND NOT ONNX
-------------------------
The ensemble is ~1,500 nodes. Emitted as flat arrays it is smaller than the ONNX
runtime's own overhead, needs no second InferenceSession, and - the reason that
actually matters - stays **synchronous**, so `classifyPosture` remains a pure
function. Making it async would ripple through inference.js, detector.worker.js
and every check script, to buy nothing.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_posture_keypoints import read_corpus, featurise  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = 0

# Threshold selection rule, fixed before looking at any cross-domain number.
#
# Raw precision is the wrong objective here, and getting that wrong once is what
# produced a first draft of this script that shipped a probe firing 13 times.
# The probe only ever runs on rows the gates predicted `sit`, so each firing
# falls into one of three buckets by ground truth:
#
#   truth squat  the gates were wrong and the probe fixes it        GAIN
#   truth sit    the gates were RIGHT and the probe breaks it       COST
#   truth fall   the gates were already wrong; both labels are
#   or stand     non-alarming, so this swaps one wrong label for
#                another and changes nothing that reaches a user    NEUTRAL
#
# 1 - precision counts the neutral bucket as damage. On corpus-2023 that bucket
# is 71% of the branch, so precision reads as a catastrophe while the actual
# exchange rate is better than 3:1 in favour.
#
# The objective is therefore GAIN - COST_WEIGHT * COST, with a correct call
# broken counted as twice as bad as a wrong one fixed. Asymmetric on purpose:
# regressions are worse than missed improvements when something already ships.
COST_WEIGHT = 2.0
MIN_RATIO = 2.0  # refuse to ship if the achieved gain:cost is worse than this

# The probe relaxes the squat gate's CEILING; it does not replace the gate's
# definition. `stanceOffset` - how far forward of the hips the ankles project -
# stays a hard precondition at the gate's own SQUAT_STANCE_OFFSET, because feet
# forward of the body is the signature separating a chair-sit from a crouch and
# the probe does not reliably learn it from raw joints. posture-check.mjs found
# this: unguarded, the probe called a canonical chair-sit (stanceOffset 0.88) a
# squat at 0.57. Measured cost on corpus-2023: 12 recovered squats become 10,
# and the exchange rate improves from 3.00:1 to 3.33:1.
#
# Must stay in step with SQUAT_STANCE_OFFSET in both copies of posture.js.
SQUAT_STANCE_OFFSET = 0.5
CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

# Where the generated module lands. The two repos share no package, so - exactly
# as with posture.js and its browser twin - the same logic ships twice in two
# module systems. The numbers are identical; only the epilogue differs, and
# `npm run squat-probe` is what asserts the two agree rather than trusting it.
TARGETS = [
    (ROOT / "human-anomaly-detection-backend-main" / "squat-probe.js", "cjs"),
    (ROOT / "frontend-new" / "src" / "lib" / "detect" / "squat-probe.js", "esm"),
]

EPILOGUE = {
    "esm": "export { SQUAT_PROBE_THRESHOLD, probeFeatures, squatProbability };\n",
    "cjs": "module.exports = { SQUAT_PROBE_THRESHOLD, probeFeatures, squatProbability };\n",
}


def tier_a(name):
    return [r for r in read_corpus(name) if r["tier"] == "A"]


def buckets(fired_truth):
    """gain / cost / neutral for a set of firings. See COST_WEIGHT above."""
    c = Counter(fired_truth)
    return c["squat"], c["sit"], c["fall"] + c["stand"]


def main():
    pol = tier_a("polar")
    X, y, groups, base, _ = featurise(pol)
    yb = (y == "squat").astype(int)
    print(f"POLAR tier A: {len(yb)} rows, {yb.sum()} squat, {len(set(groups))} images")

    # Threshold is chosen here and nowhere else. corpus-2023 is the held-out
    # domain and must not influence it.
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(X, yb, groups))
    sel = GradientBoostingClassifier(random_state=SEED).fit(X[tr], yb[tr])

    # Select on the rows the probe will actually see in production: tier A, and
    # the gates already said `sit`. Selecting over all tier-A rows would tune for
    # a population the probe never meets.
    va_fire = va[base[va] == "sit"]
    p_va = sel.predict_proba(X[va_fire])[:, 1]
    truth_va = y[va_fire]
    print(f"\nthreshold selection on {len(va_fire)} held-out POLAR `sit` rows "
          f"({(truth_va == 'squat').sum()} truly squat, {(truth_va == 'sit').sum()} truly sit)")
    best, chosen = None, None
    for t in CANDIDATES:
        m = p_va >= t
        gain, cost, neutral = buckets(truth_va[m])
        obj = gain - COST_WEIGHT * cost
        ratio = gain / cost if cost else float("inf")
        if best is None or obj > best:
            best, chosen = obj, t
        print(f"  thr={t:.2f}  fires {m.sum():4}  gain {gain:3}  cost {cost:3}  "
              f"neutral {neutral:3}  ratio {ratio:5.2f}:1  objective {obj:+6.1f}")
    print(f"\nrule: maximise gain - {COST_WEIGHT} x cost on held-out POLAR -> {chosen}")

    # Refit on all of POLAR tier A for the shipped model. The threshold stays as
    # selected above; refitting changes the fit, not the rule.
    model = GradientBoostingClassifier(random_state=SEED).fit(X, yb)

    # --- held-out domain: corpus-2023, never seen in training or selection ---
    te = tier_a("2023")
    Xt, yt, _, bt, _ = featurise(te)
    stance = np.array(
        [r["f"]["stanceOffset"] if r["f"]["stanceOffset"] is not None else np.inf for r in te]
    )
    fire = np.flatnonzero((bt == "sit") & (stance < SQUAT_STANCE_OFFSET))
    pt = model.predict_proba(Xt[fire])[:, 1]
    hit = fire[pt >= chosen]

    print(f"\n=== CROSS-DOMAIN: corpus-2023 tier A, {len(yt)} rows "
          f"({(yt == 'squat').sum()} squat) ===")
    print(f"  gates answer `sit` AND stanceOffset < {SQUAT_STANCE_OFFSET} on {len(fire)} of "
          f"them; the probe fires on {len(hit)}")
    gain, cost, neutral = buckets(yt[hit])
    ratio = gain / cost if cost else float("inf")
    print(f"    GAIN    genuine squats recovered      : {gain}")
    print(f"    COST    correct `sit` calls broken    : {cost}")
    print(f"    NEUTRAL already-wrong rows relabelled : {neutral}  "
          f"(missed falls and stands the gates had already called `sit`;")
    print(f"                                            no alarm is lost - the fall "
          f"gates declined these before the probe ran)")
    print(f"    exchange rate                         : {ratio:.2f} : 1")
    print("    For comparison, MODEL_CARD's measured cost of widening the squat gate")
    print("    instead was 4 gained : 7 broken at 130/1.2/0.5 - a losing trade.")
    if ratio < MIN_RATIO:
        sys.exit(f"\nexchange rate {ratio:.2f}:1 is worse than the {MIN_RATIO}:1 floor - "
                 f"not shipping a probe. Nothing was written.")

    after = bt.copy()
    after[hit] = "squat"
    for cls in ("squat", "sit", "stand", "fall"):
        n = (yt == cls).sum()
        if not n:
            continue
        r0 = (bt[yt == cls] == cls).mean()
        r1 = (after[yt == cls] == cls).mean()
        flag = "" if abs(r1 - r0) < 1e-9 else ("  <- improved" if r1 > r0 else "  <- cost")
        print(f"    {cls:6} recall {r0:.3f} -> {r1:.3f}{flag}")
    assert (after[bt == "fall"] == "fall").all(), "probe must never touch a fall call"
    assert (after[bt == "stand"] == "stand").all(), "probe must never touch a stand call"
    print("    invariants hold: no `fall` and no `stand` call was altered.")

    emit(model, chosen)
    write_check_fixtures(model, te, Xt)


def write_check_fixtures(model, rows, X, n=300):
    """Reference values for squat-probe-check.mjs.

    The JS has to rebuild the 52 features from the raw wire row and then walk the
    trees, and either half can drift from this script silently. Both are dumped
    so a failure says which one broke, and the rows come from corpus-2023 - the
    domain the model did NOT train on, so they exercise the occlusion and
    coordinate shapes that the training distribution under-represents.
    """
    probs = model.predict_proba(X[:n])[:, 1]
    out = HERE / "squat-probe-fixtures.json"
    out.write_text(
        json.dumps(
            {
                "note": "Generated by train_squat_probe.py. Consumed by "
                        "frontend-new/scripts/squat-probe-check.mjs.",
                "kpConfThreshold": 0.65,
                "rows": [
                    {
                        "file": r["file"],
                        "box": r["box"],
                        "kp": r["kp"],
                        "features": [round(float(v), 9) for v in X[i]],
                        "p": round(float(probs[i]), 12),
                    }
                    for i, r in enumerate(rows[:n])
                ],
            }
        )
        + "\n"
    )
    print(f"wrote {out}  ({len(probs)} reference rows for the JS check)")


def emit(model, threshold):
    """Flatten the ensemble into arrays a JS module can walk without a runtime.

    sklearn's raw score is `init + lr * sum(tree.predict(x))`; the learning rate
    is folded into the leaf values here so the consumer only sums. Probability is
    the logistic of that, matching predict_proba for a binary log-loss GB.
    """
    feat, thr, left, right, value = [], [], [], [], []
    roots = []
    for est in model.estimators_[:, 0]:
        t = est.tree_
        roots.append(len(feat))
        off = len(feat)
        for i in range(t.node_count):
            is_leaf = t.children_left[i] == -1
            feat.append(-1 if is_leaf else int(t.feature[i]))
            thr.append(0.0 if is_leaf else float(t.threshold[i]))
            left.append(-1 if is_leaf else int(t.children_left[i]) + off)
            right.append(-1 if is_leaf else int(t.children_right[i]) + off)
            value.append(float(t.value[i][0][0]) * model.learning_rate if is_leaf else 0.0)

    prior = float(model._raw_predict_init(np.zeros((1, 52), dtype=np.float32)).ravel()[0])
    js = TEMPLATE.format(
        threshold=threshold,
        prior=repr(prior),
        nodes=len(feat),
        trees=len(roots),
        roots=compact(roots),
        feat=compact(feat),
        thr=compact([repr(v) for v in thr]),
        left=compact(left),
        right=compact(right),
        value=compact([repr(v) for v in value]),
    )
    for target, flavour in TARGETS:
        target.write_text(js + EPILOGUE[flavour])
        print(f"\nwrote {target}  ({(len(js) + len(EPILOGUE[flavour])) // 1024} KB, {flavour})")
    print("The two differ only in the export epilogue. `npm run squat-probe` "
          "asserts the JS agrees with this script numerically.")


def compact(seq, width=96):
    """One long array, wrapped, so the generated file is diffable rather than
    one 40 KB line that every review tool refuses to render."""
    out, line = [], "  "
    for v in seq:
        s = f"{v},"
        if len(line) + len(s) > width:
            out.append(line)
            line = "  "
        line += s
    out.append(line)
    return "\n".join(out)


TEMPLATE = '''// GENERATED by training/train_squat_probe.py - do not edit by hand.
//
// A binary squat-vs-not probe, consulted by posture.js in exactly one place:
// when the geometric gates have already answered `sit` at tier A. It can turn
// that `sit` into a `squat` and it can do nothing else - it never sees a row the
// fall gates claimed, so it cannot cost an alarm, and it never sees a `stand`.
//
// Trained on POLAR tier A ({trees} trees, {nodes} nodes). Rationale, the
// cross-domain measurement behind it and the rejected four-class alternative are
// in training/MODEL_CARD.md.
//
// The feature encoding mirrors the WIRE form the training rows came from:
// coordinates are box-relative and integer-rounded, and any joint below
// KP_CONF_THRESHOLD contributes (0, 0, 0) - confidence included, since c === 0
// is the "joint absent" marker the model was trained to read. Do not "improve"
// this by passing full-precision coordinates for occluded joints.

const SQUAT_PROBE_THRESHOLD = {threshold};

const PRIOR = {prior};
const ROOTS = [
{roots}
];
const FEAT = [
{feat}
];
const THR = [
{thr}
];
const LEFT = [
{left}
];
const RIGHT = [
{right}
];
const VALUE = [
{value}
];

// 17 joints as box-relative x, y, confidence, then the box aspect. Must match
// featurise() in training/train_posture_keypoints.py exactly - including its
// dtype. That function returns float32, so a Float64Array here would feed the
// trees values a few 1e-7 away from the ones they were fitted against; harmless
// almost always, and a different answer for any feature sitting on a split
// threshold. Float32Array makes the rounding identical for free.
function probeFeatures(keypoints, box, kpConfThreshold) {{
  const x1 = Math.round(box.x1);
  const y1 = Math.round(box.y1);
  const w = Math.max(1, Math.round(box.x2) - x1);
  const h = Math.max(1, Math.round(box.y2) - y1);
  const f = new Float32Array(52);
  for (let i = 0; i < 17; i += 1) {{
    const k = keypoints[i];
    // Rounded to 2dp before the comparison because that is what went over the
    // wire and into the training rows: a joint at 0.6449 was recorded as 0.64.
    const c = k ? Math.round(k.confidence * 100) / 100 : 0;
    if (!k || k.x === null || k.y === null || c < kpConfThreshold) continue;
    f[i * 3] = (Math.round(k.x) - x1) / w;
    f[i * 3 + 1] = (Math.round(k.y) - y1) / h;
    f[i * 3 + 2] = c;
  }}
  f[51] = w / h;
  return f;
}}

// P(squat) for one detection. Pure, synchronous, no model file to load.
function squatProbability(features) {{
  let raw = PRIOR;
  for (let t = 0; t < ROOTS.length; t += 1) {{
    let n = ROOTS[t];
    while (FEAT[n] !== -1) {{
      n = features[FEAT[n]] <= THR[n] ? LEFT[n] : RIGHT[n];
    }}
    raw += VALUE[n];
  }}
  return 1 / (1 + Math.exp(-raw));
}}
'''


if __name__ == "__main__":
    main()
