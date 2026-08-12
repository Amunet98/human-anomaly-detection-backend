#!/usr/bin/env python3
"""Train a keypoints -> posture classifier and compare it to the geometry it replaces.

Run with the project venv:
    ../../.venv/bin/python training/train_posture_keypoints.py

Why keypoints rather than pixels: the 2023 model learned scene appearance
(bench, floor, indoor room) and collapsed to 76.7% under perturbation. This
input is 17 joint positions and nothing else, so that shortcut is not available -
the background is not in the data. See MODEL_CARD.md, "Why the architecture
changed".

What it replaces: posture.js's hand-tuned gates, measured at 49.5% fall recall
and 44.3% squat recall on the corpus. Those numbers are the bar. The geometric
classifier's own answer travels in each row as `pred`, so the comparison is made
on the identical test split rather than against a remembered figure.

Three things this script does that a naive fit would not:

  1. Splits by IMAGE, not by row. Multi-person frames put several rows in the
     same photo, and a random split would train and test on the same scene.
  2. Reports the geometric baseline on that same split, so "better" means
     better than what ships, not better than chance.
  3. Probes for a visibility shortcut. The two sources differ in occlusion
     (corpus-2023 is 76% tier A, POLAR 59%), so a model could learn "how many
     joints are visible" as a proxy for "which dataset this came from", and
     that would look like accuracy. A model trained on the confidences ALONE
     measures how much signal that leak carries.

CROSS-DOMAIN MODE (--train-on / --test-on), added 2026-08-12
------------------------------------------------------------
Everything above measures IN-distribution: the grouped split holds out images,
not domains. That is why the first version of this model looked like a clear win
(0.830 vs 0.693) and then lost on the held-out fixtures - the fixtures are a
third domain, domestic interiors, that neither corpus covers.

`--train-on polar --test-on 2023` trains on one corpus entirely and tests on the
other entirely. That is a real domain shift, and it is the number that decides
whether this model can be trusted anywhere but where it was fitted.

**One structural limit, worth knowing before reading any output.** The corpora
are nearly class-disjoint:

                fall    sit    squat   stand
  corpus-2023   1970     89      170     877
  corpus-polar     0   1908     1922    1954

So `fall` cannot be cross-domain tested at all with local data - train on POLAR
and the model has never seen one. What IS testable is sit / squat / stand, which
both corpora carry, and those are exactly the three classes the planned hybrid
integration asks the model for (posture.js keeps authority over `fall`). Use
`--classes sit,squat,stand` for that arm; it is the informative one.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSES = ["fall", "sit", "squat", "stand"]
SEED = 0

# --train-on / --test-on names -> staged corpus directory. Both were written by
# build-corpus.py and both carry a keypoints.jsonl in the same row shape.
CORPORA = {"2023": "corpus-2023", "polar": "corpus-polar"}


def load(path: Path, drop: set[str] | None = None, src: str | None = None):
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r["gt"]:
            continue  # unlabelled: the pose model found someone the annotator did not
        if drop and r["gt"] == "fall" and r["file"] in drop:
            continue  # label disputed by the geometry - see disputed-fall-labels.txt
        if src:
            r["src"] = src
        rows.append(r)
    return rows


def read_corpus(name: str):
    """One staged corpus by --train-on/--test-on name, disputed labels dropped."""
    corpus = ROOT / CORPORA[name]
    kp = corpus / "keypoints.jsonl"
    if not kp.exists():
        sys.exit(f"missing {kp}\nRun feature-dump.mjs --keypoints --corpus {CORPORA[name]} first.")
    disputed_file = corpus / "disputed-fall-labels.txt"
    disputed = set(disputed_file.read_text().split()) if disputed_file.exists() else None
    return load(kp, drop=disputed, src=name)


def featurise(rows):
    """17 joints as box-relative x, y, confidence, plus the box aspect.

    Box-relative rather than absolute so the model cannot learn where in the
    frame people tend to be, or how large the image was - both of which are
    scene properties wearing a posture costume.

    Missing joints become (0, 0, 0). The zeros are safe ONLY because the
    confidence rides alongside: c=0 marks the coordinates as meaningless. Drop
    the confidence channel and this encoding silently claims every unseen joint
    is at the top-left corner of the box.
    """
    X, y, groups, base, src = [], [], [], [], []
    for r in rows:
        x1, y1, x2, y2 = r["box"]
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        f = []
        for kx, ky, kc in r["kp"]:
            if kx is None or ky is None:
                f += [0.0, 0.0, 0.0]
            else:
                f += [(kx - x1) / w, (ky - y1) / h, kc]
        f.append(w / h)
        X.append(f)
        y.append(r["gt"])
        groups.append(r["file"])  # split by image, never by row
        base.append(r["pred"])  # what posture.js said, for the baseline
        src.append(r.get("src", ""))
    return (
        np.array(X, dtype=np.float32),
        np.array(y),
        np.array(groups),
        np.array(base),
        np.array(src),
    )


def score(name, truth, pred):
    acc = (truth == pred).mean()
    print(f"\n--- {name} — accuracy {acc:.3f} ---")
    # Always report over all four labels even on a 3-class run: the gates and the
    # hybrid can still emit `fall` on a row whose truth is sit/squat/stand, and
    # hiding that column would hide a real error.
    print(classification_report(truth, pred, labels=CLASSES, zero_division=0, digits=3))
    return acc


def per_tier(name, truth, pred, tiers, classes):
    """Recall per class per occlusion tier.

    Tier is not decoration here. The squat gate is tier-A only by construction -
    it needs ankles - so a model that beats it overall might be winning entirely
    at tier B/C where the gate declines to answer, which is a different and much
    weaker claim than beating it where it competes.
    """
    print(f"\n--- {name} — recall by tier ---")
    print("  " + "class".ljust(8) + "".join(t.ljust(14) for t in "ABCD"))
    for c in classes:
        cells = []
        for t in "ABCD":
            m = (truth == c) & (tiers == t)
            cells.append(f"{(pred[m] == c).mean():.3f} (n={m.sum()})" if m.any() else "-")
        print("  " + c.ljust(8) + "".join(x.ljust(14) for x in cells))


def calibration(name, truth, proba, classes, bins=10):
    """Per-class reliability and ECE.

    Why this matters as much as accuracy: if the model ships, its probability
    becomes part of the posture confidence that tracker.js weights its 7-frame
    majority vote by. A model that is accurate but overconfident on `squat`
    distorts that vote even when its argmax is right, so the number the tracker
    consumes has to mean what it says.

    ECE is the average gap between stated confidence and observed accuracy,
    weighted by how many predictions fall in each confidence bin. 0 is perfect;
    anything above ~0.10 means the confidence is decorative.
    """
    print(f"\n--- {name} — calibration (ECE, lower is better) ---")
    edges = np.linspace(0, 1, bins + 1)
    for i, c in enumerate(classes):
        p = proba[:, i]
        hit = (truth == c).astype(float)
        ece, shown = 0.0, []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
            if not m.any():
                continue
            gap = abs(p[m].mean() - hit[m].mean())
            ece += m.sum() / len(p) * gap
            if m.sum() >= max(20, len(p) // 100):
                shown.append(f"{lo:.1f}-{hi:.1f}: said {p[m].mean():.2f} was {hit[m].mean():.2f}")
        print(f"  {c:6} ECE={ece:.3f}   " + " | ".join(shown[:4]))
    # Overall: how well the top-1 confidence predicts top-1 correctness.
    top = proba.max(axis=1)
    correct = (np.array(classes)[proba.argmax(axis=1)] == truth).astype(float)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (top > lo) & (top <= hi) if lo > 0 else (top >= lo) & (top <= hi)
        if m.any():
            ece += m.sum() / len(top) * abs(top[m].mean() - correct[m].mean())
    print(f"  {'top-1':6} ECE={ece:.3f}   mean confidence {top.mean():.3f} vs accuracy {correct.mean():.3f}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--tier-a",
        action="store_true",
        help="only detections with the full leg chain - where the gates can compete",
    )
    ap.add_argument(
        "--classes",
        default=",".join(CLASSES),
        help="comma-separated class subset. `sit,squat,stand` is the cross-domain "
        "arm: those are the three both corpora carry, and the three the hybrid "
        "would ask the model for.",
    )
    ap.add_argument(
        "--train-on",
        choices=sorted(CORPORA),
        help="train on this corpus entirely. With --test-on it replaces the "
        "grouped split with a real domain shift.",
    )
    ap.add_argument("--test-on", choices=sorted(CORPORA), help="test on this corpus entirely")
    ap.add_argument(
        "--calibration",
        action="store_true",
        help="per-class reliability and ECE - whether the confidence means anything",
    )
    ap.add_argument(
        "--export",
        action="store_true",
        help="write posture_keypoints.onnx. Off by default now that most runs are "
        "measurement arms that should not overwrite the artifact.",
    )
    args = ap.parse_args(argv)
    args.classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    for c in args.classes:
        if c not in CLASSES:
            ap.error(f"unknown class {c!r}; known: {CLASSES}")
    if bool(args.train_on) != bool(args.test_on):
        ap.error("--train-on and --test-on go together")
    return args


def main(argv=None):
    args = parse_args(argv)
    keep = set(args.classes)
    cross = args.train_on is not None and args.train_on != args.test_on

    def prep(rows, label):
        rows = [r for r in rows if r["gt"] in keep]
        if args.tier_a:
            rows = [r for r in rows if r["tier"] == "A"]
        print(f"{label:12} rows {len(rows):5}  images {len({r['file'] for r in rows}):5}  "
              f"{dict(sorted(Counter(r['gt'] for r in rows).items()))}")
        return rows

    if args.tier_a:
        print("--tier-a: full leg chain only")
    print(f"classes: {args.classes}")

    if args.train_on:
        # Cross-domain (or single-corpus in-distribution, if the two names match).
        tr_rows = prep(read_corpus(args.train_on), f"train[{args.train_on}]")
        if cross:
            te_rows = prep(read_corpus(args.test_on), f"test[{args.test_on}]")
            X_tr, y_tr, *_ = featurise(tr_rows)
            X_te, y_te, _, base_te, src_te = featurise(te_rows)
            tiers_te = np.array([r["tier"] for r in te_rows])
            print(f"\nCROSS-DOMAIN: train on {args.train_on} entirely, test on "
                  f"{args.test_on} entirely. No image, scene or dataset is shared.")
        else:
            X, y, groups, base, src = featurise(tr_rows)
            tr, te = next(
                GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups)
            )
            X_tr, y_tr = X[tr], y[tr]
            X_te, y_te, base_te, src_te = X[te], y[te], base[te], src[te]
            tiers_te = np.array([r["tier"] for r in tr_rows])[te]
            print(f"\nIN-DISTRIBUTION on {args.train_on}: grouped 75/25 split, "
                  f"{len(set(groups[tr]))} / {len(set(groups[te]))} images, no overlap.")
    else:
        # The original arm: both corpora mixed, held out by image.
        rows = prep(read_corpus("2023") + read_corpus("polar"), "mixed")
        X, y, groups, base, src = featurise(rows)
        tr, te = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups)
        )
        X_tr, y_tr = X[tr], y[tr]
        X_te, y_te, base_te, src_te = X[te], y[te], base[te], src[te]
        tiers_te = np.array([r["tier"] for r in rows])[te]
        print(f"\nIN-DISTRIBUTION, mixed: grouped 75/25 split, "
              f"{len(set(groups[tr]))} / {len(set(groups[te]))} images, no overlap.")
        print("Read this arm as the ceiling, not the result - the test images are "
              "held out but the *domains* are not.")

    print(f"train {len(y_tr)} rows / test {len(y_te)} rows  features {X_tr.shape[1]}")

    # The bar: what ships today, on this exact test set.
    score("BASELINE — posture.js geometric gates", y_te, base_te)
    per_tier("BASELINE — posture.js geometric gates", y_te, base_te, tiers_te, args.classes)

    models = {
        "MLP (128,64)": make_pipeline(
            StandardScaler(), MLPClassifier((128, 64), max_iter=600, random_state=SEED)
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    }
    fitted, preds = {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        fitted[name] = m
        preds[name] = m.predict(X_te)
        score(name, y_te, preds[name])

    best = max(fitted, key=lambda n: (preds[n] == y_te).mean())
    print(f"\nbest by accuracy: {best}")
    per_tier(best, y_te, preds[best], tiers_te, args.classes)

    # THE HYBRID, simulated. This is the arm that decides the integration, and it
    # is not the same thing as the model's own score above.
    #
    # In the planned integration posture.js keeps authority over `fall`: its
    # gates run first and the model only ever sees rows they declined. So on any
    # row where the gates said `fall`, the system still says `fall` however the
    # model would have voted - the model can neither rescue a false alarm nor
    # cause a missed one. Comparing the model's standalone predictions to the
    # gates therefore measures a system nobody proposed to build. This does.
    gate_fall = base_te == "fall"
    hybrid = np.where(gate_fall, "fall", preds[best])
    print(f"\n({gate_fall.sum()} of {len(y_te)} test rows were claimed by the fall "
          f"gates and are passed through unchanged)")
    score("HYBRID — fall gates first, then the model", y_te, hybrid)
    per_tier("HYBRID", y_te, hybrid, tiers_te, args.classes)

    if args.calibration:
        calibration(best, y_te, fitted[best].predict_proba(X_te), list(fitted[best].classes_))

    # Ablation. The confidences encode which joints the pose model found, and
    # the two sources differ in occlusion - so confidence alone partly identifies
    # the dataset, which a model could ride instead of learning posture. Measured
    # 2026-08-12: confidences alone reach 0.621 against 0.357 chance, so the leak
    # is real. What clears the model is that coordinates alone score *identically*
    # to the full feature set - it does not use the leak even though it is there.
    #
    # If a future change makes "everything" beat "coordinates only" by more than
    # noise, that gap is the shortcut being exploited, not new signal.
    #
    # The leak is a same-corpus artifact: in a cross-domain arm every train row
    # comes from one dataset and every test row from the other, so "which dataset
    # is this" carries no information about the label. Skipped there.
    if not cross:
        conf_idx = [i for i in range(2, 51, 3)]
        coord_idx = [i for i in range(51) if i % 3 != 2] + [51]
        print("\n--- ABLATION — which inputs carry the signal ---")
        for label, idx in (
            ("coordinates only (no confidence)", coord_idx),
            ("confidences only (no coordinates)", conf_idx),
            ("everything", list(range(X_tr.shape[1]))),
        ):
            a = GradientBoostingClassifier(random_state=SEED).fit(X_tr[:, idx], y_tr)
            print(f"  {label:34} {(a.predict(X_te[:, idx]) == y_te).mean():.3f}")
        print(f"  {'chance (majority class)':34} {max(Counter(y_te).values())/len(y_te):.3f}")

    # False alarms: the POLAR rows contain no falls, so any `fall` is wrong. Only
    # meaningful when the model is allowed to emit one.
    mask = src_te == "polar"
    if mask.any() and "fall" in keep:
        print(f"\n--- FALSE ALARMS on {mask.sum()} POLAR test rows (no falls exist here) ---")
        print(f"  {'baseline (posture.js)':24} {(base_te[mask] == 'fall').mean():.3f}")
        for name in fitted:
            print(f"  {name:24} {(preds[name][mask] == 'fall').mean():.3f}")
        print("  Lower is better. A model that improves posture accuracy while")
        print("  raising this is a worse system - see MODEL_CARD, 'False-alarm rate'.")

    print("\nConfusion (rows = truth, cols = predicted), order:", CLASSES)
    print(confusion_matrix(y_te, preds[best], labels=CLASSES))

    if not args.export:
        print("\n(no --export: this was a measurement run, posture_keypoints.onnx untouched)")
        return

    # Export. ONNX so it drops into the same runtime the pose model already uses,
    # on both the browser and server sides.
    #
    # Model choice is constrained by what skl2onnx can actually convert, which is
    # not the same as what sklearn can fit: HistGradientBoosting fails to convert
    # outright, and RandomForest converts to ~33 MB, which is unshippable beside a
    # 12.9 MB pose model. GradientBoosting lands around 200 KB and MLP around
    # 20 KB.
    out = HERE / "posture_keypoints.onnx"
    try:
        from skl2onnx import to_onnx

        # zipmap=False makes the second output a plain [n, k] float tensor instead
        # of a sequence-of-maps. The map form is awkward in onnxruntime-node and
        # not readable at all from onnxruntime-web, and the probabilities are the
        # whole point of consuming this from posture.js - the label output alone
        # cannot feed tracker.js's confidence-weighted vote.
        onx = to_onnx(fitted[best], X_tr[:1], target_opset=12, options={"zipmap": False})
        out.write_bytes(onx.SerializeToString())
        # The class order is the label encoder's, not CLASSES', and a consumer
        # that guesses it silently swaps two postures. Ship it beside the model.
        meta = out.with_suffix(".json")
        meta.write_text(
            json.dumps(
                {
                    "classes": list(fitted[best].classes_),
                    "model": best,
                    "trained_on": args.train_on or "mixed",
                    "tier_a_only": args.tier_a,
                    "features": "17 joints as box-relative x,y,confidence, then box aspect",
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {out}  ({out.stat().st_size // 1024} KB)")
        print(f"wrote {meta}  (class order - do not guess it)")
        print("NOT wired into inference.js - posture.js still ships. Compare on")
        print("your own footage before swapping anything.")
    except Exception as exc:  # noqa: BLE001 - the export is optional
        print(f"\nONNX export skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
