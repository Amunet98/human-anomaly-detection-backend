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
"""

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


def load(path: Path, drop: set[str] | None = None):
    rows = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if not r["gt"]:
            continue  # unlabelled: the pose model found someone the annotator did not
        if drop and r["gt"] == "fall" and r["file"] in drop:
            continue  # label disputed by the geometry - see disputed-fall-labels.txt
        rows.append(r)
    return rows


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
    X, y, groups, base = [], [], [], []
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
    return np.array(X, dtype=np.float32), np.array(y), np.array(groups), np.array(base)


def score(name, truth, pred):
    acc = (truth == pred).mean()
    print(f"\n--- {name} — accuracy {acc:.3f} ---")
    print(classification_report(truth, pred, labels=CLASSES, zero_division=0, digits=3))
    return acc


def main(tier_a: bool = "--tier-a" in sys.argv):
    c23 = ROOT / "corpus-2023" / "keypoints.jsonl"
    pol = ROOT / "corpus-polar" / "keypoints.jsonl"
    for p in (c23, pol):
        if not p.exists():
            sys.exit(f"missing {p}\nRun feature-dump.mjs --keypoints first.")

    disputed = set((ROOT / "corpus-2023" / "disputed-fall-labels.txt").read_text().split())
    rows = load(c23, drop=disputed) + load(pol)
    # --tier-a restricts to detections with the full leg chain. Worth running
    # both ways: tier A is where the geometric gates can actually compete (the
    # squat gate cannot fire below it), so it is the fair comparison, while the
    # unrestricted run is closer to what a live camera delivers.
    if tier_a:
        rows = [r for r in rows if r["tier"] == "A"]
        print("--tier-a: full leg chain only")
    X, y, groups, base = featurise(rows)
    print(f"rows {len(y)}  features {X.shape[1]}  images {len(set(groups))}")
    print("classes:", dict(sorted(Counter(y).items())))

    # Grouped split: an image is entirely in train or entirely in test.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED).split(X, y, groups))
    print(f"\ntrain {len(tr)} rows / test {len(te)} rows "
          f"({len(set(groups[tr]))} / {len(set(groups[te]))} images, no overlap)")

    # The bar: what ships today, on this exact test set.
    score("BASELINE — posture.js geometric gates", y[te], base[te])

    models = {
        "MLP (128,64)": make_pipeline(
            StandardScaler(), MLPClassifier((128, 64), max_iter=600, random_state=SEED)
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=SEED),
    }
    fitted = {}
    for name, m in models.items():
        m.fit(X[tr], y[tr])
        fitted[name] = m
        score(name, y[te], m.predict(X[te]))

    # Ablation. The confidences encode which joints the pose model found, and
    # the two sources differ in occlusion - so confidence alone partly identifies
    # the dataset, which a model could ride instead of learning posture. Measured
    # 2026-08-12: confidences alone reach 0.621 against 0.357 chance, so the leak
    # is real. What clears the model is that coordinates alone score *identically*
    # to the full feature set - it does not use the leak even though it is there.
    #
    # If a future change makes "everything" beat "coordinates only" by more than
    # noise, that gap is the shortcut being exploited, not new signal.
    conf_idx = [i for i in range(2, 51, 3)]
    coord_idx = [i for i in range(51) if i % 3 != 2] + [51]
    print("\n--- ABLATION — which inputs carry the signal ---")
    for label, idx in (
        ("coordinates only (no confidence)", coord_idx),
        ("confidences only (no coordinates)", conf_idx),
        ("everything", list(range(X.shape[1]))),
    ):
        a = GradientBoostingClassifier(random_state=SEED).fit(X[tr][:, idx], y[tr])
        print(f"  {label:34} {(a.predict(X[te][:, idx]) == y[te]).mean():.3f}")
    print(f"  {'chance (majority class)':34} {max(Counter(y[te]).values())/len(te):.3f}")

    # False alarms: the POLAR rows contain no falls, so any `fall` is wrong.
    pol_files = {r["file"] for r in load(pol)}
    mask = np.array([g in pol_files for g in groups[te]])
    if mask.any():
        print(f"\n--- FALSE ALARMS on {mask.sum()} POLAR test rows (no falls exist here) ---")
        print(f"  {'baseline (posture.js)':24} {(base[te][mask] == 'fall').mean():.3f}")
        for name, m in fitted.items():
            print(f"  {name:24} {(m.predict(X[te][mask]) == 'fall').mean():.3f}")
        print("  Lower is better. A model that improves posture accuracy while")
        print("  raising this is a worse system - see MODEL_CARD, 'False-alarm rate'.")

    best = max(fitted, key=lambda n: (fitted[n].predict(X[te]) == y[te]).mean())
    print(f"\nbest by accuracy: {best}")
    print("Confusion (rows = truth, cols = predicted), order:", CLASSES)
    print(confusion_matrix(y[te], fitted[best].predict(X[te]), labels=CLASSES))

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

        onx = to_onnx(fitted[best], X[:1], target_opset=12)
        out.write_bytes(onx.SerializeToString())
        print(f"\nwrote {out}  ({out.stat().st_size // 1024} KB)")
        print("NOT wired into inference.js - posture.js still ships. Compare on")
        print("your own footage before swapping anything.")
    except Exception as exc:  # noqa: BLE001 - the export is optional
        print(f"\nONNX export skipped: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
