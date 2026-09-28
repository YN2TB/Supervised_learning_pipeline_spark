#!/usr/bin/env python
"""Operational reading of a finished run: no Spark, reads the saved curve files.

For each classifier: flag the top 5 / 10 / 20% of flights by score, how many of
the late ones does that catch (recall) and how many flags are right (precision);
and the late class's own precision and recall at the model's threshold, from the
confusion matrix. For each regressor with an error-band file: MAE by how late
the flight actually was.

    python scripts/business_metrics.py docs/benchmarks/flight-delay-predeparture
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_explain import CAPTURE_SHARES  # noqa: E402


def capture_from_roc(curve: dict) -> list[dict]:
    """Capture table from a saved ROC curve, for runs older than the capture field.

    Flagging the top share q of flights puts the threshold where
    q = p * tpr + (1 - p) * fpr; interpolated along the saved (thinned) curve.
    """
    p = curve["positive_rate"]
    fpr, tpr = np.asarray(curve["fpr"]), np.asarray(curve["tpr"])
    flagged = p * tpr + (1 - p) * fpr
    out = []
    for q in CAPTURE_SHARES:
        rec = float(np.interp(q, flagged, tpr))
        out.append({"flagged": q, "recall": rec, "precision": p * rec / q})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(args.results_dir, "*_curves.json"))):
        c = json.load(open(f))
        name = os.path.basename(f)[:-len("_curves.json")]
        cap = c.get("capture") or capture_from_roc(c)
        (tn, fp), (fn, tp) = c["confusion"]
        rows.append((name, cap, tp / max(tp + fp, 1), tp / max(tp + fn, 1),
                     (tp + fp) / max(tn + fp + fn + tp, 1), c["positive_rate"]))
    if rows:
        print(f"Classifiers ({rows[0][5]:.1%} of test flights are more than 30 min late)\n")
        head = "".join(f" | flag {int(q * 100)}%: caught / right" for q in CAPTURE_SHARES)
        print(f"| model{head} | at threshold: precision / recall (flags) |")
        print("|---" * (len(CAPTURE_SHARES) + 2) + "|")
        for name, cap, prec, rec, flagged, _ in rows:
            cells = "".join(f" | {r['recall']:.0%} / {r['precision']:.0%}" for r in cap)
            print(f"| {name}{cells} | {prec:.0%} / {rec:.0%} ({flagged:.0%}) |")

    bands = sorted(glob.glob(os.path.join(args.results_dir, "*_error_bands.json")))
    for f in bands:
        b = json.load(open(f))
        name = os.path.basename(f)[:-len("_error_bands.json")]
        print(f"\n{name}: MAE {b['mae']:.1f} min, median {b['median_ae']:.1f} min")
        print("| actual delay (min) | share of flights | MAE (min) | mean bias (min) |")
        print("|---|---|---|---|")
        for r in b["bands"]:
            print(f"| {r['band']} | {r['share']:.1%} | {r['mae']:.1f} | {r['bias']:+.1f} |")
    if not rows and not bands:
        print(f"no *_curves.json or *_error_bands.json in {args.results_dir}")


if __name__ == "__main__":
    main()
