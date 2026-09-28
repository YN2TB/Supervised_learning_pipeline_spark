#!/usr/bin/env python
"""Side-by-side metrics of every finished arm across experiments sharing a prefix.

Read-only on the MLflow SQLite store; no Spark. Used locally and at the end of the
Kaggle notebook.

    python scripts/compare_experiments.py rot1-predeparture_
    python scripts/compare_experiments.py kg- --db /kaggle/working/repo/mlflow.db --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sqlite3

METRICS = ("r2", "rmse", "mae", "median_ae", "areaUnderROC", "areaUnderPR",
           "recall_at_10pct", "precision_at_10pct", "precision_late", "recall_late", "f1")


def collect(db: str, prefix: str) -> tuple[list[str], dict]:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    exps = con.execute("select experiment_id, name from experiments where name like ? "
                       "order by experiment_id", (prefix + "%",)).fetchall()
    rows: dict = {}
    for eid, ename in exps:
        # newest finished run per name wins
        for rid, rname in con.execute("select run_uuid, name from runs where experiment_id=? "
                                      "and status='FINISHED' order by start_time", (eid,)):
            if "__" not in rname:
                continue
            m = dict(con.execute("select key, value from metrics where run_uuid=?", (rid,)).fetchall())
            rows[(ename, rname)] = m
    return [e[1] for e in exps], rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prefix")
    ap.add_argument("--db", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "mlflow.db"))
    ap.add_argument("--csv", default=None, help="also write the long table here")
    args = ap.parse_args()

    exps, rows = collect(args.db, args.prefix)
    if not rows:
        print(f"no finished runs in experiments starting with {args.prefix!r}")
        return
    runs = sorted({r for _, r in rows})
    short = [e[len(args.prefix):] or e for e in exps]
    width = max(10, *(len(s) for s in short)) + 2
    for metric in METRICS:
        have = [r for r in runs if any(metric in rows.get((e, r), {}) for e in exps)]
        if not have:
            continue
        print(f"\n{metric}")
        print(f"{'run':<42}" + "".join(f"{s[:width - 2]:>{width}}" for s in short))
        for r in have:
            cells = [rows.get((e, r), {}).get(metric) for e in exps]
            print(f"{r:<42}" + "".join(f"{c:>{width}.4f}" if c is not None else f"{'-':>{width}}"
                                        for c in cells))
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["experiment", "run", *METRICS])
            for (e, r), m in sorted(rows.items()):
                w.writerow([e, r, *[m.get(k, "") for k in METRICS]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
