#!/usr/bin/env python
"""Copy the runs of the Kaggle trial databases into the main MLflow store.

Each Kaggle kernel logs to its own mlflow.db, downloaded to
experiments/<folder>/results/mlflow.db. This copies every finished top-level run
(params, full metric history, tags, times, status) into the main store under the
experiment "<folder>/<experiment>", so the Kaggle trials sit next to the local
ones in one MLflow UI. Several kernels share experiment names (kg-v2 in most of
them, with different splits and seeds), hence the folder prefix.

Artifacts are not copied: they lived in the kernel's mlruns, deleted when the
kernel ended. The plots are in experiments/<folder>/results/<experiment>/, and
each copied run carries that path in its "results_dir" tag.

Idempotent: an experiment already present in the main store is skipped. The
source databases are opened read-only. Back up mlflow.db before running.

    python scripts/merge_kaggle_runs.py            # dry run: what would be copied
    python scripts/merge_kaggle_runs.py --apply
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow.entities import Metric, Param, RunTag  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

from spark_session import TRACKING_URI, path  # noqa: E402

BATCH = 900          # MLflow caps a log_batch at 1000 entries


def source_runs(db: str):
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    exps = con.execute("select experiment_id, name from experiments "
                       "where lifecycle_stage='active' and name != 'Default'").fetchall()
    for eid, ename in exps:
        runs = con.execute("select run_uuid, name, status, start_time, end_time from runs "
                           "where experiment_id=? and lifecycle_stage='active' "
                           "order by start_time", (eid,)).fetchall()
        out = []
        for rid, rname, status, t0, t1 in runs:
            tags = dict(con.execute("select key, value from tags where run_uuid=?", (rid,)))
            if "mlflow.parentRunId" in tags or "__" not in (rname or ""):
                continue            # autolog's CV children; only the arms and baselines
            params = con.execute("select key, value from params where run_uuid=?",
                                 (rid,)).fetchall()
            metrics = con.execute("select key, value, timestamp, step from metrics "
                                  "where run_uuid=? and is_nan=0", (rid,)).fetchall()
            out.append(dict(id=rid, name=rname, status=status, t0=t0, t1=t1,
                            tags=tags, params=params, metrics=metrics))
        yield ename, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write; without it, a dry run")
    args = ap.parse_args()

    client = MlflowClient(TRACKING_URI)
    existing = {e.name for e in client.search_experiments()}
    total = 0
    for db in sorted(glob.glob(path("experiments", "kaggle-*", "results", "mlflow.db"))):
        folder = os.path.basename(os.path.dirname(os.path.dirname(db)))
        for ename, runs in source_runs(db):
            target = f"{folder}/{ename}"
            if target in existing:
                print(f"  skip {target} (already in the main store)")
                continue
            print(f"  {'copy' if args.apply else 'would copy'} {target}: {len(runs)} runs")
            total += len(runs)
            if not args.apply:
                continue
            eid = client.create_experiment(target, tags={
                "source_db": os.path.relpath(db, path()).replace("\\", "/"),
                "mlflow.note.content": f"Kaggle trial copied from {folder}; see experiments/LOG.md"})
            results_dir = os.path.relpath(os.path.join(os.path.dirname(db), ename),
                                          path()).replace("\\", "/")
            for r in runs:
                keep = {k: v for k, v in r["tags"].items()
                        if not k.startswith("mlflow.") or k in ("mlflow.note.content",)}
                keep.update({"source_run_id": r["id"], "results_dir": results_dir})
                run = client.create_run(eid, start_time=r["t0"], run_name=r["name"],
                                        tags=keep)
                rid = run.info.run_id
                params = [Param(k, v) for k, v in r["params"]]
                metrics = [Metric(k, v, ts, st) for k, v, ts, st in r["metrics"]]
                for i in range(0, max(len(params), 1), 100):
                    client.log_batch(rid, params=params[i:i + 100])
                for i in range(0, len(metrics), BATCH):
                    client.log_batch(rid, metrics=metrics[i:i + BATCH])
                client.set_terminated(rid, status=r["status"], end_time=r["t1"])
    print(f"{'copied' if args.apply else 'would copy'} {total} runs")


if __name__ == "__main__":
    main()
