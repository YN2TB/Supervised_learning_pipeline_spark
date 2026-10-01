#!/usr/bin/env python
"""Rebuild an experiment's tournament_results.json from MLflow.

The latest finished run of every arm (run name model__arm) in the experiment,
in the format mllib_pipeline.checkpoint writes. Needed when a file was
overwritten, as a two-pass run did before checkpoint() merged.

    python scripts/results_from_mlflow.py flight-delay-final-a [more experiments]
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow.tracking import MlflowClient  # noqa: E402

from spark_session import TRACKING_URI, results_dir  # noqa: E402

KEEP = ("mae", "rmse", "r2", "r2_arrival", "median_ae", "areaUnderROC", "areaUnderPR",
        "accuracy", "f1", "precision_late", "recall_late", "train_seconds")


def rebuild(experiment: str) -> dict:
    client = MlflowClient(TRACKING_URI)
    exp = client.get_experiment_by_name(experiment)
    runs = client.search_runs([exp.experiment_id], max_results=1000,
                              order_by=["attributes.start_time ASC"])
    out = {}
    for run in runs:
        p, m = run.data.params, run.data.metrics
        if run.info.status != "FINISHED" or "arm" not in p or "__" not in (run.info.run_name or ""):
            continue
        entry = {"arm": p["arm"], "model": p.get("model", run.info.run_name.split("__")[0]),
                 "task": p.get("task")}
        entry.update({k: m[k] for k in m if k in KEEP or k.startswith(("recall_at_", "precision_at_"))})
        entry.update({k: _num(v) for k, v in p.items() if k.startswith("best_") and "." not in k})
        out[run.info.run_name] = entry          # later runs overwrite earlier ones
    return out


def _num(v: str):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


if __name__ == "__main__":
    for name in sys.argv[1:]:
        res = rebuild(name)
        path = os.path.join(results_dir(name), "tournament_results.json")
        with open(path, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"{name}: {len(res)} arms -> {path}")
