#!/usr/bin/env python
"""Training curves for the final run, without retraining the tournament.

GBT: the logged no-PCA model of each task is loaded from MLflow and
GBT*Model.evaluateEachIteration gives the loss after every boosting iteration,
on a 10% sample of the training days and on the validation days.
Linear models: Spark does not save a training summary with the model, so the
no-PCA LinearRegression (A, B) and LinearSVC (C) are refitted on the training
days with their chosen parameters to read objectiveHistory.

The curves go back into each model's own MLflow run as stepped metrics, and to
docs/benchmarks/flight-delay-final-<task>/training_curves.{json,png}.

    source ./env.sh
    python scripts/training_curves.py          # all three tasks, or e.g. ... C B
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 10g pyspark-shell")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.spark  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from pyspark.ml import Pipeline, PipelineModel  # noqa: E402
from pyspark.ml.classification import LinearSVC  # noqa: E402
from pyspark.ml.regression import LinearRegression  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

import mllib_pipeline as mp  # noqa: E402
from model_explain import BLUE, INK_2, _style  # noqa: E402
from spark_session import TRACKING_URI, build_spark, results_dir  # noqa: E402

ORANGE = "#B4532A"
TASKS = {
    "A": dict(exp="flight-delay-final-a", fs="predeparture_a", label=mp.REG_LABEL,
              gbt="gbt_regressor__nopca", linear="linear_regression__nopca"),
    "C": dict(exp="flight-delay-final-c", fs="predeparture_c", label=mp.CLF_LABEL,
              gbt="gbt_classifier__nopca", linear="linear_svc__nopca"),
    "B": dict(exp="flight-delay-final-b-pad", fs="wheelsoff_gain_pad", label=mp.GAIN_LABEL,
              gbt="gbt_regressor__nopca", linear="linear_regression__nopca"),
}


def latest_run(client, exp_name: str, run_name: str):
    exp = client.get_experiment_by_name(exp_name)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0]


def frames(spark, df, fs: str):
    """Training and validation days exactly as the final run built them."""
    train, val = mp.make_split(df, 1.0, "day", mp.SPLIT_SEED)
    numeric = mp.FEATURE_SETS[fs]["numeric"]
    train = mp.add_context_features(spark, df, train, numeric)
    val = mp.add_context_features(spark, df, val, numeric)
    gain = F.col(mp.REG_LABEL) - F.col("DEPARTURE_DELAY")
    train, val = train.withColumn(mp.GAIN_LABEL, gain), val.withColumn(mp.GAIN_LABEL, gain)
    pos = float(train.agg(F.avg(mp.CLF_LABEL)).first()[0])
    weight = F.when(F.col(mp.CLF_LABEL) == 1, F.lit(0.5 / pos)).otherwise(F.lit(0.5 / (1 - pos)))
    return train.withColumn(mp.WEIGHT_COL, weight), val.withColumn(mp.WEIGHT_COL, weight)


def _value(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def gbt_curves(run, train, val, task: str, label: str) -> dict:
    model = mlflow.spark.load_model(f"runs:/{run.info.run_id}/pipeline_model")
    feats, gbt = PipelineModel(stages=model.stages[:-1]), model.stages[-1]
    keep = ["features", label, mp.WEIGHT_COL]
    tr = feats.transform(mp.hash_sample(train, 0.10, 99)).select(*keep).cache()
    va = feats.transform(val).select(*keep).cache()
    if task == "C":
        # Spark's log loss, 2 * log(1 + exp(-2yF)), weighted by class like training.
        curve_tr, curve_va = gbt.evaluateEachIteration(tr), gbt.evaluateEachIteration(va)
        metric = "log loss (weighted)"
    else:
        # Squared error per iteration, reported as RMSE in minutes.
        curve_tr = [math.sqrt(x) for x in gbt.evaluateEachIteration(tr, "squared")]
        curve_va = [math.sqrt(x) for x in gbt.evaluateEachIteration(va, "squared")]
        metric = "RMSE (min)"
    tr.unpersist(), va.unpersist()
    best = min(range(len(curve_va)), key=curve_va.__getitem__)
    return dict(metric=metric, train=curve_tr, validation=curve_va,
                best_iteration=best + 1, iterations=len(curve_va))


def linear_curve(run, train, task: str, label: str, fs: str) -> dict:
    est = (LinearSVC(featuresCol="features", labelCol=label, maxIter=30, weightCol=mp.WEIGHT_COL)
           if task == "C" else
           LinearRegression(featuresCol="features", labelCol=label, maxIter=50))
    for k, v in run.data.params.items():
        name = k[len("best_"):]
        if k.startswith("best_") and "." not in name and est.hasParam(name):
            est._set(**{name: _value(v)})
    stages, _ = mp.build_feature_stages(label, False, 0, feature_set=fs)
    fitted = Pipeline(stages=stages + [est]).fit(train).stages[-1]
    s = fitted.summary() if callable(fitted.summary) else fitted.summary   # LinearSVC: a method
    hist = list(s.objectiveHistory)
    solver = ("OWL-QN / L-BFGS" if task == "C" or est.getElasticNetParam() > 0
              or est.getSolver() == "l-bfgs" else "normal equation (closed form)")
    return dict(model=type(est).__name__, objective=hist, iterations=s.totalIterations,
                solver=solver, params={p.name: v for p, v in est.extractParamMap().items()
                                       if p.name in ("regParam", "elasticNetParam", "loss",
                                                     "maxIter")})


def log_steps(client, run_id: str, key: str, values: list) -> None:
    for i, v in enumerate(values, start=1):
        client.log_metric(run_id, key, float(v), step=i)


def plot(task: str, g: dict, lin: dict, path: str) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.2),
                                 gridspec_kw={"width_ratios": [1.6, 1]})
    x = range(1, g["iterations"] + 1)
    a1.plot(x, g["train"], color=BLUE, linewidth=2)
    a1.plot(x, g["validation"], color=ORANGE, linewidth=2)
    b = g["best_iteration"]
    a1.scatter([b], [g["validation"][b - 1]], color=ORANGE, s=40, zorder=3)
    a1.annotate(f"validation minimum, iteration {b}", (b, g["validation"][b - 1]),
                textcoords="offset points", xytext=(-6, 22), ha="right", fontsize=8.5, color=INK_2,
                arrowprops=dict(arrowstyle="-", color=INK_2, linewidth=0.8))
    for series, col, name in ((g["train"], BLUE, "train (10% sample)"),
                              (g["validation"], ORANGE, "validation days")):
        a1.annotate(name, (g["iterations"], series[-1]), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8.5, color=col)
    a1.set_xlim(0, g["iterations"] * 1.22)
    _style(a1, "boosting iteration", g["metric"], f"Task {task}: GBT loss by iteration")
    h = lin["objective"]
    if len(h) > 1:
        a2.plot(range(len(h)), h, color=INK_2, linewidth=2, marker="o", markersize=3)
        _style(a2, "optimizer iteration", "training objective",
               f"{lin['model']}: objective ({lin['solver']})")
    else:
        a2.axis("off")
        a2.text(0.02, 0.6, f"{lin['model']}\nsolved by the {lin['solver']}:\n"
                "one step, no iteration curve", fontsize=10, color=INK_2, va="top")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    spark = build_spark("training_curves")
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(TRACKING_URI)
    df = spark.read.parquet(mp.CURATED)
    wanted = sys.argv[1:] or list(TASKS)          # e.g. "C B" to redo two tasks
    for task, t in ((k, v) for k, v in TASKS.items() if k in wanted):
        train, val = frames(spark, df, t["fs"])
        train, val = train.cache(), val.cache()
        g_run, l_run = latest_run(client, t["exp"], t["gbt"]), latest_run(client, t["exp"], t["linear"])
        g = gbt_curves(g_run, train, val, task, t["label"])
        print(f"CURVE {task} GBT {g['metric']}: train {g['train'][0]:.4f} -> {g['train'][-1]:.4f}, "
              f"validation {g['validation'][0]:.4f} -> {g['validation'][-1]:.4f}, "
              f"minimum at iteration {g['best_iteration']} of {g['iterations']}")
        lin = linear_curve(l_run, train, task, t["label"], t["fs"])
        print(f"CURVE {task} {lin['model']}: {lin['iterations']} iterations, {lin['solver']}, "
              f"objective {lin['objective'][0]:.5f} -> {lin['objective'][-1]:.5f}")
        log_steps(client, g_run.info.run_id, "curve_train_" + ("logloss" if task == "C" else "rmse"), g["train"])
        log_steps(client, g_run.info.run_id, "curve_val_" + ("logloss" if task == "C" else "rmse"), g["validation"])
        log_steps(client, l_run.info.run_id, "curve_objective", lin["objective"])
        out = results_dir(t["exp"])
        with open(os.path.join(out, "training_curves.json"), "w") as fh:
            json.dump({"gbt": dict(g, run_id=g_run.info.run_id),
                       "linear": dict(lin, run_id=l_run.info.run_id)}, fh, indent=1)
        png = os.path.join(out, "training_curves.png")
        plot(task, g, lin, png)
        client.log_artifact(g_run.info.run_id, png, "plots")
        train.unpersist(), val.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
