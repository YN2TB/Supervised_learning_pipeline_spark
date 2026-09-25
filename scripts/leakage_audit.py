#!/usr/bin/env python
"""Tests that would expose a leak, run against any feature set.

A high score is not evidence of leakage and a plausible story is not evidence
against it. Each test below has a result that a leak would change:

  overlap     flights present in both train and test. Must be 0.
  score       the full pipeline on the tournament's random split (reference).
  single      a straight line on one legitimate column (DEPARTURE_DELAY; and
              prev_arr_delay when the set has it). If this alone reaches most of
              the score, the score needs no leak to explain it.
  shuffled    the same pipeline trained on labels shuffled within train, scored
              on the real test labels. Any path from a label into the features
              that survives to test time keeps this above 0; without one it is ~0.
  temporal    train on January to September, test on October to December. A
              score that came from neighbouring rows of a random split (same
              day, same aircraft) falls here; a real relationship holds.
  physics     (wheels-off sets) ARRIVAL_DELAY - DEPARTURE_DELAY is by definition
              ELAPSED_TIME - SCHEDULED_TIME. At wheels-off, taxi-out is known and
              air time + taxi-in are not. A predictor that knows the physics and
              guesses the unknown part by its average per route, month and hour
              shows what is achievable; a model far better than it would know
              what it cannot know.

    source ./env.sh && python scripts/leakage_audit.py --feature-set wheelsoff
    source ./env.sh && python scripts/leakage_audit.py --feature-set predeparture_inbound

Read-only on the data and the registry. Writes experiments/leakage-audit/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.ml import Pipeline  # noqa: E402
from pyspark.ml.evaluation import RegressionEvaluator  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.ml.regression import GBTRegressor, LinearRegression  # noqa: E402
from pyspark.sql import Window, functions as F  # noqa: E402

import custom_transformers  # noqa: E402
from mllib_pipeline import (CURATED, FEATURE_SETS, FLIGHT_KEY, REG_LABEL, SPLIT_SEED,  # noqa: E402
                            add_context_features, build_feature_stages, make_split)
from spark_session import build_spark, results_dir  # noqa: E402

OUT = results_dir("leakage-audit")


def estimator(kind: str):
    if kind == "lr":
        return LinearRegression(featuresCol="features", labelCol=REG_LABEL,
                                regParam=0.01, elasticNetParam=0.0, maxIter=50)
    return GBTRegressor(featuresCol="features", labelCol=REG_LABEL,
                        maxIter=40, maxDepth=5, maxBins=64, seed=7)


def metrics(preds) -> dict:
    ev = RegressionEvaluator(labelCol=REG_LABEL, predictionCol="prediction")
    return {m: round(float(ev.setMetricName(m).evaluate(preds)), 4) for m in ("r2", "rmse")}


def fit_score(train, test, feature_set: str, kind: str) -> dict:
    stages, _ = build_feature_stages(REG_LABEL, use_pca=False, pca_k=10,
                                     feature_set=feature_set)
    model = Pipeline(stages=stages + [estimator(kind)]).fit(train)
    out = metrics(model.transform(test))
    custom_transformers.release_caches()
    return out


def shuffle_label(frame):
    """Same rows, labels dealt out at random among them."""
    w_lab = Window.orderBy(F.rand(SPLIT_SEED))
    w_row = Window.orderBy(F.monotonically_increasing_id())
    labels = frame.select(F.col(REG_LABEL).alias("_y")).withColumn("_i", F.row_number().over(w_lab))
    rows = frame.drop(REG_LABEL).withColumn("_i", F.row_number().over(w_row))
    return rows.join(labels, "_i").withColumnRenamed("_y", REG_LABEL).drop("_i")


def physics_reference(spark, train, test) -> dict:
    """Wheels-off physics: arrival = departure delay + taxi-out + (air time +
    taxi-in, unknown, guessed by its train mean per route, month and hour)
    - scheduled block time."""
    from data_prep import _repair_airport_codes, hhmm_to_minutes
    raw = (_repair_airport_codes(spark, spark.read.parquet(
               os.path.join(os.path.dirname(CURATED), "flights_raw")))
           .filter((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0))
           .withColumn("SCHED_DEP_MIN", hhmm_to_minutes("SCHEDULED_DEPARTURE"))
           .select(*FLIGHT_KEY, "AIR_TIME", "TAXI_IN", "ELAPSED_TIME"))
    tr = train.join(raw, FLIGHT_KEY, "inner")
    te = test.join(raw, FLIGHT_KEY, "inner")
    ident = te.agg(F.avg((F.abs((F.col(REG_LABEL) - F.col("DEPARTURE_DELAY"))
                                - (F.col("ELAPSED_TIME") - F.col("SCHEDULED_TIME"))) < 0.5)
                         .cast("double"))).first()[0]
    unknown = F.col("AIR_TIME") + F.col("TAXI_IN")
    fine = tr.groupBy("ROUTE", "MONTH", "DEP_HOUR").agg(F.avg(unknown).alias("_u1"))
    route = tr.groupBy("ROUTE").agg(F.avg(unknown).alias("_u2"))
    overall = tr.agg(F.avg(unknown)).first()[0]
    te = (te.join(fine, ["ROUTE", "MONTH", "DEP_HOUR"], "left").join(route, "ROUTE", "left")
            .withColumn("prediction", F.col("DEPARTURE_DELAY") + F.col("TAXI_OUT")
                        + F.coalesce("_u1", "_u2", F.lit(overall)) - F.col("SCHEDULED_TIME")))
    out = metrics(te.filter(F.col("prediction").isNotNull()))
    out["identity_holds"] = round(float(ident), 5)
    out["test_rows_joined"] = te.count()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="wheelsoff", choices=sorted(FEATURE_SETS))
    ap.add_argument("--sample", type=float, default=0.05,
                    help="fraction of the curated data for the model fits")
    ap.add_argument("--models", default="lr,gbt")
    ap.add_argument("--tests", default="overlap,score,single,shuffled,temporal,physics")
    ap.add_argument("--max-dep-delay", type=float, default=None,
                    help="wheels-off only: audit within the scope departure delay <= N")
    args = ap.parse_args()
    tests = set(args.tests.split(","))
    kinds = args.models.split(",")
    numeric = FEATURE_SETS[args.feature_set]["numeric"]
    os.makedirs(OUT, exist_ok=True)

    spark = build_spark("leakage-audit")
    res = {"feature_set": args.feature_set, "sample": args.sample,
           "max_dep_delay": args.max_dep_delay}
    scope = ((lambda d: d.filter(F.col("DEPARTURE_DELAY") <= args.max_dep_delay))
             if args.max_dep_delay is not None else (lambda d: d))
    tag = args.feature_set + (f"_le{int(args.max_dep_delay)}" if args.max_dep_delay else "")
    t0 = time.time()
    try:
        df = spark.read.parquet(CURATED)
        train, test = make_split(df, args.sample)
        train = scope(add_context_features(spark, df, train, numeric)).cache()
        test = scope(add_context_features(spark, df, test, numeric)).cache()
        res["rows"] = {"train": train.count(), "test": test.count()}

        if "overlap" in tests:
            res["overlap"] = train.select(*FLIGHT_KEY).join(test.select(*FLIGHT_KEY),
                                                            FLIGHT_KEY).count()
            print(f"AUDIT overlap: {res['overlap']} flights in both train and test", flush=True)

        if "score" in tests:
            res["score"] = {k: fit_score(train, test, args.feature_set, k) for k in kinds}
            print(f"AUDIT score (random split): {res['score']}", flush=True)

        if "single" in tests:
            res["single"] = {}
            for col in ["DEPARTURE_DELAY"] + (["prev_arr_delay"] if "prev_arr_delay" in numeric else []):
                pipe = Pipeline(stages=[VectorAssembler(inputCols=[col], outputCol="features",
                                                        handleInvalid="skip"),
                                        estimator("lr")])
                res["single"][col] = metrics(pipe.fit(train).transform(test))
            print(f"AUDIT single column, straight line: {res['single']}", flush=True)

        if "shuffled" in tests:
            shuffled = shuffle_label(train).cache()
            res["shuffled"] = {k: fit_score(shuffled, test, args.feature_set, k) for k in kinds}
            shuffled.unpersist()
            print(f"AUDIT shuffled labels: {res['shuffled']}", flush=True)

        if "temporal" in tests:
            sample = df.sample(False, args.sample, seed=SPLIT_SEED) if args.sample < 1 else df
            early = scope(add_context_features(spark, df, sample.filter("MONTH <= 9"), numeric)).cache()
            late = scope(add_context_features(spark, df, sample.filter("MONTH >= 10"), numeric)).cache()
            res["temporal"] = {k: fit_score(early, late, args.feature_set, k) for k in kinds}
            res["temporal_rows"] = {"train_jan_sep": early.count(), "test_oct_dec": late.count()}
            early.unpersist(); late.unpersist()
            print(f"AUDIT temporal split (Jan-Sep -> Oct-Dec): {res['temporal']}", flush=True)

        if "physics" in tests and "DEPARTURE_DELAY" in numeric:
            full_train, full_test = make_split(df)       # full size: group means need data
            res["physics"] = physics_reference(spark, scope(full_train), scope(full_test))
            print(f"AUDIT physics reference (full split): {res['physics']}", flush=True)
    finally:
        res["seconds"] = round(time.time() - t0)
        with open(os.path.join(OUT, f"audit_{tag}.json"), "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"AUDIT wrote {os.path.join(OUT, f'audit_{tag}.json')}")
        spark.stop()


if __name__ == "__main__":
    main()
