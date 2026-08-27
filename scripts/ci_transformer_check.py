#!/usr/bin/env python
"""CI gate: custom transformers, on synthetic data.

scripts/test_transformers.py is the real gate, but it reads
data/parquet/flights_curated, which is gitignored (the Kaggle sources are
592 MB and never enter the repo). So it cannot run on a clean checkout.

This script covers the part that actually breaks silently - Estimator/Model
leak-safety and the save/load round-trip - against a DataFrame it builds
itself. It is deliberately small: the point is to catch a regression in
custom_transformers.py on push, not to reproduce the full validation.

Run locally the same way CI does:

    python scripts/ci_transformer_check.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((ok, name, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    return ok


def synthetic(spark, n: int = 600):
    """A frame with the columns the transformers expect.

    Includes a deliberate outlier tail (every 50th row) so the IQR fences are
    non-degenerate, and a rare category so unseen-key fallback is exercised.
    """
    from pyspark.sql import functions as F

    df = spark.range(n).select(
        F.col("id"),
        # Real airport coordinates so the haversine has a known answer.
        F.lit(47.45).alias("ORIGIN_LAT"),
        F.lit(-122.31).alias("ORIGIN_LON"),
        F.lit(28.43).alias("DEST_LAT"),
        F.lit(-81.31).alias("DEST_LON"),
        F.lit(2554.0).alias("DISTANCE"),
        F.lit(330.0).alias("SCHEDULED_TIME"),
        (F.when(F.col("id") % 50 == 0, F.lit(900.0))
          .otherwise((F.col("id") % 37).cast("double") - 12.0)
         ).alias("DEPARTURE_DELAY"),
        F.concat(F.lit("AP"), (F.col("id") % 8).cast("string")).alias("ORIGIN_AIRPORT"),
    )
    return df.withColumn("label", F.col("DEPARTURE_DELAY") * 0.9 + 3.0)


def main() -> int:
    from spark_session import build_spark

    # 2 cores and few shuffle partitions: CI runners are small, and the
    # default 64 just creates empty tasks on a 600-row frame.
    spark = build_spark("ci-transformer-check", cores="2", driver_memory="2g",
                        shuffle_partitions=4)
    spark.sparkContext.setLogLevel("ERROR")
    tmp = tempfile.mkdtemp(prefix="ci_tx_")
    try:
        from pyspark.ml import Pipeline, PipelineModel
        from pyspark.sql import functions as F

        from custom_transformers import (
            HaversineTransformer,
            OutlierIQRTruncator,
            SignedLog1pTransformer,
            TargetEncoder,
        )

        df = synthetic(spark)
        train, test = df.randomSplit([0.7, 0.3], seed=7)
        train.cache().count()

        print("== 1. Arrow pandas_udf ==")
        hv = HaversineTransformer()
        row = hv.transform(train).select("GC_DISTANCE_MI", "DISTANCE").first()
        got, published = row[0], row[1]
        # Checked against the route's own published mileage rather than a
        # hardcoded constant: SEA -> MCO ships as 2554 mi, and great-circle
        # should land within a few miles of it.
        check("haversine matches published DISTANCE within 5 mi",
              abs(got - published) < 5.0,
              f"gc={got:.1f} published={published:.1f} delta={abs(got-published):.1f}")

        print("== 2. Estimator/Model leak safety ==")
        iqr = OutlierIQRTruncator(inputCols=["DEPARTURE_DELAY"],
                                  outputCols=["dep_clip"])
        iqr_model = iqr.fit(train)
        bounds_train = iqr_model.bounds()
        # Transforming test data must NOT move the fences: that is exactly the
        # refit-on-test leak the Estimator/Model split exists to prevent.
        iqr_model.transform(test).count()
        check("IQR fences unchanged after transforming test",
              bounds_train == iqr_model.bounds(), f"{bounds_train}")

        hi = bounds_train["DEPARTURE_DELAY"][1]
        clipped = iqr_model.transform(train).agg(
            F.max("dep_clip").alias("mx")).first()["mx"]
        check("outliers actually truncated to the upper fence",
              clipped <= hi + 1e-9, f"max={clipped:.2f} fence={hi:.2f}")

        te = TargetEncoder(inputCols=["ORIGIN_AIRPORT"], outputCols=["origin_te"],
                           labelCol="label", smoothing=20.0)
        te_model = te.fit(train)
        unseen = spark.createDataFrame([("ZZZ_NEVER_SEEN",)], ["ORIGIN_AIRPORT"])
        prior = te_model.getOrDefault(te_model.prior)
        val = te_model.transform(unseen).first()["origin_te"]
        check("unseen category falls back to the global prior",
              abs(val - prior) < 1e-9, f"got {val:.4f} prior {prior:.4f}")

        print("== 3. Serialization round-trip ==")
        pipe = Pipeline(stages=[
            HaversineTransformer(),
            SignedLog1pTransformer(inputCols=["DISTANCE"], outputCols=["log_distance"]),
            OutlierIQRTruncator(inputCols=["DEPARTURE_DELAY"], outputCols=["dep_clip"]),
            TargetEncoder(inputCols=["ORIGIN_AIRPORT"], outputCols=["origin_te"],
                          labelCol="label"),
        ])
        model = pipe.fit(train)
        path = os.path.join(tmp, "pipeline")
        model.write().overwrite().save(path)
        reloaded = PipelineModel.load(path)

        cols = ["GC_DISTANCE_MI", "log_distance", "dep_clip", "origin_te"]
        # Order-independent fingerprint: comparing row-by-row needs a total
        # ordering, and this frame has abundant ties.
        def fingerprint(m):
            agg = m.transform(test).agg(*[F.sum(F.col(c)).alias(c) for c in cols])
            return [round(v, 6) for v in agg.first()]

        check("PipelineModel.save/load produces identical output",
              fingerprint(model) == fingerprint(reloaded))

        print("== 4. Leak policy ==")
        from flight_schema import LEAKY_COLUMNS
        attribution = {"AIR_SYSTEM_DELAY", "SECURITY_DELAY", "AIRLINE_DELAY",
                       "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"}
        missing = attribution - set(LEAKY_COLUMNS)
        check("delay-attribution columns are all in LEAKY_COLUMNS",
              not missing, f"missing: {sorted(missing)}" if missing else "")
        for col in ("ARRIVAL_TIME", "ELAPSED_TIME", "AIR_TIME", "WHEELS_ON", "TAXI_IN"):
            if col not in LEAKY_COLUMNS:
                check(f"{col} is in LEAKY_COLUMNS", False)
                break
        else:
            check("post-departure columns are in LEAKY_COLUMNS", True)

    finally:
        spark.stop()
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(1 for ok, _, _ in results if ok)
    total = len(results)
    print(f"\n{'ALL CHECKS PASSED' if passed == total else 'FAILURES'} ({passed}/{total})")
    if passed != total:
        for ok, name, detail in results:
            if not ok:
                print(f"  FAILED: {name} {detail}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
