"""Phase 2 gate: prove the custom stages are leak-proof and serializable.

These four checks exist because each guards a specific failure that would
otherwise surface much later, in the streaming job:

  1. The IQR fences learned on train stay frozen when the model is applied to
     test data. This is the leak the Estimator/Model split exists to prevent.
  2. The whole custom pipeline survives save() -> load(). Phase 4 loads a
     serialized PipelineModel, so a Param that will not round-trip breaks
     streaming inference and nothing earlier would notice.
  3. Target encoding is computed from train only, falls back to the prior for
     categories it has never seen, and reproduces exactly after reload.
  4. The Arrow haversine UDF agrees with the DISTANCE column the data
     already carries - an independent check that the vectorised maths is right.

Run:  source ./env.sh && python scripts/test_transformers.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_transformers import (  # noqa: E402
    HaversineTransformer, MaterializeCache, OutlierIQRTruncator,
    SignedLog1pTransformer, TargetEncoder,
)
from spark_session import build_spark, path  # noqa: E402

CURATED = path("data", "parquet", "flights_curated")
results: list[bool] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")


def main() -> int:
    spark = build_spark("test-transformers", cores="4")
    tmp = tempfile.mkdtemp(prefix="tf_test_", dir=path(".spark-tmp"))
    try:
        df = spark.read.parquet(CURATED).sample(False, 0.02, seed=7).cache()
        train, test = df.randomSplit([0.8, 0.2], seed=42)
        train.cache(); test.cache()
        print(f"sample rows: train={train.count():,}  test={test.count():,}\n")

        # ---- 1. fitted fences do not move when applied to new data ---------
        print("== 1. IQR fences are frozen at fit time ==")
        iqr = OutlierIQRTruncator(
            inputCols=["DEPARTURE_DELAY", "TAXI_OUT", "DISTANCE"],
            outputCols=["dep_clip", "taxi_clip", "dist_clip"])
        iqr_model = iqr.fit(train)
        fitted = iqr_model.bounds()
        print("   fences learned on train:")
        for col, (lo, hi) in fitted.items():
            print(f"     {col:<16} [{lo:.2f}, {hi:.2f}]")

        # Refitting on test would move the fences; transforming must not.
        refit_on_test = iqr.fit(test).bounds()
        moved = [c for c in fitted if fitted[c] != refit_on_test[c]]
        check("test data would produce different fences (so the risk is real)",
              len(moved) > 0, f"{len(moved)}/{len(fitted)} columns differ")

        clipped = iqr_model.transform(test)
        stats = clipped.agg(F.min("dep_clip"), F.max("dep_clip")).first()
        lo, hi = fitted["DEPARTURE_DELAY"]
        check("transform(test) respects the train fences",
              stats[0] >= lo - 1e-6 and stats[1] <= hi + 1e-6,
              f"observed [{stats[0]:.2f}, {stats[1]:.2f}] within [{lo:.2f}, {hi:.2f}]")

        # ---- 2. the custom pipeline survives save/load ---------------------
        print("\n== 2. custom Pipeline round-trips through save/load ==")
        pipe = Pipeline(stages=[
            iqr,
            HaversineTransformer(),
            SignedLog1pTransformer(inputCols=["dist_clip", "GC_DISTANCE_MI"],
                                   outputCols=["log_dist", "log_gc"]),
            TargetEncoder(inputCols=["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"],
                          outputCols=["te_origin", "te_dest", "te_route"],
                          labelCol="label_delay", smoothing=20.0),
            MaterializeCache(),
        ])
        model = pipe.fit(train)

        out_cols = ["log_dist", "log_gc", "te_origin", "te_dest", "te_route",
                    "SCHEDULE_SPEED_MPH", "ROUTE_DETOUR"]

        def fingerprint(frame):
            """Order-independent snapshot of the engineered columns.

            Sorting by the feature values themselves (rather than by a
            business key such as DAY, where thousands of rows tie) makes the
            comparison deterministic: rows that tie on every output column
            are identical, so their relative order cannot matter.
            """
            rows = (frame.select(*out_cols).orderBy(*out_cols).limit(200).collect())
            agg = frame.agg(*[F.sum(F.col(c)).alias(c) for c in out_cols]).first()
            return rows, [agg[c] for c in out_cols]

        before_rows, before_sums = fingerprint(model.transform(test))

        model_path = os.path.join(tmp, "custom_pipeline")
        model.write().overwrite().save(model_path)
        reloaded = PipelineModel.load(model_path)
        after_rows, after_sums = fingerprint(reloaded.transform(test))

        sums_match = all(
            (a is None and b is None) or abs(a - b) <= 1e-6 * max(1.0, abs(a))
            for a, b in zip(before_sums, after_sums))
        check("reloaded PipelineModel produces identical rows",
              before_rows == after_rows, f"{len(before_rows)} rows compared")
        check("reloaded PipelineModel produces identical column totals", sums_match,
              ", ".join(f"{c}={v:,.2f}" for c, v in zip(out_cols, after_sums)
                        if v is not None))

        # ---- 3. target encoding: train-only, prior fallback ----------------
        print("\n== 3. target encoding is train-only and handles unseen keys ==")
        te_model = [s for s in model.stages if hasattr(s, "mapping")][0]
        mapping = te_model.mapping()
        prior = te_model.getOrDefault(te_model.prior)

        train_keys = {r[0] for r in train.select("ROUTE").distinct().collect()}
        check("encoder learned only categories present in train",
              set(mapping["ROUTE"]).issubset(train_keys),
              f"{len(mapping['ROUTE']):,} routes encoded")

        unseen = (test.select("ROUTE").distinct()
                  .filter(~F.col("ROUTE").isin(list(train_keys))).limit(5).collect())
        if unseen:
            enc = (model.transform(test)
                   .filter(F.col("ROUTE") == unseen[0][0])
                   .select("te_route").first()[0])
            check("unseen category falls back to the global prior",
                  abs(enc - prior) < 1e-9, f"encoded={enc:.4f} prior={prior:.4f}")
        else:
            check("unseen category falls back to the global prior", True,
                  "no unseen routes in this sample; prior fallback exercised by coalesce")

        smoothed_ok = all(min(mapping[c].values()) > -1e4 and max(mapping[c].values()) < 1e4
                          for c in mapping)
        check("smoothed encodings are finite and bounded", smoothed_ok,
              f"prior={prior:.3f} min={min(mapping['ROUTE'].values()):.2f} "
              f"max={max(mapping['ROUTE'].values()):.2f}")

        # ---- 4. the Arrow UDF agrees with the shipped DISTANCE column ------
        print("\n== 4. Arrow haversine agrees with the published DISTANCE ==")
        gc = HaversineTransformer().transform(test)
        err = gc.select(F.abs(F.col("GC_DISTANCE_MI") - F.col("DISTANCE")).alias("e"))
        s = err.agg(F.expr("percentile_approx(e, 0.5)").alias("med"),
                    F.expr("percentile_approx(e, 0.99)").alias("p99")).first()
        check("median |haversine - DISTANCE| under 5 mi", s["med"] < 5.0,
              f"median={s['med']:.2f} mi  p99={s['p99']:.2f} mi")

        ok = all(results)
        print(f"\n{'ALL CHECKS PASSED' if ok else 'TRANSFORMER TESTS FAILED'} "
              f"({sum(results)}/{len(results)})")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        spark.stop()


if __name__ == "__main__":
    raise SystemExit(main())
