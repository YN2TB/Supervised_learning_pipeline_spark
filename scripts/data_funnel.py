#!/usr/bin/env python
"""Where the rows go between the raw feed and the modelling table.

Recounts data_prep.step_curate's filters one at a time on flights_raw, with
cancelled and diverted counted separately, so the report can state the scope:
the model answers "if this flight operates, how late does it arrive".
Read-only; writes docs/benchmarks/data_funnel.json.

    python scripts/data_funnel.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pyspark.sql import functions as F  # noqa: E402

from data_prep import RAW_PARQUET, _repair_airport_codes  # noqa: E402
from mllib_pipeline import CURATED  # noqa: E402
from spark_session import build_spark, path  # noqa: E402


def main() -> None:
    spark = build_spark("data-funnel", cores="4", driver_memory="3g")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        raw = spark.read.parquet(RAW_PARQUET)
        c = raw.agg(
            F.count(F.lit(1)).alias("raw"),
            F.sum(F.col("CANCELLED")).alias("cancelled"),
            F.sum(F.when((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 1), 1)
                  .otherwise(0)).alias("diverted"),
            F.sum(F.when((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0)
                         & (F.col("ARRIVAL_DELAY").isNull()
                            | F.col("DEPARTURE_DELAY").isNull()
                            | F.col("TAXI_OUT").isNull()), 1)
                  .otherwise(0)).alias("null_target"),
            F.sum(F.when(F.col("MONTH") == 10, 1).otherwise(0)).alias("october"),
        ).first().asDict()
        flown = (_repair_airport_codes(spark, raw)
                 .filter((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0))
                 .filter(F.col("ARRIVAL_DELAY").isNotNull()
                         & F.col("DEPARTURE_DELAY").isNotNull()
                         & F.col("TAXI_OUT").isNotNull()))
        n_flown = flown.count()
        n_curated = spark.read.parquet(CURATED).count()

        steps = [
            ("raw rows in flights.csv", c["raw"], None),
            ("cancelled (never flew, no arrival delay)", -c["cancelled"], "out of scope"),
            ("diverted (landed elsewhere)", -c["diverted"], "out of scope"),
            ("operated but a delay or taxi time missing", -c["null_target"], "data gap"),
            ("operated, with a measured arrival delay", n_flown, None),
            ("airport without coordinates (5 unmapped October codes and a few "
             "reference airports)", -(n_flown - n_curated), "data gap"),
            ("modelling table (flights_curated)", n_curated, None),
        ]
        print(f"{'step':<78}{'rows':>12}{'share of raw':>14}")
        for name, n, _ in steps:
            print(f"{name:<78}{n:>12,}{abs(n) / c['raw']:>14.2%}")
        out = {"steps": [{"step": s, "rows": n, "kind": k} for s, n, k in steps],
               "october_rows_raw": c["october"]}
        dst = path("docs", "benchmarks", "data_funnel.json")
        with open(dst, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {dst}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
