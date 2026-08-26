"""Phase 1: turn the raw CSV feed into a curated, leak-free Parquet table.

Two steps, run in order:

    python data_prep.py --step raw      # CSV -> data/parquet/flights_raw
    python data_prep.py --step curate   # -> data/parquet/flights_curated

Between them, ``scripts/build_airport_code_map.py`` recovers the airport-code
mapping that the October slice of the feed needs (see that script for why).
"""
from __future__ import annotations

import argparse
import os
import sys

from pyspark.sql import DataFrame, functions as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flight_schema import (  # noqa: E402
    ALLOWED_RAW_FEATURES, FLIGHTS_SCHEMA, LEAKY_COLUMNS, SEVERE_DELAY_THRESHOLD,
)
from spark_session import build_spark, path  # noqa: E402

RAW_PARQUET = path("data", "parquet", "flights_raw")
CURATED_PARQUET = path("data", "parquet", "flights_curated")
CODE_MAP_CSV = path("data", "dot_to_iata.csv")


def hhmm_to_minutes(col: str):
    """'0005' -> 5, '1430' -> 870. The feed uses '2400' for midnight."""
    v = F.col(col).cast("int")
    return F.when(v.isNull(), None).otherwise(((v / 100).cast("int") % 24) * 60 + (v % 100))


# ---------------------------------------------------------------------------
# Step 1: CSV -> Parquet
# ---------------------------------------------------------------------------
def step_raw(spark) -> None:
    print(f"reading {path('data', 'flights.csv')}")
    df = spark.read.csv(path("data", "flights.csv"), header=True,
                        schema=FLIGHTS_SCHEMA, mode="PERMISSIVE")

    n = df.count()
    print(f"  rows read: {n:,}")

    (df.repartition("MONTH")
       .write.mode("overwrite")
       .partitionBy("MONTH")
       .parquet(RAW_PARQUET))
    print(f"  wrote {RAW_PARQUET}")

    back = spark.read.parquet(RAW_PARQUET).count()
    print(f"  rows in parquet: {back:,}   {'OK' if back == n else 'MISMATCH'}")


# ---------------------------------------------------------------------------
# Step 2: repair, join, filter, label
# ---------------------------------------------------------------------------
def _repair_airport_codes(spark, df: DataFrame) -> DataFrame:
    """Normalise the October numeric DOT ids back to IATA codes.

    The feed changes key encoding for one month: October rows carry numeric
    DOT airport ids (14747) where every other month carries IATA codes (SEA).
    Left alone, the airports.csv join drops those rows silently.
    """
    if not os.path.exists(CODE_MAP_CSV):
        raise SystemExit(
            f"missing {CODE_MAP_CSV}\n"
            "run: python scripts/build_airport_code_map.py"
        )
    code_map = spark.read.csv(CODE_MAP_CSV, header=True)
    mapping = F.broadcast(code_map.select(
        F.col("dot_code").alias("_dot"), F.col("iata_code").alias("_iata")))

    for col in ("ORIGIN_AIRPORT", "DESTINATION_AIRPORT"):
        df = (df.join(mapping, df[col] == F.col("_dot"), "left")
                .withColumn(col, F.coalesce(F.col("_iata"), F.col(col)))
                .drop("_dot", "_iata"))
    return df


def step_curate(spark) -> None:
    df = spark.read.parquet(RAW_PARQUET)
    n_start = df.count()
    print(f"raw rows: {n_start:,}")

    df = _repair_airport_codes(spark, df)
    still_numeric = df.filter(
        F.col("ORIGIN_AIRPORT").rlike("^[0-9]+$") |
        F.col("DESTINATION_AIRPORT").rlike("^[0-9]+$")).count()
    print(f"  rows still holding a numeric airport code after repair: {still_numeric:,}")

    # Arrival delay is undefined for flights that never arrived.
    df = df.filter((F.col("CANCELLED") == 0) & (F.col("DIVERTED") == 0))
    df = df.filter(F.col("ARRIVAL_DELAY").isNotNull()
                   & F.col("DEPARTURE_DELAY").isNotNull()
                   & F.col("TAXI_OUT").isNotNull())
    print(f"  after dropping cancelled/diverted/null-target: {df.count():,}")

    # --- reference joins (both tiny -> broadcast) ---
    airports = spark.read.csv(path("data", "airports.csv"), header=True, inferSchema=True)
    airlines = spark.read.csv(path("data", "airlines.csv"), header=True, inferSchema=True)

    org = F.broadcast(airports.select(
        F.col("IATA_CODE").alias("_o"),
        F.col("LATITUDE").cast("double").alias("ORIGIN_LAT"),
        F.col("LONGITUDE").cast("double").alias("ORIGIN_LON")))
    dst = F.broadcast(airports.select(
        F.col("IATA_CODE").alias("_d"),
        F.col("LATITUDE").cast("double").alias("DEST_LAT"),
        F.col("LONGITUDE").cast("double").alias("DEST_LON")))
    air = F.broadcast(airlines.select(
        F.col("IATA_CODE").alias("_a"), F.col("AIRLINE").alias("AIRLINE_NAME")))

    df = (df.join(org, df.ORIGIN_AIRPORT == F.col("_o"), "left").drop("_o")
            .join(dst, df.DESTINATION_AIRPORT == F.col("_d"), "left").drop("_d")
            .join(air, df.AIRLINE == F.col("_a"), "left").drop("_a"))

    # --- derived features, all known at wheels-off ---
    df = (df
          .withColumn("SCHED_DEP_MIN", hhmm_to_minutes("SCHEDULED_DEPARTURE"))
          .withColumn("SCHED_ARR_MIN", hhmm_to_minutes("SCHEDULED_ARRIVAL"))
          .withColumn("WHEELS_OFF_MIN", hhmm_to_minutes("WHEELS_OFF"))
          .withColumn("DEP_HOUR", (F.col("SCHED_DEP_MIN") / 60).cast("int"))
          .withColumn("IS_WEEKEND", F.col("DAY_OF_WEEK").isin(6, 7).cast("int"))
          .withColumn("ROUTE", F.concat_ws("-", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT"))
          # labels
          .withColumn("label_delay", F.col("ARRIVAL_DELAY").cast("double"))
          .withColumn("label_severe",
                      (F.col("ARRIVAL_DELAY") > SEVERE_DELAY_THRESHOLD).cast("double")))

    # --- enforce the leakage policy in code, not just in prose ---
    df = df.drop(*[c for c in LEAKY_COLUMNS if c in df.columns])
    leaked = [c for c in LEAKY_COLUMNS if c in df.columns]
    assert not leaked, f"leak columns survived: {leaked}"

    # The code repair resolves 302 of 307 encoded airport ids; a handful of
    # small airports have no confident match and a few reference airports
    # ship without coordinates. Those rows cannot feed the haversine stage,
    # and there are too few to be worth imputing.
    n_pre_coord = df.count()
    df = df.filter(F.col("ORIGIN_LAT").isNotNull() & F.col("DEST_LAT").isNotNull())

    df = df.repartition(64).cache()
    n_final = df.count()
    print(f"  dropped for missing airport coordinates: {n_pre_coord - n_final:,}")

    (df.write.mode("overwrite").parquet(CURATED_PARQUET))
    print(f"\ncurated rows: {n_final:,}  ({n_final / n_start:.1%} of raw)")
    print(f"wrote {CURATED_PARQUET}")
    print(f"severe-delay rate: {df.agg(F.avg('label_severe')).first()[0]:.4f}")
    print("\ncolumns retained:")
    print("  " + ", ".join(df.columns))
    print("\nfeature columns allowed by policy that are present:")
    print("  " + ", ".join(c for c in ALLOWED_RAW_FEATURES if c in df.columns))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["raw", "curate"], required=True)
    args = ap.parse_args()

    spark = build_spark(f"data_prep-{args.step}")
    try:
        {"raw": step_raw, "curate": step_curate}[args.step](spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
