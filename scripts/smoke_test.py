"""Phase 0 gate: prove the Spark runtime actually works before any real work.

Exercises the three things that break Spark on Windows, in the order they bite:
  1. JVM startup            -> JAVA_HOME points at a Spark-supported JDK
  2. Arrow pandas_udf       -> pyarrow <-> JVM Arrow versions agree
  3. PipelineModel save/load -> winutils.exe + hadoop.dll are present and loadable

Run:  source ./env.sh && python scripts/smoke_test.py

The Kaggle CSV is optional here.  When it has not been downloaded yet, this
runtime-only gate uses a deterministic 1,000-row frame with the same two
numeric columns.  Data validation belongs to ``data_prep.py``; making Phase 0
depend on a 592 MB download would hide environment failures behind a missing
file error.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

# Importing this module bootstraps the same local Hadoop/temporary-directory
# settings used by every production entry point.
from spark_session import raw_data_path  # noqa: E402


def _check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    return ok


def main() -> int:
    # Keep the smoke test useful even when the caller forgot to source env.*.
    # build_spark() performs the same interpreter pin for all project scripts.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    local_hadoop = os.path.join(REPO, ".hadoop")
    if os.path.isfile(os.path.join(local_hadoop, "bin", "winutils.exe")):
        os.environ.setdefault("HADOOP_HOME", local_hadoop)
        os.environ["PATH"] = os.path.join(local_hadoop, "bin") + os.pathsep + os.environ["PATH"]
    spark_tmp = os.path.join(REPO, ".spark-tmp")
    os.makedirs(spark_tmp, exist_ok=True)

    print("== environment ==")
    print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
    for var in ("JAVA_HOME", "HADOOP_HOME", "PYSPARK_PYTHON", "SPARK_LOCAL_DIRS"):
        print(f"  {var:<18} {os.environ.get(var, '<unset>')}")

    java = os.path.join(os.environ.get("JAVA_HOME", ""), "bin", "java.exe")
    if not os.path.isfile(java):
        java = shutil.which("java") or java
    ver = subprocess.run([java, "-version"], capture_output=True, text=True)
    print(f"  java        {ver.stderr.splitlines()[0] if ver.stderr else '??'}")

    import pyspark
    print(f"  pyspark     {pyspark.__version__}")

    import pyarrow
    print(f"  pyarrow     {pyarrow.__version__}")

    from pyspark.sql import SparkSession
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.feature import VectorAssembler, StandardScaler

    results: list[bool] = []

    print("\n== 1. JVM startup ==")
    spark = (
        SparkSession.builder.appName("smoke-test")
        .master("local[2]")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    results.append(_check("SparkSession started", True, f"Spark {spark.version}"))

    print("\n== 2. input data + Arrow pandas_udf ==")
    try:
        csv = raw_data_path("flights.csv")
    except FileNotFoundError:
        csv = ""
    if csv:
        df = spark.read.csv(csv, header=True, inferSchema=False).limit(1000)
        source = "flights.csv"
    else:
        from pyspark.sql import functions as F
        df = spark.range(1000).select(
            (F.lit(200.0) + F.col("id") % 2500).cast("string").alias("DISTANCE"),
            (F.lit(45.0) + F.col("id") % 300).cast("string").alias("SCHEDULED_TIME"),
        )
        source = "deterministic synthetic fallback (flights.csv not downloaded)"
    n = df.count()
    results.append(_check("read 1000 input rows", n == 1000, f"{source}; got {n}"))

    @pandas_udf(DoubleType())
    def arrow_hypot(a: pd.Series, b: pd.Series) -> pd.Series:
        # Vectorised at C level via Arrow - no per-row Python round trip.
        return pd.Series(np.hypot(a.to_numpy(dtype="float64"),
                                  b.to_numpy(dtype="float64")))

    feat = (
        df.selectExpr("cast(DISTANCE as double) d",
                      "cast(SCHEDULED_TIME as double) t")
        .na.drop()
        .withColumn("h", arrow_hypot("d", "t"))
    )
    row = feat.first()
    expected = float(np.hypot(row["d"], row["t"]))
    results.append(_check("Arrow pandas_udf executed", abs(row["h"] - expected) < 1e-6,
                          f"{row['h']:.4f} vs {expected:.4f}"))

    print("\n== 3. PipelineModel save + reload (needs hadoop.dll) ==")
    pipe = Pipeline(stages=[
        VectorAssembler(inputCols=["d", "t"], outputCol="raw"),
        StandardScaler(inputCol="raw", outputCol="scaled", withMean=False, withStd=True),
    ])
    model = pipe.fit(feat)
    before = [r["scaled"] for r in model.transform(feat).select("scaled").take(5)]

    tmp = tempfile.mkdtemp(prefix="smoke_model_", dir=spark_tmp)
    path = os.path.join(tmp, "pipeline")
    try:
        model.write().overwrite().save(path)
        results.append(_check("PipelineModel.save()", True, path))
        after = [r["scaled"] for r in PipelineModel.load(path).transform(feat).select("scaled").take(5)]
        results.append(_check("PipelineModel.load() round-trips", before == after))
    except Exception as exc:  # noqa: BLE001 - we want the diagnosis, not a traceback
        results.append(_check("PipelineModel save/load", False, f"{type(exc).__name__}: {exc}"))
        if "NativeIO" in str(exc) or "UnsatisfiedLink" in str(exc):
            print("        -> hadoop.dll is missing or does not match Spark's Hadoop jars")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        spark.stop()

    ok = all(results)
    print(f"\n{'ALL CHECKS PASSED' if ok else 'SMOKE TEST FAILED'} ({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
