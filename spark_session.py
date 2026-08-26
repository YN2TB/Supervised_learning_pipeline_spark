"""Single place where the SparkSession is configured.

Every script in this project builds its session here so that memory,
Arrow, and shuffle settings stay consistent between the training run and
the streaming inference run.
"""
from __future__ import annotations

import os

REPO = os.path.dirname(os.path.abspath(__file__))


# MLflow tracking store. Deliberately SQLite rather than a bare `file:` store:
# the Model Registry (which the brief requires, including Staging -> Production
# transitions) is not supported by the filesystem store at all, and the file
# store also races when CrossValidator's parallel folds make autologging write
# nested runs from several threads at once.
TRACKING_URI = "sqlite:///" + os.path.join(REPO, "mlflow.db").replace("\\", "/")


def build_spark(app_name: str, cores: str = "*", driver_memory: str = "10g",
                shuffle_partitions: int = 64):
    """Return a configured local SparkSession.

    ``driver_memory`` has to be applied *before* the JVM starts. In local
    mode ``.config("spark.driver.memory", ...)`` is silently ignored because
    py4j has already launched the JVM, so it goes through PYSPARK_SUBMIT_ARGS.
    """
    os.environ.setdefault(
        "PYSPARK_SUBMIT_ARGS", f"--driver-memory {driver_memory} pyspark-shell"
    )

    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        # Arrow is what makes the pandas_udf stages run at C speed.
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "20000")
        # 5.8M rows on 16 cores: the 200 default just makes tiny partitions.
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.driver.maxResultSize", "2g")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def path(*parts: str) -> str:
    """Absolute path inside the repo, regardless of cwd."""
    return os.path.join(REPO, *parts)
