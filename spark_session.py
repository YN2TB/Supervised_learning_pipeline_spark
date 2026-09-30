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
                shuffle_partitions: int = 64, task_attempts: int = 4):
    """Return a configured local SparkSession.

    ``task_attempts``: plain ``local[n]`` gives every task exactly one attempt,
    so one crashed Python worker kills the whole job. That happens here, rarely
    and not reproducibly, to the Arrow UDF workers under full load (seen in
    scripts/feature_correlation.py: the same plan failed and passed on
    consecutive runs). ``local[n,4]`` retries the task instead. The core count,
    and so the read partitioning that randomSplit depends on, is unchanged;
    scripts/check_split.py asserts it.

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
        .master(f"local[{cores},{task_attempts}]")
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


# Where each MLflow experiment writes its results and plots. The two official
# tournaments are deliverables and live under docs/benchmarks; every trial run
# (dev checks, ablations, probes) goes to experiments/<name>/, next to the
# annotated log in experiments/LOG.md, so trials never mix with the results.
OFFICIAL_EXPERIMENTS = {
    "flight-delay-mllib": ("docs", "benchmarks"),                       # wheels-off
    "flight-delay-predeparture": ("docs", "benchmarks", "flight-delay-predeparture"),
    # The three-task tournament (2026-09-29): A and C, then B.
    "flight-delay-v3": ("docs", "benchmarks", "flight-delay-v3"),
    "flight-delay-gain": ("docs", "benchmarks", "flight-delay-gain"),
    # The same three tasks with --search wide.
    "flight-delay-v3-wide": ("docs", "benchmarks", "flight-delay-v3-wide"),
    "flight-delay-gain-wide": ("docs", "benchmarks", "flight-delay-gain-wide"),
    # Task B without DEPARTURE_DELAY as a feature (the team's choice, 2026-09-30).
    "flight-delay-gain-nodep": ("docs", "benchmarks", "flight-delay-gain-nodep"),
    # The final run: the chosen set of each task, wide search on the no-PCA arm.
    "flight-delay-final-a": ("docs", "benchmarks", "flight-delay-final-a"),
    "flight-delay-final-b": ("docs", "benchmarks", "flight-delay-final-b"),
    "flight-delay-final-c": ("docs", "benchmarks", "flight-delay-final-c"),
}


def results_dir(experiment: str) -> str:
    parts = OFFICIAL_EXPERIMENTS.get(experiment, ("experiments", experiment))
    return os.path.join(REPO, *parts)


def path(*parts: str) -> str:
    """Absolute path inside the repo, regardless of cwd."""
    return os.path.join(REPO, *parts)
