"""Single place where the SparkSession is configured.

Every script in this project builds its session here so that memory,
Arrow, and shuffle settings stay consistent between the training run and
the streaming inference run.
"""
from __future__ import annotations

import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))


def _bootstrap_local_runtime() -> None:
    """Make the repository runnable without first sourcing ``env.*``.

    Shell profiles are convenient for humans, but graders and CI commonly call
    ``python ...`` directly (and Windows may block local PowerShell scripts).
    Hadoop's local writer still needs ``winutils.exe`` on Windows, so discover
    the checked-in/rebuilt runtime before PySpark starts its JVM.
    """
    hadoop_home = os.path.join(REPO, ".hadoop")
    if os.path.isfile(os.path.join(hadoop_home, "bin", "winutils.exe")):
        os.environ.setdefault("HADOOP_HOME", hadoop_home)
        hadoop_bin = os.path.join(hadoop_home, "bin")
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if hadoop_bin not in path_parts:
            os.environ["PATH"] = hadoop_bin + os.pathsep + os.environ.get("PATH", "")

    # Custom PipelineModels refer to their Python classes by module name.
    python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if REPO not in python_path:
        os.environ["PYTHONPATH"] = REPO + os.pathsep + os.environ.get("PYTHONPATH", "")

    os.makedirs(os.path.join(REPO, ".spark-tmp"), exist_ok=True)
    os.environ.setdefault("SPARK_LOCAL_DIRS", os.path.join(REPO, ".spark-tmp"))


_bootstrap_local_runtime()


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
    # On Windows PySpark's upstream default is ``python3``, an executable name
    # that normally does not exist.  Pin workers to the interpreter that is
    # running this script.  An explicit cluster setting still wins.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
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


def raw_data_path(filename: str) -> str:
    """Locate a source CSV without requiring a 592 MB copy inside the repo.

    Search order is explicit environment override, the conventional in-repo
    ``data`` directory, then the assignment's sibling ``data_raw`` directory.
    """
    roots = []
    if os.environ.get("FLIGHT_RAW_DIR"):
        roots.append(os.path.abspath(os.environ["FLIGHT_RAW_DIR"]))
    roots.extend([path("data"), os.path.abspath(path("..", "data_raw"))])
    for root in roots:
        candidate = os.path.join(root, filename)
        if os.path.isfile(candidate):
            return candidate
    searched = "\n  - ".join(os.path.join(root, filename) for root in roots)
    raise FileNotFoundError(
        f"could not locate {filename}; searched:\n  - {searched}\n"
        "Set FLIGHT_RAW_DIR or pass --raw-dir to data_prep.py."
    )
