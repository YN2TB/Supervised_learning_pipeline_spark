#!/usr/bin/env bash
# Spark runtime environment for the Big Data MLlib assignment (Git Bash).
# Python 3.11 + JDK 17 + Hadoop 3.3.5 natives. See SETUP.md for why each is pinned.
# Usage:  source ./env.sh
#
# Every value here is project-local: nothing outside D:\BigData is modified.
# The system JDK 26 (unsupported by Spark) and the Hadoop 3.4.1 install in
# C:\Hadoop (whose natives do not match Spark's bundled Hadoop 3.3.4 client
# jars) are both deliberately bypassed.

# --- Windows-style paths: consumed by Spark's .cmd launchers and by the JVM ---
export JAVA_HOME='D:\BigData\.jdk\jdk-17.0.20.1+1'
export HADOOP_HOME='D:\BigData\.hadoop'
export SPARK_LOCAL_DIRS='D:\BigData\.spark-tmp'

# Executors and driver must both launch the venv interpreter, otherwise Spark
# picks whatever `python` is first on PATH (here: a broken 3.14 + pyspark 3.4).
export PYSPARK_PYTHON='D:\BigData\.venv\Scripts\python.exe'
export PYSPARK_DRIVER_PYTHON='D:\BigData\.venv\Scripts\python.exe'

# So `import custom_transformers` resolves when a saved PipelineModel is
# deserialized (Windows uses ';' as the PYTHONPATH separator).
export PYTHONPATH="D:\BigData;${PYTHONPATH}"

# --- POSIX-style paths: consumed by bash itself ---
export PATH="/d/BigData/.jdk/jdk-17.0.20.1+1/bin:/d/BigData/.hadoop/bin:/d/BigData/.venv/Scripts:${PATH}"

export PYTHONUTF8=1
