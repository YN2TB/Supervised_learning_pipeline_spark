#!/usr/bin/env bash
# Spark runtime environment for the Big Data MLlib assignment (Git Bash).
# Python 3.11 + JDK 17 + Hadoop 3.3.5 natives. See SETUP.md for why each is pinned.
# Usage:  source ./env.sh
#
# Resolve from this file, so cloning/moving the repository never invalidates
# the runtime configuration. This file targets Git Bash on Windows.
PROJECT_ROOT_POSIX="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if command -v cygpath >/dev/null 2>&1; then
  PROJECT_ROOT_WIN="$(cygpath -w "$PROJECT_ROOT_POSIX")"
else
  PROJECT_ROOT_WIN="$PROJECT_ROOT_POSIX"
fi

# --- Windows-style paths: consumed by Spark's .cmd launchers and by the JVM ---
LOCAL_JDK_POSIX="$(find "$PROJECT_ROOT_POSIX/.jdk" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -n 1)"
if [[ -n "$LOCAL_JDK_POSIX" ]]; then export JAVA_HOME="$(cygpath -w "$LOCAL_JDK_POSIX")"; fi
if [[ -d "$PROJECT_ROOT_POSIX/.hadoop" ]]; then export HADOOP_HOME="$PROJECT_ROOT_WIN\\.hadoop"; fi
export SPARK_LOCAL_DIRS="$PROJECT_ROOT_WIN\\.spark-tmp"
mkdir -p "$PROJECT_ROOT_POSIX/.spark-tmp"

# Executors and driver must both launch the venv interpreter, otherwise Spark
# picks whatever `python` is first on PATH (here: a broken 3.14 + pyspark 3.4).
if [[ -x "$PROJECT_ROOT_POSIX/.venv/Scripts/python.exe" ]]; then
  PROJECT_PYTHON="$PROJECT_ROOT_WIN\\.venv\\Scripts\\python.exe"
else
  PROJECT_PYTHON="$(command -v python)"
fi
export PYSPARK_PYTHON="$PROJECT_PYTHON"
export PYSPARK_DRIVER_PYTHON="$PROJECT_PYTHON"

# So `import custom_transformers` resolves when a saved PipelineModel is
# deserialized (Windows uses ';' as the PYTHONPATH separator).
export PYTHONPATH="$PROJECT_ROOT_WIN;${PYTHONPATH:-}"

# --- POSIX-style paths: consumed by bash itself ---
export PATH="$PROJECT_ROOT_POSIX/.venv/Scripts:$PROJECT_ROOT_POSIX/.hadoop/bin${LOCAL_JDK_POSIX:+:$LOCAL_JDK_POSIX/bin}:${PATH}"

export PYTHONUTF8=1
