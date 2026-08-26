#!/usr/bin/env bash
#
# Production execution script: submits the training pipeline through
# spark-submit rather than running it as a bare Python process.
#
#   ./submit_pipeline.sh                      # full run, default grids
#   ./submit_pipeline.sh --sample-fraction 0.1 --folds 3
#   MODE=inference ./submit_pipeline.sh       # streaming scorer
#
# On this machine it runs under Git Bash. env.sh pins the JDK, the Hadoop
# natives and the venv interpreter; see SETUP.md for why each is needed.
set -euo pipefail

cd "$(dirname "$0")"
# shellcheck source=env.sh
source ./env.sh

MODE="${MODE:-train}"

# local[16] matches the 16 cores on this host. Driver memory is set on the
# submit line because in local mode the driver JVM is already running by the
# time a SparkSession builder config would be read.
MASTER="${MASTER:-local[16]}"
DRIVER_MEM="${DRIVER_MEM:-12g}"

# custom_transformers.py must ship with the job: the serialized PipelineModel
# refers to those classes by qualified name, so both the driver and every
# Python worker have to be able to import them.
PY_FILES="custom_transformers.py,spark_session.py,flight_schema.py,mllib_pipeline.py"

COMMON_CONF=(
  --master "$MASTER"
  --driver-memory "$DRIVER_MEM"
  --py-files "$PY_FILES"
  --conf spark.sql.execution.arrow.pyspark.enabled=true
  --conf spark.sql.execution.arrow.maxRecordsPerBatch=20000
  --conf spark.sql.shuffle.partitions=64
  --conf spark.sql.adaptive.enabled=true
  --conf spark.driver.maxResultSize=2g
  --conf spark.local.dir="${SPARK_LOCAL_DIRS}"
)

case "$MODE" in
  train)
    echo "==> training pipeline (spark-submit, $MASTER)"
    exec spark-submit "${COMMON_CONF[@]}" mllib_pipeline.py "$@"
    ;;
  inference)
    echo "==> streaming inference (spark-submit, $MASTER)"
    exec spark-submit "${COMMON_CONF[@]}" inference.py "$@"
    ;;
  benchmark)
    echo "==> rendering benchmark figures"
    exec python benchmark_results.py "$@"
    ;;
  *)
    echo "unknown MODE=$MODE (expected train|inference|benchmark)" >&2
    exit 2
    ;;
esac
