#!/usr/bin/env bash
# Wait for the running tournament to finish, then re-run the GLM arm alone.
#
# The Gamma/log GLM overflowed on the full data (MAE 10.80 but RMSE 568.90 -
# a handful of exp(X.beta) blowups dragging R^2 to -207). mllib_pipeline.py now
# uses Poisson/log with heavier regularisation; this re-runs just that arm
# rather than repeating the whole 14-arm tournament.
set -u
cd /d/BigData
source ./env.sh
export PYSPARK_SUBMIT_ARGS="--driver-memory 14g pyspark-shell"

TARGET_PID="${1:-20164}"
echo "waiting for tournament driver PID ${TARGET_PID} to exit... $(date)"
while tasklist //FI "PID eq ${TARGET_PID}" //FO CSV 2>/dev/null | grep -q "python.exe"; do
  sleep 60
done
echo "tournament finished at $(date); starting GLM re-run"

python -u mllib_pipeline.py \
  --tune-fraction 0.02 --folds 5 \
  --models glm_poisson_log \
  --experiment flight-delay-mllib \
  > .spark-tmp/glm_rerun.log 2>&1
echo "GLM re-run exited $? at $(date)"
