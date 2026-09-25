#!/usr/bin/env bash
# Feature-set ablation on a 5% sample, no-PCA arms only, nothing registered.
#
#   source ./env.sh && bash scripts/run_ablation.sh
#
# One experiment per feature set (ablation-<set>), so they never mix with each
# other or with the real tournament. Compare with:
#
#   python benchmark_results.py --experiment ablation-predeparture --compare ablation-predeparture_all
#
# Reading it: predeparture should be within noise of predeparture_all (the pruned
# columns carried nothing); predeparture_nofreq tells whether the frequency
# encoder earns its stage; wheelsoff is the old horizon, far above everything
# else because it is handed DEPARTURE_DELAY.
set -euo pipefail
cd "$(dirname "$0")/.."

SETS="${SETS:-predeparture_all predeparture predeparture_nofreq predeparture_sched predeparture_inbound_dep predeparture_inbound wheelsoff}"
MODELS="${MODELS:-linear_regression,gbt_regressor,linear_svc,gbt_classifier}"
SAMPLE="${SAMPLE:-0.05}"
FOLDS="${FOLDS:-3}"
PREFIX="${PREFIX:-ablation}"
NOTE="${NOTE:-feature-set ablation}"

for fs in $SETS; do
  echo "=== ablation: $fs ==="
  python -u mllib_pipeline.py \
    --feature-set "$fs" --experiment "$PREFIX-$fs" \
    --sample-fraction "$SAMPLE" --tune-fraction 0.5 --folds "$FOLDS" \
    --models "$MODELS" --arms nopca --no-register --note "$NOTE: $fs" \
    2>&1 | tee ".spark-tmp/${PREFIX}_$fs.log" | grep -a --line-buffered -E "^(===|  (pca|nopca|baseline)|train=|Traceback)"
done
