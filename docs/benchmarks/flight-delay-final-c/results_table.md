
### Regression arm

| model | arm | mae | median_ae | rmse | r2 | r2_arrival | train_s |
|---|---|---|---|---|---|---|---|
| rule_prev_arr_delay_regression | none | 18.5922 | - | 37.3167 | 0.2192 | - | nan |

### Classification arm

| model | arm | recall_at_10pct | precision_at_10pct | precision_late | recall_late | areaUnderROC | areaUnderPR | train_s |
|---|---|---|---|---|---|---|---|---|
| gbt_classifier | nopca | 0.5350 | 0.6833 | 0.4330 | 0.6866 | 0.8559 | 0.6721 | 6906 |
| random_forest_classifier | nopca | 0.5249 | 0.6703 | 0.4853 | 0.6259 | 0.8430 | 0.6524 | 1043 |
| gbt_classifier | pca | 0.5160 | 0.6589 | 0.4228 | 0.6596 | 0.8381 | 0.6473 | 440 |
| random_forest_classifier | pca | 0.4979 | 0.6359 | 0.4401 | 0.6311 | 0.8307 | 0.6308 | 446 |
| linear_svc | nopca | 0.4916 | 0.6278 | 0.4673 | 0.5989 | 0.8176 | 0.6154 | 148 |
| linear_svc | pca | 0.4860 | 0.6206 | 0.5099 | 0.5411 | 0.8087 | 0.5992 | 132 |
| rule_prev_arr_delay_classification | none | - | - | 0.7873 | 0.3465 | 0.7610 | 0.5074 | nan |
| baseline_classification | none | - | - | 0.0000 | 0.0000 | 0.5000 | 0.1275 | nan |
