Reference `flight-delay-mllib` against `flight-delay-final-c`, test split, same seed.


### Regression

| model | arm | mae (ref) | mae (now) | rmse (ref) | rmse (now) | r2 (ref) | r2 (now) |
|---|---|---|---|---|---|---|---|
| gbt_regressor | nopca | 7.5262 | - | 12.8530 | - | 0.8938 | - |
| gbt_regressor | pca | 10.0364 | - | 15.2927 | - | 0.8497 | - |
| glm_poisson_log | nopca | 11.4800 | - | 40.1434 | - | -0.0357 | - |
| glm_poisson_log | pca | 12.5177 | - | 27.0180 | - | 0.5309 | - |
| linear_regression | nopca | 7.1939 | - | 10.5171 | - | 0.9289 | - |
| linear_regression | pca | 14.3708 | - | 21.4813 | - | 0.7034 | - |
| random_forest_regressor | nopca | 8.3463 | - | 14.0499 | - | 0.8731 | - |
| random_forest_regressor | pca | 10.0716 | - | 15.0479 | - | 0.8545 | - |
| rule_prev_arr_delay_regression | none | - | 18.5922 | - | 37.3167 | - | 0.2192 |

### Classification

| model | arm | areaUnderROC (ref) | areaUnderROC (now) | areaUnderPR (ref) | areaUnderPR (now) | recall_at_10pct (ref) | recall_at_10pct (now) |
|---|---|---|---|---|---|---|---|
| baseline_classification | none | - | 0.5000 | - | 0.1275 | - | - |
| gbt_classifier | nopca | 0.9838 | 0.8559 | 0.9433 | 0.6721 | - | 0.5350 |
| gbt_classifier | pca | 0.9700 | 0.8381 | 0.8799 | 0.6473 | - | 0.5160 |
| linear_svc | nopca | 0.9806 | 0.8176 | 0.9338 | 0.6154 | - | 0.4916 |
| linear_svc | pca | 0.9657 | 0.8087 | 0.8769 | 0.5992 | - | 0.4860 |
| random_forest_classifier | nopca | 0.9792 | 0.8430 | 0.9277 | 0.6524 | - | 0.5249 |
| random_forest_classifier | pca | 0.9637 | 0.8307 | 0.8713 | 0.6308 | - | 0.4979 |
| rule_prev_arr_delay_classification | none | - | 0.7610 | - | 0.5074 | - | - |
