Reference `flight-delay-mllib` against `flight-delay-final-a`, test split, same seed.


### Regression

| model | arm | mae (ref) | mae (now) | rmse (ref) | rmse (now) | r2 (ref) | r2 (now) |
|---|---|---|---|---|---|---|---|
| baseline_regression | none | - | 22.2062 | - | 42.3240 | - | -0.0043 |
| gbt_regressor | nopca | 7.5262 | 15.8505 | 12.8530 | 33.3960 | 0.8938 | 0.3747 |
| gbt_regressor | pca | 10.0364 | 16.4202 | 15.2927 | 34.0407 | 0.8497 | 0.3503 |
| glm_gaussian_identity | nopca | - | 17.3749 | - | 35.2939 | - | 0.3016 |
| glm_gaussian_identity | pca | - | 17.2666 | - | 35.4258 | - | 0.2964 |
| glm_poisson_log | nopca | 11.4800 | - | 40.1434 | - | -0.0357 | - |
| glm_poisson_log | pca | 12.5177 | - | 27.0180 | - | 0.5309 | - |
| linear_regression | nopca | 7.1939 | 17.3715 | 10.5171 | 35.2937 | 0.9289 | 0.3016 |
| linear_regression | pca | 14.3708 | 17.2650 | 21.4813 | 35.4257 | 0.7034 | 0.2964 |
| random_forest_regressor | nopca | 8.3463 | 16.1678 | 14.0499 | 33.8634 | 0.8731 | 0.3571 |
| random_forest_regressor | pca | 10.0716 | 16.9112 | 15.0479 | 34.6530 | 0.8545 | 0.3267 |
| rule_prev_arr_delay_regression | none | - | 18.5922 | - | 37.3167 | - | 0.2192 |

### Classification

| model | arm | areaUnderROC (ref) | areaUnderROC (now) | areaUnderPR (ref) | areaUnderPR (now) | recall_at_10pct (ref) | recall_at_10pct (now) |
|---|---|---|---|---|---|---|---|
| gbt_classifier | nopca | 0.9838 | - | 0.9433 | - | - | - |
| gbt_classifier | pca | 0.9700 | - | 0.8799 | - | - | - |
| linear_svc | nopca | 0.9806 | - | 0.9338 | - | - | - |
| linear_svc | pca | 0.9657 | - | 0.8769 | - | - | - |
| random_forest_classifier | nopca | 0.9792 | - | 0.9277 | - | - | - |
| random_forest_classifier | pca | 0.9637 | - | 0.8713 | - | - | - |
| rule_prev_arr_delay_classification | none | - | 0.7610 | - | 0.5074 | - | - |
