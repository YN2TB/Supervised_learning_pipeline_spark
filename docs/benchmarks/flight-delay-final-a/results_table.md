
### Regression arm

| model | arm | mae | median_ae | rmse | r2 | r2_arrival | train_s |
|---|---|---|---|---|---|---|---|
| gbt_regressor | nopca | 15.8505 | 9.5035 | 33.3960 | 0.3747 | - | 7632 |
| random_forest_regressor | nopca | 16.1678 | 9.7706 | 33.8634 | 0.3571 | - | 1237 |
| gbt_regressor | pca | 16.4202 | 10.0276 | 34.0407 | 0.3503 | - | 674 |
| random_forest_regressor | pca | 16.9112 | 10.6148 | 34.6530 | 0.3267 | - | 521 |
| linear_regression | nopca | 17.3715 | 11.0396 | 35.2937 | 0.3016 | - | 195 |
| glm_gaussian_identity | nopca | 17.3749 | 11.0413 | 35.2939 | 0.3016 | - | 82 |
| linear_regression | pca | 17.2650 | 10.8125 | 35.4257 | 0.2964 | - | 141 |
| glm_gaussian_identity | pca | 17.2666 | 10.8140 | 35.4258 | 0.2964 | - | 111 |
| rule_prev_arr_delay_regression | none | 18.5922 | - | 37.3167 | 0.2192 | - | nan |
| baseline_regression | none | 22.2062 | - | 42.3240 | -0.0043 | - | nan |

### Classification arm

| model | arm | recall_at_10pct | precision_at_10pct | precision_late | recall_late | areaUnderROC | areaUnderPR | train_s |
|---|---|---|---|---|---|---|---|---|
| rule_prev_arr_delay_classification | none | - | - | 0.7873 | 0.3465 | 0.7610 | 0.5074 | nan |
