
### Regression arm

| model | arm | rmse | mae | r2 | train_s |
|---|---|---|---|---|---|
| linear_regression | nopca | 10.5171 | 7.1939 | 0.9289 | 1111 |
| gbt_regressor | nopca | 12.8530 | 7.5262 | 0.8938 | 2325 |
| random_forest_regressor | nopca | 14.0499 | 8.3463 | 0.8731 | 1881 |
| random_forest_regressor | pca | 14.9809 | 10.0465 | 0.8558 | 1731 |
| gbt_regressor | pca | 15.2086 | 10.0520 | 0.8513 | 2262 |
| linear_regression | pca | 21.4813 | 14.3725 | 0.7034 | 1142 |
| glm_poisson_log | pca | 27.0180 | 12.5177 | 0.5309 | 2149 |
| glm_poisson_log | nopca | 40.1434 | 11.4800 | -0.0357 | 2073 |

### Classification arm

| model | arm | areaUnderROC | areaUnderPR | f1 | accuracy | train_s |
|---|---|---|---|---|---|---|
| gbt_classifier | nopca | 0.9817 | 0.9384 | 0.9716 | 0.9723 | 2336 |
| linear_svc | nopca | 0.9806 | 0.9338 | 0.9675 | 0.9692 | 959 |
| random_forest_classifier | nopca | 0.9792 | 0.9277 | 0.9683 | 0.9691 | 2084 |
| gbt_classifier | pca | 0.9699 | 0.8801 | 0.9546 | 0.9559 | 1533 |
| linear_svc | pca | 0.9657 | 0.8769 | 0.9547 | 0.9561 | 803 |
| random_forest_classifier | pca | 0.9635 | 0.8709 | 0.9523 | 0.9547 | 2074 |
