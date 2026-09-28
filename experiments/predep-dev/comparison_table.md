Reference `flight-delay-mllib` against `predep-dev`, test split, same seed.


### Regression

| model | arm | rmse (ref) | rmse (now) | mae (ref) | mae (now) | r2 (ref) | r2 (now) |
|---|---|---|---|---|---|---|---|
| baseline_regression | none | - | 39.7727 | - | 21.3409 | - | -0.0000 |
| gbt_regressor | nopca | 12.8530 | 39.6628 | 7.5262 | 20.9280 | 0.8938 | 0.0055 |
| gbt_regressor | pca | 15.2927 | 39.6045 | 10.0364 | 21.0837 | 0.8497 | 0.0084 |
| glm_poisson_log | nopca | 40.1434 | 39.3906 | 11.4800 | 20.9526 | -0.0357 | 0.0191 |
| glm_poisson_log | pca | 27.0180 | 39.4282 | 12.5177 | 21.0184 | 0.5309 | 0.0172 |
| linear_regression | nopca | 10.5171 | 39.4110 | 7.1939 | 20.9823 | 0.9289 | 0.0181 |
| linear_regression | pca | 21.4813 | 39.4389 | 14.3708 | 21.0397 | 0.7034 | 0.0167 |
| random_forest_regressor | nopca | 14.0499 | 39.4518 | 8.3463 | 20.9150 | 0.8731 | 0.0161 |
| random_forest_regressor | pca | 15.0479 | 39.4633 | 10.0716 | 20.9633 | 0.8545 | 0.0155 |

### Classification

| model | arm | areaUnderROC (ref) | areaUnderROC (now) | areaUnderPR (ref) | areaUnderPR (now) | f1 (ref) | f1 (now) |
|---|---|---|---|---|---|---|---|
| baseline_classification | none | - | 0.5000 | - | 0.1097 | - | 0.8386 |
| gbt_classifier | nopca | 0.9838 | 0.6561 | 0.9433 | 0.1869 | 0.9729 | 0.6893 |
| gbt_classifier | pca | 0.9700 | 0.6391 | 0.8799 | 0.1740 | 0.9546 | 0.8393 |
| linear_svc | nopca | 0.9806 | 0.6374 | 0.9338 | 0.1758 | 0.9675 | 0.6350 |
| linear_svc | pca | 0.9657 | 0.5412 | 0.8769 | 0.1207 | 0.9547 | 0.8386 |
| random_forest_classifier | nopca | 0.9792 | 0.6396 | 0.9277 | 0.1732 | 0.9683 | 0.6839 |
| random_forest_classifier | pca | 0.9637 | 0.6281 | 0.8713 | 0.1698 | 0.9526 | 0.8386 |
