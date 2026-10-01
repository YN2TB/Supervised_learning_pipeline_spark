Reference `flight-delay-mllib` against `flight-delay-final-b`, test split, same seed.


### Regression

| model | arm | mae (ref) | mae (now) | rmse (ref) | rmse (now) | r2 (ref) | r2 (now) |
|---|---|---|---|---|---|---|---|
| baseline_regression | none | - | 9.2323 | - | 13.3507 | - | -0.0019 |
| gbt_regressor | nopca | 7.5262 | 6.2813 | 12.8530 | 9.2808 | 0.8938 | 0.5158 |
| gbt_regressor | pca | 10.0364 | 7.3108 | 15.2927 | 10.3602 | 0.8497 | 0.3967 |
| glm_gaussian_identity | nopca | - | 6.6523 | - | 9.6765 | - | 0.4737 |
| glm_gaussian_identity | pca | - | 7.5881 | - | 10.6702 | - | 0.3600 |
| glm_poisson_log | nopca | 11.4800 | - | 40.1434 | - | -0.0357 | - |
| glm_poisson_log | pca | 12.5177 | - | 27.0180 | - | 0.5309 | - |
| linear_regression | nopca | 7.1939 | 6.6523 | 10.5171 | 9.6765 | 0.9289 | 0.4737 |
| linear_regression | pca | 14.3708 | 7.5881 | 21.4813 | 10.6702 | 0.7034 | 0.3600 |
| physics_reference | none | - | 6.8751 | - | 9.8606 | - | 0.4534 |
| random_forest_regressor | nopca | 8.3463 | 7.0437 | 14.0499 | 10.0868 | 0.8731 | 0.4281 |
| random_forest_regressor | pca | 10.0716 | 7.4966 | 15.0479 | 10.5535 | 0.8545 | 0.3739 |

### Classification

| model | arm | areaUnderROC (ref) | areaUnderROC (now) | areaUnderPR (ref) | areaUnderPR (now) | recall_at_10pct (ref) | recall_at_10pct (now) |
|---|---|---|---|---|---|---|---|
| gbt_classifier | nopca | 0.9838 | - | 0.9433 | - | - | - |
| gbt_classifier | pca | 0.9700 | - | 0.8799 | - | - | - |
| linear_svc | nopca | 0.9806 | - | 0.9338 | - | - | - |
| linear_svc | pca | 0.9657 | - | 0.8769 | - | - | - |
| random_forest_classifier | nopca | 0.9792 | - | 0.9277 | - | - | - |
| random_forest_classifier | pca | 0.9637 | - | 0.8713 | - | - | - |
