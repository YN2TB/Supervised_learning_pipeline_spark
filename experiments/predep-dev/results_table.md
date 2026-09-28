
### Regression arm

| model | arm | rmse | mae | r2 | train_s |
|---|---|---|---|---|---|
| glm_poisson_log | nopca | 39.3906 | 20.9526 | 0.0191 | 16 |
| linear_regression | nopca | 39.4110 | 20.9823 | 0.0181 | 20 |
| glm_poisson_log | pca | 39.4282 | 21.0184 | 0.0172 | 357 |
| linear_regression | pca | 39.4389 | 21.0397 | 0.0167 | 590 |
| random_forest_regressor | nopca | 39.4518 | 20.9150 | 0.0161 | 45 |
| random_forest_regressor | pca | 39.4633 | 20.9633 | 0.0155 | 596 |
| gbt_regressor | pca | 39.6045 | 21.0837 | 0.0084 | 690 |
| gbt_regressor | nopca | 39.6628 | 20.9280 | 0.0055 | 162 |
| baseline_regression | none | 39.7727 | 21.3409 | -0.0000 | nan |

### Classification arm

| model | arm | areaUnderROC | areaUnderPR | f1 | accuracy | train_s |
|---|---|---|---|---|---|---|
| gbt_classifier | nopca | 0.6561 | 0.1869 | 0.6893 | 0.6176 | 151 |
| random_forest_classifier | nopca | 0.6396 | 0.1732 | 0.6839 | 0.6110 | 44 |
| gbt_classifier | pca | 0.6391 | 0.1740 | 0.8393 | 0.8906 | 466 |
| linear_svc | nopca | 0.6374 | 0.1758 | 0.6350 | 0.5532 | 45 |
| random_forest_classifier | pca | 0.6281 | 0.1698 | 0.8386 | 0.8903 | 573 |
| linear_svc | pca | 0.5412 | 0.1207 | 0.8386 | 0.8903 | 364 |
| baseline_classification | none | 0.5000 | 0.1097 | 0.8386 | 0.8903 | nan |
