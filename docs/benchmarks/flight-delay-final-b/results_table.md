
### Regression arm

| model | arm | mae | median_ae | rmse | r2 | r2_arrival | train_s |
|---|---|---|---|---|---|---|---|
| gbt_regressor | nopca | 6.2813 | 4.4489 | 9.2808 | 0.5158 | 0.9517 | 7462 |
| linear_regression | nopca | 6.6523 | 4.8050 | 9.6765 | 0.4737 | 0.9475 | 229 |
| glm_gaussian_identity | nopca | 6.6523 | 4.8050 | 9.6765 | 0.4737 | 0.9475 | 86 |
| physics_reference | none | 6.8751 | - | 9.8606 | 0.4534 | 0.9455 | nan |
| random_forest_regressor | nopca | 7.0437 | 5.1421 | 10.0868 | 0.4281 | 0.9430 | 1761 |
| gbt_regressor | pca | 7.3108 | 5.3734 | 10.3602 | 0.3967 | 0.9398 | 659 |
| random_forest_regressor | pca | 7.4966 | 5.5590 | 10.5535 | 0.3739 | 0.9376 | 548 |
| linear_regression | pca | 7.5881 | 5.6214 | 10.6702 | 0.3600 | 0.9362 | 134 |
| glm_gaussian_identity | pca | 7.5881 | 5.6214 | 10.6702 | 0.3600 | 0.9362 | 106 |
| baseline_regression | none | 9.2323 | - | 13.3507 | -0.0019 | 0.9001 | nan |
