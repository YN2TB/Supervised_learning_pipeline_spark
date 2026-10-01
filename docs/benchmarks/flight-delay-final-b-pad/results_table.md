
### Regression arm

| model | arm | mae | median_ae | rmse | r2 | r2_arrival | train_s |
|---|---|---|---|---|---|---|---|
| gbt_regressor | nopca | 6.2365 | 4.4035 | 9.2372 | 0.5204 | 0.9522 | 413 |
| linear_regression | nopca | 6.6781 | 4.8421 | 9.7060 | 0.4704 | 0.9472 | 41 |
| glm_gaussian_identity | nopca | 6.6781 | 4.8421 | 9.7060 | 0.4704 | 0.9472 | 37 |
| random_forest_regressor | nopca | 6.7981 | 4.9113 | 9.8472 | 0.4549 | 0.9456 | 612 |
| physics_reference | none | 6.8751 | - | 9.8606 | 0.4534 | 0.9455 | nan |
| gbt_regressor | pca | 7.1089 | 5.2270 | 10.1344 | 0.4227 | 0.9424 | 161 |
| random_forest_regressor | pca | 7.2967 | 5.3865 | 10.3393 | 0.3991 | 0.9401 | 294 |
| linear_regression | pca | 7.3209 | 5.3849 | 10.3876 | 0.3935 | 0.9395 | 49 |
| glm_gaussian_identity | pca | 7.3209 | 5.3849 | 10.3876 | 0.3935 | 0.9395 | 53 |
| baseline_regression | none | 9.2323 | - | 13.3507 | -0.0019 | 0.9001 | nan |
