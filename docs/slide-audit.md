# Slide audit

Every checkable claim in the 42-slide deck, verified against the curated lake,
the fitted `PipelineModel`, the MLflow store or the committed benchmarks. Nothing
below was taken from the report or the deck itself, so the two can be compared
without circularity.

**249 numeric claims extracted. 8 problems found, one of them serious.**

Method: claims pulled mechanically from the slide text, ground truth rebuilt from
artifacts, then compared. Prose without a figure was judged separately by reading
each slide against the code it describes.

---

## 1. Serious: the deck mixes two different tournaments

Every arm was fitted **twice**, once on 26 to 27 August and again on 13 to 14
September. The September run is the current one, and its
`random_forest_regressor__pca` (`1d3f0134`) is what sits in the registry at stage
Production. Some slides quote August and some quote September.

| slide | claim | which run | September value |
|---|---|---|---|
| 36 Results, Regression | all eight R² | September | correct |
| 37 Results, Classification | all six AUC | September | correct |
| **32 Gradient-Boosted Trees** | AUC **0.9817** | August | **0.9838** |
| **32** | "2,336 s against Random Forest's 2,084 s" | August | **1,052 s vs 1,437 s** |
| **38 The PCA Verdict** | 4 of 7 deltas | August | see below |
| **42 Summary** | GBT AUC **0.9817** | August | **0.9838** |

Slide 38's deltas, recomputed on the September run:

| arm | on the slide | September |
|---|---|---|
| linear regression | −0.2255 | −0.2255 (unchanged) |
| **GBT regressor** | −0.0425 | **−0.0441** |
| **random forest regressor** | −0.0173 | **−0.0186** |
| Poisson GLM | +0.5666 | +0.5666 (unchanged) |
| **GBT classifier** | −0.0118 | **−0.0138** |
| LinearSVC | −0.0149 | −0.0149 (unchanged) |
| **random forest classifier** | −0.0157 | **−0.0155** |

### The part that matters most

Slide 32 uses the wall-clock comparison to illustrate that GBT cannot parallelise
across trees: *"our best classifier, at 2,336 s against Random Forest's 2,084 s"*.

**In the September run that comparison inverts.** GBT took **1,052 s** and the
random forest **1,437 s**, so GBT was the *faster* of the two. The theoretical
claim is still true, because tree *m* genuinely cannot start before tree *m-1*
finishes, but the measurement no longer illustrates it: grid sizes and iteration
counts differ between the arms, and they dominate the comparison.

Either quote the September numbers and drop the illustration, or keep the
illustration and say plainly that it comes from a run where the grids happened to
favour the forest. Leaving a stale number that supports the argument is the worst
of the three.

---

## 2. Claims the data contradicts

Both are on slide 11, and both concern derived features whose stated purpose does
not survive measurement.

**`ROUTE_DETOUR`, described as capturing "airway dog-legs".** Measured over all
5,704,000 rows:

```
p1 0.9960   p25 0.9990   median 1.0009   p75 1.0020   p99 1.0039
```

The feature is constant to three decimal places. It cannot capture dog-legs
because `DISTANCE` in this feed **is itself a great-circle distance**, which is
the same fact as the 0.97 mi agreement quoted on the same slide. Its ratio to our
haversine is 1 by construction. Only 0.015% of rows exceed 1.01, and those are
the GUM and PPG rows with the bad coordinates.

**`SCHEDULE_SPEED_MPH`, described as "tight schedule, no slack".** The
distribution is genuine (median 322.4 mph, p1 118, p99 476), but the relationship
runs the wrong way:

| decile | implied speed | mean arrival delay |
|---|---|---|
| 1 | 57 to 198 mph | +4.53 min |
| 5 | 299 to 322 | +4.29 |
| 10 | 416 to 902 | +3.60 |

Correlation with arrival delay is **−0.0078**. Delay *falls* as scheduled speed
rises, because implied speed is mostly a proxy for leg length: short flights have
low implied speed (taxi and climb dominate the block time) and short flights run
late. It is not measuring schedule tightness.

Both are honest negatives and fit the deck's own framing better than the claims
do. They should be stated as measured, not asserted as motivation.

---

## 3. A measurement that cannot support its claim

**Slide 17, `0.28 s → 6.5 s (23×)`.** These were timed as
`model.transform(df).count()`. `count()` consumes none of the computed columns,
so Catalyst's column pruning is free to delete the projections entirely and the
scoring never runs. Re-measured with an aggregation on `prediction`, which cannot
be pruned, scoring costs **1.8 s of fixed cost plus 26 μs per row**, so 100,000
rows take 3.79 s rather than the 0.32 s the same method reported.

The direction of the cache trade is not in doubt. The ratio is. The same flaw
affects `scripts/profile_stages.py`, whose `transform` column should not be
quoted; its `fit` column is sound, because a fit cannot be pruned.

`docs/REPORT.md` B2.5 already carries this correction. Slide 17 does not.

---

## 4. Wrong figures, small

| slide | claim | measured | note |
|---|---|---|---|
| 05 The Data | `flights.csv` is **592 MB** | **565 MB** | off by 5% |
| 10 Outlier Clipping | DISTANCE fence would be **2,091 mi** | **2,106 mi** | Q3 + 1.5·IQR on the curated feed |
| 14 Imputer | median **322.35** | **322.3404** | rounds to 322.34 |

None changes an argument. All three are the kind of thing a marker checking one
number at random would find.

---

## 5. A committed artifact that describes the wrong arm

`docs/benchmarks/pca_explained_variance.json` sums to **28.875%**. That is the
**classification** arms' figure. The registered model is a regression arm and
retains **28.457%**.

They differ legitimately: `TargetEncoder` is fitted on the label, so regression
encodes against `label_delay` and classification against `label_severe`. The
feature matrix reaching PCA is not the same, so neither is its covariance.

Slide 26 is correct: its ten shares sum to 28.47%, the regression figure. It is
the committed JSON, and anything regenerated from it, that describes an arm the
project does not ship.

---

## 6. Verified

The large majority of claims check out exactly. Spot list:

| claim | source | status |
|---|---|---|
| 5,819,079 raw rows | `flights.csv` line count | exact |
| 5,704,000 curated rows | parquet | exact |
| 98.0% retained | 5,704,000 / 5,819,079 | exact |
| 29 curated columns | parquet schema | exact |
| 319 airports, 14 carriers | parquet cardinality | exact |
| 4,635 directed routes | parquet cardinality | exact |
| 2,336 airport pairs | parquet, direction collapsed | exact |
| 4,627 routes in the encoder | fitted `TargetEncoderModel` | exact |
| 11.08% severe | 11.0762% measured | rounds correctly |
| 15,627 flights a day | 5,704,000 / 365 | exact |
| Southwest 21.7% | 21.72% measured | rounds correctly |
| Atlanta 6.6% of departures | 6.554% measured | rounds correctly |
| the five `*_DELAY` sum to the target in 100.00% of rows | raw feed | exact |
| 1,063,439 rows carry the breakdown | raw feed | exact |
| 302 of 307 October codes | `dot_to_iata.csv` | exact |
| fences −23 / +25 and −1 / +31 | fitted `OutlierIQRTruncatorModel` | exact |
| prior 4.4172, smoothing 20 | fitted `TargetEncoderModel` | exact |
| LGA-LIT encodes +56.2 | fitted mapping | exact |
| ROUTE_DETOUR median 1.0009 | fitted `ImputerModel` | exact |
| corr(DEPARTURE_DELAY, target) 0.9446 | parquet | exact |
| corr(DISTANCE, SCHEDULED_TIME) 0.9844 | parquet | exact |
| 80 slots = 20 + 15 + 13 + 8 + 24 | fitted encoders | exact |
| dead slots 34, 35, 48; 80 to 77 | fitted `VarianceThresholdSelector` | exact |
| maxBins 64 | fitted `RandomForestRegressionModel` | exact |
| ElasticNet chose λ = 0.01, α = 0.0 | MLflow params | exact |
| 616 bytes per pass | 77 × 8 | exact |
| covariance 46 KB / 169 MB / 703 MB | n² × 8 at n = 77 / 4,703 / 9,599 | exact |
| GLM: 61.3% early, 3,494,095 rows, min −87 | parquet | exact |
| all eight regression R² on slide 36 | MLflow, September run | exact |
| all six classification AUC on slide 37 | MLflow, September run | exact |

---

## 7. Not verifiable from the artifacts

These are real measurements that were not persisted, so they can be neither
confirmed nor challenged without re-running the experiment:

- slide 13, the feature-treatment A/B (R² 0.390 / 0.409 / 0.937 on a 2% sample)
- slide 19, 336 against 672 bytes per row from `SizeEstimator`
- slide 23, `local-eigs` at 59.8 s, and slide 27's 16.8× against `dist-eigs`
- slide 27, λ10 = 1.22221 and λ11 = 1.21564 from the convergence probe
- slide 33, about 29 MB of histograms crossing the network
- `docs/REPORT.md` B2.5, the historical 146 s to 24.6 s pair, whose "before"
  reproduces at 151.9 s today but whose "after" does not

Where a figure like this is load-bearing, the cheapest fix is to re-run the
measurement and persist the output beside the benchmarks rather than quoting a
number from a session log.

---

## Triage

**Before submission**

1. Slide 32, 38, 42: move to September figures, and resolve the inverted GBT
   versus random forest timing rather than leaving the stale number that happens
   to support the argument.
2. Slide 11: restate `ROUTE_DETOUR` and `SCHEDULE_SPEED_MPH` as measured
   negatives.
3. Slide 17: qualify or replace the `0.28 s → 6.5 s` pair.

**Worth doing**

4. Regenerate `pca_explained_variance.json` from the regression arm, or label it.
5. Fix 592 MB, 2,091 mi, 322.35.

**Optional**

6. Re-run and persist the six unverifiable measurements in section 7.
