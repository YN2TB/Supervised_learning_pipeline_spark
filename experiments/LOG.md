# Experiment log

Trial runs for the pre-departure rework. One entry per trial: the question,
the command, what came back, what was decided. Each trial writes its results
and plots to `experiments/<experiment>/` and logs to MLflow under the same
experiment name (the note is the experiment's description in the MLflow UI).

The two official tournaments do not live here: `flight-delay-mllib`
(wheels-off, September) writes to `docs/benchmarks/`, and
`flight-delay-predeparture` to `docs/benchmarks/flight-delay-predeparture/`.

Unless stated, trials use `--no-register`, the tournament's split
(`make_split`, `cores="*"`), balanced class weights, and no-PCA arms only.
Numbers on a 1% sample move by about ±0.01 in AUC between runs; read small
gaps as ties.

---

## 2026-09-25 · predep-dev: does the pre-departure pipeline run end to end?

- **Command:** `python mllib_pipeline.py --sample-fraction 0.01 --tune-fraction 0.5 --folds 2 --no-register --experiment predep-dev`
- **Result:** all 14 arms finished. Pre-departure R² 0.006 to 0.019, AUC 0.54 to 0.65.
  PCA at 90% variance chose k = 51 (k = 10 had kept 23.7%).
- **Found:** without class weights every classifier predicted "not late" for every
  flight (F1 0.8386, identical to the all-negative baseline).
- **Decided:** balanced class weights by default (`--class-weight balanced`).
  Re-run of the three classifiers: LinearSVC AUC 0.584 → 0.637, and they now flag delays.
- **Also learned:** a PCA fit costs ~22 s at this size whatever the mode (Python row
  serialisation in RowMatrixPCA), so choosing k by variance adds nothing.

## 2026-09-25 · Haversine UDF crash in `feature_correlation.py --stage transformed`

- **Symptom:** the Python worker of the Arrow UDF died (EOFError / connection reset)
  at full size and at 25%, under Correlation.corr, a SQL aggregation and a persist.
- **Isolated:** every stage combination ran at 5%, 25% and 100% in separate probes;
  the same plan failed and passed on consecutive runs. Intermittent, not the code.
- **Decided:** `local[*,4]` so a crashed task is retried instead of killing the job
  (plain `local[n]` allows one attempt). Split verified unchanged. The full-size
  matrix then ran (4,564,168 rows).

## 2026-09-25 · pause-test: does the safe pause work?

- **Command:** 0.5% run; `.pause_tournament` created before the first arm, then
  during CV of the first arm; then resumed.
- **Result:** paused before the arm; paused after CV with the checkpoint saved;
  resume printed "CV restored from checkpoint" and went straight to the refit.

## 2026-09-25 · Why the wheels-off score is so high: the tail of DEPARTURE_DELAY

- **Question (from review):** should extreme departure delays be cut, e.g. over ~2 h?
- **Measured on all 5.7M curated flights:**
  - Flights departing more than 120 min late are **2.0% of flights but 62% of the
    target's total sum of squares** (> 60 min: 5.6% of flights, 77%). R² is mostly
    decided by them, and for them arrival ≈ departure delay is obvious.
  - A straight line on DEPARTURE_DELAY: R² 0.892 on all flights, **0.721 within
    departure delay ≤ 120 min**.
  - Time made up in the air does **not** grow for small delays: the median of
    (arrival − departure delay) is −6 min in every 15-min band from on time to
    300 min late; only the spread grows (sd 12 → 20). So the argument for cutting
    is "the tail is trivial and dominates the score", not "long delays cannot be
    recovered".
- **Decided:** `--max-dep-delay N` scopes the wheels-off model (train and test
  alike; legal because departure delay is known at wheels-off; refused for
  pre-departure sets). One-line rule baselines (`rule_departure_delay_*`,
  `rule_prev_arr_delay_*`) are logged whenever the set has that column, so a
  model is judged by what it adds over the rule.
- **Queued:** wheels-off at 1% within ≤ 120 min; leakage audit within ≤ 120 min.

## 2026-09-25 · leakage-audit: is either model leaking? (`scripts/leakage_audit.py`, 5% sample)

| test | wheels-off | predeparture_inbound |
|---|---|---|
| flights in both train and test | 0 | 0 |
| full pipeline, random split (LR / GBT R²) | 0.927 / 0.854 | 0.275 / 0.353 |
| straight line on one column | DEPARTURE_DELAY 0.890 | (rerun needed, see below) |
| labels shuffled within train (LR / GBT R²) | −0.022 / −0.051 | 0.001 / −0.013 |
| temporal split, Jan–Sep → Oct–Dec (LR / GBT R²) | 0.931 / 0.806 | 0.240 / 0.322 |
| physics reference (full split) | RMSE 8.78, R² 0.950 | n/a |

- **Wheels-off: no leak.** Shuffled labels give R² ≈ 0 (no path from a label into
  the features); the temporal split scores the same as the random one (the score
  does not come from neighbouring rows); and the physics reference, which knows
  taxi-out and guesses air time + taxi-in by their train mean per route, month and
  hour, beats the model (RMSE 8.78 against 10.57). A leaking model would beat it.
  The identity ARRIVAL_DELAY − DEPARTURE_DELAY = ELAPSED_TIME − SCHEDULED_TIME held
  on 100% of 1.14M test flights. The problem is easy, not leaky.
- **Pre-departure with rotation: no leak; a small random-split advantage.** Shuffled
  labels ≈ 0. The temporal split costs ~0.03 R² (GBT 0.353 → 0.322): flights of the
  same day sit on both sides of a random split. Report the temporal figure as the
  expectation for future months.
- **Bug found in the audit itself:** the single-column test dropped rows with a
  missing value, so for prev_arr_delay (null on the 25% first legs) it scored a
  different test set (0.320, not comparable). Fixed in `7db6e93` to fill with the
  train mean, as the rule baselines do; rerun pending.

### Wheels-off within departure delay ≤ 120 min (audit_wheelsoff_le120, 5%)

| test | all flights | ≤ 120 min (98% of flights) |
|---|---|---|
| straight line on DEPARTURE_DELAY, R² | 0.890 | 0.717 |
| model LR / GBT, R² | 0.927 / 0.854 | 0.813 / 0.818 |
| model LR / GBT, RMSE (min) | 10.57 / 14.96 | 10.41 / 10.27 |
| shuffled labels, R² | −0.02 / −0.05 | −0.02 / −0.02 |
| temporal split, R² | 0.931 / 0.806 | 0.800 / 0.800 |
| physics reference, RMSE / R² | 8.78 / 0.950 | 8.70 / 0.870 |

- Cutting the tail barely moves the per-flight error (RMSE ~10.4 either way) but
  drops R² by ~0.11: the tail inflated R² through its variance, not through better
  predictions. RMSE is the honest number to quote; R² needs the scope stated.
- Within the scope the model adds **+0.10 R² over the one-line rule** (vs +0.04 on all
  flights), still no leak signs (shuffled ≈ 0, temporal holds, physics reference ahead).
- The 1% scoped tournament (rot1-wheelsoff_le120) finished only LR (0.813), GLM
  Poisson (0.796) and LinearSVC (AUC 0.965 vs rule 0.950) before the queue was
  stopped for low memory (see below).

### 2026-09-26 01:23–02:10 · Stopped runs finished (driver capped at 6 GB)

- **rot1-wheelsoff_le120**, all 7 models within departure delay ≤ 120 min, 1%:
  R² LR 0.813, GBT 0.803, GLM Poisson 0.796, RF 0.787 against the rule's 0.715
  (RMSE 10.4 against 12.8); AUC GBT 0.973, RF 0.967, LinearSVC 0.965 against the
  rule's 0.950. Every model beats the rule within the scope.
- **rot1-predeparture_inbound_lean** (no prev_dep_delay, no has_prev) against
  `inbound`: every model loses a little (R² −0.009 to −0.017, AUC −0.002 to −0.010;
  only LinearSVC AUC-PR +0.006). Each gap is within 1% noise, but all seven point the
  same way, so the two columns carry something. **Decided:** keep the information,
  not the duplication. Proposed next: replace prev_dep_delay by
  prev_air_gain = prev_arr_delay − prev_dep_delay (what the inbound made up or lost
  in the air), nearly uncorrelated with prev_arr_delay. Not yet tested.

### 2026-09-26 02:20–03:00 · prev_air_gain, and dropping the frequency encoder (1%)

One change at a time against `predeparture_inbound`:

| model | inbound | lean | inbound_gain | final |
|---|---|---|---|---|
| GBT classifier AUC | 0.835 | 0.833 | 0.836 | 0.834 |
| RF classifier AUC | 0.835 | 0.826 | 0.834 | 0.830 |
| LinearSVC AUC | 0.802 | 0.792 | 0.801 | 0.799 |
| RF regressor R² | 0.313 | 0.296 | 0.316 | 0.316 |
| GBT regressor R² | 0.311 | 0.297 | 0.309 | 0.312 |
| LinearRegression R² | 0.267 | 0.258 | 0.267 | 0.266 |
| GLM Poisson R² | 0.191 | 0.190 | 0.200 | 0.200 |

- `inbound_gain` (prev_dep_delay → prev_air_gain) recovers what `lean` lost: the
  information stays, the near-duplicate goes.
- `final` (gain, without the frequency encoder) ties `inbound_gain` (≤ 0.005).
- **Decided:** `predeparture_final` is the tournament set and the new
  `DEFAULT_FEATURE_SET`: log_distance, SCHED_ARR_MIN, the four coordinates,
  leg_of_day, has_prev, turn_slack, prev_arr_delay, prev_air_gain, inbound_overrun,
  plus one-hot airline, month, weekday, hour. has_prev stays because it marks
  prev_* on a first leg as an imputed median. GLM stays Poisson/log (the brief);
  its gap to identity is documented above.
- Gates after the switch: smoke 5/5, transformers 17/17.

## 2026-09-26 03:0x · Full tournament, flight-delay-predeparture

- **Command:** `python -u mllib_pipeline.py --feature-set predeparture_final --no-register`
  (full data, 10% tuning sample, 5 folds, 7 models × PCA-by-90%-EVR / no-PCA),
  launched in its own PowerShell window so a low-memory reap of Claude Code's
  background shells cannot stop it. Log: `.spark-tmp/tournament_predeparture.log`.
- **Backup before it:** `backup_pre_tournament_predep_20260926_030321/`
  (mlflow.db, models). `--no-register`: the registry keeps v4 in Production.
- **First pass (03:03–04:29):** LR, GLM Poisson and LinearSVC finished, both arms:

  | arm | R² / AUC | AUC-PR | minutes |
  |---|---|---|---|
  | linear_regression pca / nopca | R² 0.248 / 0.270 | | 23 / 2 |
  | glm_poisson_log pca / nopca | R² 0.138 / 0.145 | | 14 / 2 |
  | linear_svc pca / nopca | AUC 0.779 / 0.807 | 0.520 / 0.541 | 15 / 2 |
  | rule on prev_arr_delay | R² 0.201, AUC 0.760 | 0.480 | |

- **Crash:** `java.lang.OutOfMemoryError: Java heap space` in the random forest
  regressor's PCA-arm cross-validation: 4 fits in parallel (`--parallelism 4`), trees
  up to depth 10 × 80, on 51 dense principal components (the September run had 10).
  The OOM shut the SparkContext down and every later arm "failed" within
  milliseconds (connection refused), which says nothing about those models.
- **Fixed:** the loop now stops when the SparkContext is gone (`d6166a5`), so a rerun
  resumes cleanly. **Resumed at 04:31 with `--parallelism 2`** (half the concurrent
  fits); the six finished arms are skipped.
- **Second OOM (~05:25), same arm.** The JVM held 11 GB and the machine had 1.1 GB
  free; stopped by hand (a JVM after an OutOfMemoryError is not trustworthy).
  PCA picked k = 53. Real cause: `MaterializeCache` persisted one frame per pipeline
  fit and nothing released them until the arm ended, so 20 copies (4 grid points × 5
  folds) piled up; with 10 components in September they fitted, with 53 they do not.
  **Fixed** (`d071d7b`): at most 4 frames stay persisted; older ones are released
  when a new one arrives (a released frame still in use is recomputed, never fails).
- **Resumed ~05:40** in two phases (`.spark-tmp/run_tournament_predeparture_trees.ps1`):
  the four tree models' no-PCA arms with `--parallelism 2`, then their PCA arms with
  `--parallelism 1`. (A first relaunch failed at argument parsing, the note's spaces
  split by cmd; relaunched at 05:17 with a space-free note.)
- **Result, test split (1,139,832 flights), 10 of 14 arms (the four tree PCA arms
  still running at the time of writing):**

  | model | arm | R² | RMSE | AUC | AUC-PR | minutes |
  |---|---|---|---|---|---|---|
  | GBT regressor | nopca | **0.349** | **31.83** | | | 12 |
  | RF regressor | nopca | 0.339 | 32.07 | | | 13 |
  | Linear regression | nopca | 0.270 | 33.70 | | | 2 |
  | Linear regression | pca | 0.248 | 34.21 | | | 23 |
  | GLM Poisson/log | nopca | 0.145 | 36.47 | | | 2 |
  | GLM Poisson/log | pca | 0.138 | 36.62 | | | 14 |
  | rule on prev_arr_delay | | 0.201 | 35.25 | | | |
  | train-mean baseline | | 0.000 | 39.45 | | | |
  | GBT classifier | nopca | | | **0.849** | **0.645** | 9 |
  | RF classifier | nopca | | | 0.840 | 0.622 | 10 |
  | LinearSVC | nopca | | | 0.807 | 0.541 | 3 |
  | LinearSVC | pca | | | 0.779 | 0.520 | 15 |
  | rule on prev_arr_delay | | | | 0.760 | 0.480 | |
  | all-negative baseline | | | | 0.500 | 0.111 | |

  In line with the 1% and 5% trials (GBT 0.31 → 0.35 R² with more data). Every model
  beats the one-line rule except the log-link GLM; the no-PCA arm wins every pair so
  far; PCA at 90% EVR kept k = 53.

- **Tree PCA arms (06:02–07:57):** RF regressor pca R² 0.287 (CV restored from the
  first attempt's checkpoint); GBT regressor pca R² 0.300; GBT classifier pca AUC
  0.817, AUC-PR 0.578. **RF classifier pca did not finish:** cross-validation
  completed (chose depth 10 × 80 trees, CV AUC 0.799, saved in
  `cv_checkpoints/random_forest_classifier__pca.json`), and the refit on 4.56M rows ×
  53 dense components ran the 10 GB heap out of memory again, even one fit at a time.
  Not retried: its CV score already places it below the no-PCA arm (0.839).
- **Final: 13 of 14 arms.** Best regressor GBT no-PCA (R² 0.349, RMSE 31.8, MAE
  15.4); best classifier GBT no-PCA (AUC 0.849, AUC-PR 0.645). The no-PCA arm wins
  all six complete pairs. Importance in every model is led by `inbound_overrun` and
  `prev_arr_delay`, then SCHED_ARR_MIN / turn_slack / leg_of_day / has_prev.
  Figures, `results_table.md` and `comparison_table.md` (against the wheels-off run)
  are in `docs/benchmarks/flight-delay-predeparture/`; `tournament_results.json`
  rebuilt from MLflow (each resume had overwritten it with its own arms only).
- **Registry untouched** (v4, wheels-off, still Production).
- **Lessons:** PCA to 90% EVR here keeps 53 of ~67 slots, so it neither compresses
  nor helps and costs 3–10x the time; the heavy tree arms need `--parallelism` ≤ 2
  and the bounded MaterializeCache on a 10 GB heap.

### 2026-09-26 afternoon · The three anomaly tests from FEATURE_DECISIONS.md (1%)

One change each against `predeparture_final` (differences from final, test split of
the 1% sample):

| model | final | zero-fill first legs | no log_distance | no prev_air_gain |
|---|---|---|---|---|
| GBT classifier AUC | 0.834 | −0.001 | −0.001 | +0.003 |
| RF classifier AUC | 0.830 | +0.004 | +0.000 | +0.004 |
| LinearSVC AUC | 0.799 | +0.000 | +0.000 | −0.000 |
| GBT regressor R² | 0.312 | −0.003 | −0.010 | −0.000 |
| RF regressor R² | 0.316 | −0.001 | −0.002 | −0.005 |
| LinearRegression R² | 0.266 | +0.000 | −0.000 | −0.001 |
| GLM Poisson R² | 0.200 | +0.001 | −0.000 | −0.009 |

- **Zero fill of prev_* on first legs:** metrics unchanged. The has_prev coefficient
  shrinks by ~20% (LinearRegression −5.17 → −4.12) but stays negative in every linear
  model: has_prev acts as the intercept of the flights that have an inbound, offsetting
  the average the prev_* terms add for them (e.g. prev_air_gain averages about −5 min).
  Its sign is not readable alone; the marginal one (first legs less delayed) is.
- **log_distance:** no measurable contribution (largest change −0.010 on one model).
- **prev_air_gain:** no measurable contribution, and no consistent direction
  (classifiers slightly better without it). The recovery credited to it on
  2026-09-26 02:20 came from has_prev, restored in the same step.
- **Not yet decided (user's call):** a leaner final without log_distance and
  prev_air_gain, zero-filled; the two removals have not been tested together.
- The run was stopped by Claude Code for low memory after the first two sets; the
  orphaned process finished the third on its own (memory had recovered to ~6 GB).

### 2026-09-28 · Kaggle C: airport coordinates and ROUTE_DETOUR (1%)

Private kernel `buihuynhgiahuy/flight-delay-exp-coords-detour`; output in
`experiments/kaggle-coords-detour-20260928/results/`. Same Kaggle split as A and B
(the `v2` rows match to four decimals).

| model | v2 | without coordinates | with ROUTE_DETOUR |
|---|---|---|---|
| RF classifier AUC | 0.829 | 0.819 (−0.009) | 0.825 (−0.004) |
| GBT classifier AUC | 0.832 | 0.827 (−0.005) | 0.830 (−0.001) |
| RF regressor R² | 0.282 | 0.278 (−0.005) | 0.282 (0.000) |
| GBT regressor R² | 0.280 | 0.280 (0.000) | 0.279 (−0.001) |
| LinearRegression, LinearSVC, GLM | | ≈ 0 | identical |

- LinearRegression is identical to four decimals in all three: CV picked
  elasticNetParam 0.5, and the lasso half set ROUTE_DETOUR's coefficient to exactly 0;
  the coordinates keep small coefficients (longitudes −0.41 / +0.89) whose removal moves
  R² only in the fifth decimal.
- **Coordinates: keep.** The trees lose a little without them, all in the same direction
  (RF classifier −0.009 AUC): it is the airport identity the trees split on.
- **ROUTE_DETOUR: stays out.** The lasso drops it and the trees do not improve.

### 2026-09-27 · Kaggle B: PCA on the whole vector against PCA on the numeric columns only (1%)

Private kernel `buihuynhgiahuy/flight-delay-exp-pca`; output in
`experiments/kaggle-pca-20260927/results/`. Same Kaggle split as run A (the `v2` rows
match A's to four decimals). 90% explained variance: the whole vector needs k = 51;
the 10 numeric columns need k = 8 (93%), as estimated.

| model | no PCA | PCA whole vector (k 51) | PCA numeric only (k 8) |
|---|---|---|---|
| GBT regressor R² | **0.280** | 0.228 | 0.267 |
| RF regressor R² | **0.282** | 0.236 | 0.263 |
| LinearRegression R² | 0.2455 | 0.238 | 0.2452 |
| GLM Poisson R² | **0.166** | 0.157 | 0.153 |
| GBT classifier AUC / PR | **0.832 / 0.634** | 0.796 / 0.569 | 0.817 / 0.594 |
| RF classifier AUC / PR | **0.829 / 0.618** | 0.787 / 0.546 | 0.796 / 0.576 |
| LinearSVC AUC / PR | 0.802 / 0.553 | 0.784 / 0.555 | **0.807 / 0.568** |

- Numeric-only PCA beats whole-vector PCA on six of seven models (+0.01 to +0.04), brings
  LinearRegression back to the no-PCA level, and runs lighter.
- For LinearSVC it beats no PCA too (+0.005 AUC, +0.015 AUC-PR): orthogonal inputs stop
  correlated columns offsetting each other.
- The trees still do best without PCA: rotation blurs the axis-aligned thresholds they
  split on (inbound_overrun's hinge).
- **Reading:** if the brief's PCA arm stays, numeric-only is the PCA arm to keep; the
  no-PCA arm remains the main one. Adoption is the team's call.

### 2026-09-27 · Kaggle A: cyclical encoding against one-hot (1%, kg-v2 vs kg-v2_cyclic)

First run of `kaggle/run_experiments.ipynb` as private kernel
`buihuynhgiahuy/flight-delay-exp-cyclic` (commit from `origin/test`), worked end to end
(setup, curated rebuild, 14 arms, results). Output in
`experiments/kaggle-cyclic-20260927/results/` (summary.csv, mlflow.db, plots, logs).
Kaggle's split differs from the laptop's (4 cores), so only the two rows of this run
compare: baseline RMSE 42.0 here against 39.8 at home.

| model | v2 (one-hot) | v2_cyclic | diff |
|---|---|---|---|
| GBT classifier AUC | 0.832 | 0.834 | +0.002 |
| RF classifier AUC | 0.829 | 0.828 | −0.001 |
| LinearSVC AUC | 0.802 | 0.803 | +0.002 |
| RF regressor R² | 0.282 | 0.285 | +0.003 |
| GBT regressor R² | 0.280 | 0.280 | 0.000 |
| LinearRegression R² | 0.245 | 0.244 | −0.001 |
| GLM Poisson R² | 0.166 | 0.166 | 0.000 |

A tie on every model (≤ 0.005), including the linear ones that were expected to lose
the one-hot's flexibility: one sin/cos pair for hour and two for month carry what 35
one-hot slots did. By the tie rule the cyclical version is the simpler one (6 dense
columns, a readable "evening peak"); adopting it is the team's call.

### 2026-09-27 · Open idea (not implemented): cyclical encoding of hour, month, weekday

Hour, month and weekday are one-hot (43 sparse slots). They are cyclical, which one-hot
ignores (23h is not next to 0h), but one-hot can take any shape, while a sin/cos pair
assumes one smooth wave per cycle. Checked against the EDA severe-delay rates
(unweighted by flight count; numpy only, nothing run):

| variable | 1 sin/cos pair explains | 2 pairs | shape |
|---|---|---|---|
| hour (24) | 82.7% | 83.7% | one wave, low morning, peak ~20h; misses the 4h → 5h cliff (11.6% → 3.5%, the first departures of the day) |
| month (12) | 25.8% | 91.0% | two peaks (Feb, Jun–Jul) and a Sep–Oct trough; one pair puts the peak in April, which is actually a trough (9.5%) |
| weekday (7) | 15.6% | 97.9% | no smooth cycle; two pairs is 5 parameters for 7 levels |

- Suggested variant: hour as 1 pair (2 columns for 23), month as 2 pairs (4 for 11),
  weekday kept one-hot. Expected: trees unchanged (they take time of day from
  SCHED_ARR_MIN, and would prefer a plain integer hour); linear models possibly a little
  worse (one-hot fits the full shape, and 4.5M rows are plenty for 43 coefficients).
  The gains are compactness (43 sparse → ~6 dense), a readable effect ("delays peak in
  the evening") and a natural fit for the numeric-only PCA idea below.
- **Decision (user's approach):** a separate variant to compare at 1%, not a replacement.

### 2026-09-26 · Open idea (not implemented): PCA on the numeric columns only

The PCA arm currently rotates the whole assembled vector, one-hot slots included. PCA
suits correlated continuous columns, not one-hot indicators (nearly independent, and a
0/1 column's variance has no geometric meaning), and projecting them turns a sparse
vector (~18 stored values per flight) into a dense one (53), which is what ran the heap
out in the tree PCA arms. Alternative: scale and PCA only the 10 numeric columns of
`predeparture_v2`, keep the one-hot groups as they are, assemble after.

- For it: the right tool for the right columns; no densification; decorrelated inputs
  remove the linear-model sign puzzles (has_prev, the longitude pair); real correlation
  to fold (prev_arr_delay ~ inbound_overrun 0.66, leg_of_day ~ has_prev 0.62, the
  longitudes 0.59, SCHED_ARR_MIN ~ leg_of_day 0.55).
- Against: small reduction (an estimate of ~7–8 of 10 components for 90%, not
  measured; ~67 → ~64 overall); components mix delays with coordinates and lose the
  interpretability the review asked for; trees split on axes, and rotating
  inbound_overrun's threshold into several components works against them.
- **Decision (user):** keep the current PCA arm as the default; treat numeric-only PCA
  as a separate direction to compare side by side (e.g. an opt-in `--pca-scope numeric`),
  not a replacement. Nothing built or run yet.

### 2026-09-26 · Decision: predeparture_v2

Dropped log_distance and prev_air_gain (no measurable contribution); median imputation
kept for first legs. `predeparture_v2` (10 numeric + 4 one-hot groups) is the new
`DEFAULT_FEATURE_SET`. Pending: a 1% run confirming the two removals together, then the
next full tournament, after the team's visual EDA review. The 2026-09-26 tournament
results (predeparture_final) stay the current reference.

### 2026-09-26 00:5x · Queue stopped: system low on memory

Claude Code stopped the Q2 shell while the machine was critically short of memory.
Its child (the rot1-wheelsoff_le120 run, a 5.7 GB JVM) kept running orphaned and was
then stopped by hand, as was the Q3 waiter (`inbound_lean`, never started).
Free memory went from 3.6 GB to 9.8 GB. Not rerun; pending the user's go-ahead:
the rest of rot1-wheelsoff_le120 (RF, GBT arms) and `inbound_lean`.

## 2026-09-25 · What the public Kaggle notebooks on this dataset report, and why it does not compare

Read (not run) from kaggle.com/datasets/usdot/flight-delays/code, sorted by votes.

- **"Predicting flight delays [Tutorial]"** (FabienDaniel, top voted), linear
  regression with MSE ~50 to 54 (RMSE ~7). Not comparable with ours:
  January 2015 only, one airline (American); each row is the **mean departure
  delay of a group** (airport × departure hour), 1,831 rows in all, not one flight;
  **delays over 1 h removed before averaging**; and the MSE ~54 is scored on the
  training set, as the author notes. Its held-out test (last week of January) is
  MSE 74.8, RMSE 8.65. Averaging removes most of the per-flight noise.
  Worth borrowing: a pre-departure framing (airport and hour only), stating the
  1 h cut openly, and reporting "share of predictions off by more than 15 min"
  (4.6% there) next to RMSE.
- **"Flight Delay: Prediction - 0.9983 Accuracy"**: a textbook leak. The target is
  ARRIVAL_DELAY in four bands (15/30/60 min); the features keep the five
  `*_DELAY` attribution columns, which by DOT definition sum to the arrival delay
  for every flight ≥ 15 min late, and are null otherwise (filled with the mean,
  so "is this the mean" alone tells which side of 15 min a flight is on). The
  0.9983 is an AUC, not an accuracy. These are exactly the columns
  `flight_schema.LEAKY_COLUMNS` removes at ingest. Useful as the contrast when
  asked how a leak would look.

### 2026-09-26 · Who else reports R² as high as the old model, and why (GitHub READMEs)

| project | inputs | reported | reading |
|---|---|---|---|
| pranaykmr/FlightDelayPrediction | DEPARTURE_DELAY, ELAPSED_TIME, AIR_TIME, TAXI_IN, ARRIVAL_TIME, ARRIVAL_DELAY itself | R² ~0.9, MAE 1.5e−14, classification 1.0 | the target is among the features; without DEPARTURE_DELAY their R² goes negative |
| akasha456/Flight-Delay-Detection | NAS_Delay + Dep_Delay | R² 0.972, accuracy 98% | NAS delay is one of the attribution columns that sum to the target |
| chris0andra/flight-delay-prediction (2015 data) | pre-departure only; time split (Dec test); ≥ 15 min | ROC-AUC 0.58–0.61, PR-AUC 0.20–0.25 | honest; matches our schedule-only set (~0.66 at > 30 min, random split) |
| priyankatelukuntla/flight-delay-prediction-ml (newer BTS data) | pre-flight schedule only | R² 0.06 (LR) to 0.16 (XGBoost), RMSE ~38 | honest; same level as ours before the rotation features |

On Kaggle itself (read in the browser, 2026-09-26):

| notebook | inputs | reported | reading |
|---|---|---|---|
| rahulstephenites2 / Airline_Flight_DelayTime_Prediction | four `*_DELAY` attribution columns + SECURITY_DELAY, ELAPSED_TIME, AIR_TIME, TAXI_IN, TAXI_OUT, DEPARTURE_DELAY | random forest **R² 0.988 test**, MAE ~2.1 min; one feature holds 88% of importance | post-arrival columns and the attribution columns; the author calls it "too accurate" and suggests pruning DEPARTURE_DELAY |
| manasichhibber / Flight Delay Predictions (178 votes) | the five attribution columns, DEPARTURE_DELAY | decision tree AUC 0.998 (4 delay bands) | the source the "0.9983 Accuracy" notebook copies |
| erezalon / regression tree | departure delay of delayed flights only; airport "score" = mean delay over all data | (no clear R²) | target encoding without a split |

R² ≥ 0.9 on this data comes with DEPARTURE_DELAY or a leak. Honest schedule-only work
lands at R² ~0.1 / AUC ~0.6, as ours did; the rotation features (inbound aircraft
delay) are what lift ours to R² 0.35 / AUC 0.85. Not like for like: a time split and a
15-min threshold are both harder (our temporal audit costs ~0.03).

## 2026-09-25 · rot1-*: which feature sets are worth it? (1% sample, all 7 models)

- **Question:** does the aircraft's previous leg help, and at which horizon? Does
  scheduled congestion? Do the pruned columns and the frequency encoder earn their place?
- **Command:** `PREFIX=rot1 SAMPLE=0.01 FOLDS=2 MODELS=all bash scripts/run_ablation.sh`
- **Sets:** predeparture_all, predeparture, predeparture_nofreq, predeparture_sched,
  predeparture_inbound_dep, predeparture_inbound, wheelsoff (reference).
  Congestion sets added after this run started; tested separately.
- **Signals before training (full train split, Pearson r with arrival delay / severe):**
  prev_arr_delay 0.56 / 0.48, prev_dep_delay 0.53 / 0.45, inbound_overrun (unclipped)
  0.26 / 0.22, leg_of_day 0.08 / 0.09; congestion columns ≤ 0.04.
- **Timing caveat:** the rotation sets ran ~3x slower than `predeparture` (32 to 41 min
  against 11 at 1%). Cause: the join left 64 partitions (shuffle default) where the
  read had 16, so every job ran 4x the tasks. Fixed afterwards (`8d36dce`, coalesce
  to defaultParallelism); rot1 timings for those sets are inflated, the metrics are
  not affected. `predeparture_all` and `wheelsoff` are slow for another reason: the
  Haversine Arrow UDF (~5x).
- **Result (best of the 7 models per metric, test split of the 1% sample):**

  | set | R² | AUC | AUC-PR |
  |---|---|---|---|
  | predeparture | 0.018 | 0.656 | 0.187 |
  | predeparture_all | 0.027 | 0.666 | 0.194 |
  | predeparture_nofreq | 0.022 | 0.654 | 0.183 |
  | predeparture_sched (+ schedule-only rotation) | 0.055 | 0.685 | 0.269 |
  | predeparture_inbound_dep (+ inbound departure delay) | 0.265 | 0.817 | 0.571 |
  | predeparture_inbound (+ inbound arrival delay, overrun) | **0.313** | **0.835** | **0.623** |
  | rule: straight line on prev_arr_delay | 0.209 | 0.756 | 0.477 |
  | wheelsoff (reference) | 0.930 | 0.979 | 0.923 |
  | rule: straight line on DEPARTURE_DELAY | 0.894 | 0.959 | 0.897 |

  Linear regression wins the wheels-off set (0.930 against 0.74 to 0.79 for the
  trees): the relationship is y ≈ x − 6, which depth-limited trees can only step.
- **Decided:**
  - The aircraft-rotation features go into the tournament: +0.29 R² and +0.18 AUC
    over `predeparture`, far beyond 1% noise. Horizon: the inbound aircraft has
    landed (or has an ETA). `inbound_dep` is the fallback if an earlier horizon is
    wanted (loses ~0.02 AUC).
  - The model adds +0.10 R² / +0.08 AUC over its one-line rule; the wheels-off
    model adds +0.04 R² over its rule. That is the answer to "it looks like a leak".
  - The pruned set ties `predeparture_all` (within ±0.01): the dropped columns carried
    nothing. The frequency encoder does not clearly help (`nofreq` ties); to be
    confirmed at 5% before dropping it.
  - Pending: `inbound_lean` (Q3) against `inbound`.

## 2026-09-26 · Scheduled congestion (rot1-*congestion) and GLM link (glm1-*), 1%

- **Congestion adds nothing:** every set ties its congestion twin within ±0.01
  (GBT AUC: predeparture 0.656 vs 0.657; sched RF R² 0.055 vs 0.062; inbound GBT
  AUC 0.835 vs 0.836). Matches the EDA (|r| ≤ 0.04, and r 0.90 with the airport
  frequency). **Decided: dropped.**
- **GLM:** on `predeparture_inbound`, Gaussian/identity R² 0.267 against
  Poisson/log 0.191 (−0.08); on `predeparture` both ~0.02. Gaussian/identity equals
  LinearRegression (0.2670 vs 0.2666): it is least squares fitted by IRLS. The log
  link models a multiplicative response; delay adds up. Decision for the
  tournament: open (keep Poisson as the brief asks and report this, or swap).
