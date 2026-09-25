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
- **Result:** _pending_
- **Decided:** _pending_
