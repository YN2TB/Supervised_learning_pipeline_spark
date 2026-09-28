# Feature decisions: `predeparture_final`

Every column the tournament model uses, and every candidate that was dropped, with the
evidence. Built on 2026-09-26 from existing artifacts only (nothing re-run):

- **r**: Pearson correlation with arrival delay (min) / with the severe label (> 30 min),
  measured on a 20% sample of all curated flights with the rotation columns joined
  (2026-09-25 pair check). prev_air_gain was added later and has no measured r.
- **max |r| with a kept feature**: the strongest correlation with another feature in the
  final set, from the same check and `docs/benchmarks/feature_correlation_*.json`.
- **share**: the feature's share of total |importance| in the full tournament, no-PCA arms
  (`docs/benchmarks/flight-delay-predeparture/importance_*__nopca.json`). Trees report
  Gini gain; linear models report |coefficient| on the scaled feature. "trees" is GBT and
  RF (regressor, classifier); "linear" is LinearRegression, GLM Poisson, LinearSVC.
- **LR sign**: sign of the LinearRegression coefficient (effect with every other feature held
  fixed). A sign opposite to r is flagged; see "Anomalies" below.

## Kept (12 numeric + 4 one-hot groups)

| feature | meaning | known | r delay / severe | max \|r\| with a kept feature | share, trees | share, linear | LR sign | why kept |
|---|---|---|---|---|---|---|---|---|
| inbound_overrun | max(0, prev_arr_delay − turn_slack): minutes the inbound lands after this flight should leave | inbound landed / ETA | +0.466 / +0.370 | 0.66 (prev_arr_delay) | **25–52%** (1st in all four) | 12–20% | + | the delay that actually propagates; a hinge a linear model cannot build itself |
| prev_arr_delay | arrival delay of this aircraft's previous leg | inbound landed / ETA | **+0.557 / +0.477** | 0.66 (inbound_overrun) | 9–37% | 14–18% | + | strongest single signal; one-line rule on it alone: R² 0.20, AUC 0.76 |
| prev_air_gain | prev_arr_delay − prev_dep_delay: time the inbound made up or lost in the air | inbound landed | not measured | not measured | 0.3–5% | 0.1–7% | + | replaces prev_dep_delay (r 0.93 with prev_arr_delay); see anomaly 4 |
| turn_slack | scheduled ground time between inbound arrival and this departure | schedule | −0.027 / −0.019 | < 0.3 | 4–7% | 0.2–2% | − | more buffer, less delay (sign as expected) |
| leg_of_day | 1st, 2nd, … flight of this aircraft today | schedule | +0.079 / +0.087 | 0.62 (has_prev), 0.55 (SCHED_ARR_MIN) | 2–10% | 0.1–2% | − (≈0) | delays accumulate over the day; see anomaly 2 |
| has_prev | 1 if there is a linked previous leg today | schedule | +0.027 / +0.048 | 0.62 (leg_of_day) | 2–4% | 3–12% | **−** | marks prev_* on a first leg as an imputed median; see anomaly 1 |
| SCHED_ARR_MIN | scheduled arrival, minutes after midnight | schedule | +0.088 / +0.104 | 0.70 (DEP_HOUR), 0.55 (leg_of_day) | 1–7% | 1–3% | + | arrival bank, differs from departure hour on long and overnight legs |
| log_distance | log(1 + distance) | schedule | −0.020 / +0.004 | < 0.3 | 0.3–3% | 0.2–1% | − | one distance measure; see anomaly 3 |
| ORIGIN_LAT, ORIGIN_LON, DEST_LAT, DEST_LON | airport coordinates | schedule | all \|r\| ≤ 0.03 | 0.59 (ORIGIN_LON ~ DEST_LON) | 0.1–4% each | 0–1.4% | LON: origin −, dest + | airport identity for the trees (they replaced the frequency encoder with no loss); see anomaly 5 |
| DEP_HOUR (one-hot, 24) | scheduled departure hour | schedule | severe rate 3.9% (5 am) to 16.9% (8 pm) | 0.70 (SCHED_ARR_MIN) | 1–2% | **19–41%** | by hour | the strongest schedule effect; linear models need it, trees use SCHED_ARR_MIN instead |
| MONTH (one-hot, 12) | month | schedule | 6.7% (Oct) to 14.9% (Jun) | low | 0.4–8% | 6–13% | by month | seasonal pattern |
| AIRLINE (one-hot, 14) | carrier | schedule | 4.1% (HA) to 19.3% (NK) | low | 1–6% | 7–12% | by carrier | carrier effect |
| DAY_OF_WEEK (one-hot, 7) | weekday | schedule | 9.4% to 12.2% | 0.78 (IS_WEEKEND, dropped) | 0.1–2% | 1–4% | by day | weak but seven cheap slots |

Target and label: `label_delay` (arrival delay in minutes) for the regressors and the GLM
(shifted by +88 for the log link); `label_severe` = arrival delay > 30 min for the
classifiers, with balanced class weights. No feature is computed from either label.

## Decided 2026-09-26: `predeparture_v2`

The team dropped the two columns that showed no measurable contribution, log_distance
and prev_air_gain (tests below). `predeparture_v2` is the new default: the kept rows
above minus those two, 10 numeric columns + the four one-hot groups; prev_* on a first
leg stays median-imputed (zero fill made no difference; `first_leg_zero=True` switches
it). The two removals were tested one at a time, not together, so confirm v2 at 1%
before the next tournament. The next tournament waits for the team's visual EDA review.

## Dropped

| candidate | why | evidence |
|---|---|---|
| DEPARTURE_DELAY, TAXI_OUT, WHEELS_OFF (and clipped/derived forms) | only known after pushback | forbidden in code (`PRE_DEPARTURE_FORBIDDEN`); alone they give R² 0.89 (`docs/benchmarks/departure_delay_alone.ipynb`) |
| te_origin, te_dest, te_route (target encoding) | built from the label | removed on review; the 0/1 label is the classification target only |
| freq_origin, freq_dest, freq_route (frequency encoding) | adds nothing the coordinates do not | `nofreq` ties `predeparture`; `final` ties `inbound_gain` (≤ 0.005) |
| prev_dep_delay | r 0.93 with prev_arr_delay | replaced by prev_air_gain; `inbound_gain` ties `inbound` |
| prev_air_gain | no measurable contribution | `final_nogain` ties `final` (±0.009, no consistent direction) |
| log_distance | no measurable contribution | `final_nodist` ties `final` (max −0.010) |
| SCHED_DEP_MIN | r 0.998 with DEP_HOUR | |
| SCHEDULED_TIME / log_sched_time | r 0.97–0.98 with distance | |
| GC_DISTANCE_MI / log_gc_distance | r 1.00 with log_distance | also needed the Haversine Arrow UDF (~5x slower) |
| SCHEDULE_SPEED_MPH | r 0.94 with log_distance | no outliers worth a stage either (12 data-error rows above 550 mph, up to 902; 99.99% below 525) |
| ROUTE_DETOUR | \|r\| < 0.01, ratio pinned near 1 | |
| IS_WEEKEND | a function of DAY_OF_WEEK (r 0.78) | |
| DAY (of month) | \|r\| < 0.01, no pattern across deciles | |
| sched_origin_hour, sched_dest_hour, origin_day_ratio (congestion) | r ≤ 0.04, and 0.90 with the airport frequency | congestion sets tie their twins within ±0.01 |

## Anomalies worth a question

1. **has_prev: positive r, negative coefficient.** On its own, a flight with a previous leg
   is later (+0.027): first legs leave early in the morning, the least delayed time. With
   everything else held fixed the LinearRegression coefficient is strongly negative
   (−5.4 on the scaled feature, the third largest among the numeric features). Cause: on a first leg, prev_arr_delay,
   inbound_overrun, prev_air_gain and turn_slack are filled with training medians, so
   has_prev ends up correcting those placeholder values rather than meaning "has an
   inbound". The coefficient is not interpretable on its own. Common-sense fix, not yet
   tested: fill the prev_* of a first leg with 0 (no inbound, so no inbound delay and no
   overrun) instead of the median, then has_prev keeps its plain meaning.
2. **leg_of_day: positive r, coefficient ≈ 0 (slightly negative).** The time-of-day effect
   it carries is already in DEP_HOUR and SCHED_ARR_MIN (r 0.55–0.70). The trees still use
   it (GBT regressor 10%), the linear models do not. Not a problem, a duplication the
   linear models resolve by ignoring it.
3. **log_distance: r −0.020 with delay but +0.004 with severe.** The two targets disagree on
   the sign, so the marginal relationship is noise-level. The negative LR coefficient has
   a plausible reading (longer legs have more schedule padding to recover in the air) but
   importance is ≤ 3% everywhere. Candidate to drop; untested.
4. **prev_air_gain's own value is not proven.** `inbound_gain` recovered what `lean` lost,
   but `lean` also dropped has_prev, which `inbound_gain` restored: the two were changed
   together, so the recovery may be has_prev's. Its importance is small in the
   classifiers (0.1–0.3%). Test to settle it: `final` without prev_air_gain.
5. **ORIGIN_LON negative, DEST_LON positive in the linear model.** The two are correlated
   (0.59) and the coefficients nearly cancel (−0.88, +0.85), i.e. the model reads their
   difference (east- or westbound) rather than either coordinate. Coordinates are there
   for the trees, which use them as airport identity; for the linear models their
   coefficients should not be read one by one.
6. **inbound_overrun ranks above prev_arr_delay despite a lower r (0.47 vs 0.56).**
   Expected: r measures a straight line, and overrun is the hinge (zero until the buffer
   is used up) that the trees split on.
7. **DEP_HOUR is 19–41% of the linear models but 1–2% of the trees.** The trees take time
   of day from SCHED_ARR_MIN (one split instead of 24 slots); the linear models need the
   one-hot to bend. Both read the same effect.

### Tested (2026-09-26 afternoon, 1%, one change each; see LOG.md)

- **Anomaly 1, zero fill:** no metric change; the has_prev coefficient shrinks ~20% and
  stays negative. Revised explanation: has_prev is the intercept of the flights that have
  an inbound, offsetting the average contribution of the prev_* terms for them. Not a
  standalone effect; the marginal sign is the readable one.
- **Anomaly 3, log_distance:** removing it changes nothing measurable (max −0.010).
- **Anomaly 4, prev_air_gain:** removing it changes nothing measurable, in no consistent
  direction. The earlier recovery was has_prev's.
- **Open:** drop both (the tie rule says so) and zero-fill; the two removals were not
  tested together.

### Tested (2026-09-28, Kaggle 1%, one change each; see LOG.md runs C to F)

- **Coordinates:** keep (the trees lose up to −0.009 AUC without them).
- **ROUTE_DETOUR:** stays out (the lasso zeroes it; the trees do not improve).
- **Month:** keep (all seven models lose a little without it).
- **Weekday:** no measurable contribution, as one-hot or as a sin/cos pair. Drop candidate.
- **Time of day:** keep both copies. Without either, every model loses (up to −0.019
  AUC). DEP_HOUR serves the linear models, SCHED_ARR_MIN the trees; adding
  SCHED_DEP_MIN changes nothing.
- **Cyclical hour:** one sin/cos pair ties the 24-slot one-hot for every model, linear
  included (run G). A free simplification; the team's call.
- **Cyclical month:** a tie leaning slightly worse than its one-hot (run G). Keep one-hot.
