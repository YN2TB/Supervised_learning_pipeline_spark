"""Phase 3: end-to-end PySpark ML pipeline, model tournament and MLflow tracking.

    python mllib_pipeline.py --tune-fraction 0.01 --folds 3   # fast shakeout
    python mllib_pipeline.py                                  # full run

The pipeline is built once and reused by every model, so all five algorithms
see byte-identical preprocessing and the same 80/20 split. Where a choice was
made rather than inherited from the brief, the reasoning sits next to the code.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import mlflow
import mlflow.pyspark.ml
import mlflow.spark
from mlflow.tracking import MlflowClient
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, LinearSVC, RandomForestClassifier
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                   MulticlassClassificationEvaluator,
                                   RegressionEvaluator)
from pyspark.ml.feature import (Imputer, OneHotEncoder, SQLTransformer,
                                StandardScaler, StringIndexer, VarianceThresholdSelector,
                                VectorAssembler)
from pyspark.ml.regression import (GBTRegressor, GeneralizedLinearRegression,
                                   LinearRegression, RandomForestRegressor)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import custom_transformers  # noqa: E402  (for release_caches between arms)
import model_explain  # noqa: E402
from custom_transformers import (  # noqa: E402
    FrequencyEncoder, HaversineTransformer, MaterializeCache, OutlierIQRTruncator,
    RowMatrixPCA, SignedLog1pTransformer, TargetEncoder,
)
from flight_schema import SEVERE_DELAY_THRESHOLD  # noqa: E402
from spark_session import TRACKING_URI, build_spark, path, results_dir  # noqa: E402

CURATED = path("data", "parquet", "flights_curated")
MODEL_DIR = path("models")
BENCH_DIR = path("docs", "benchmarks")
REGISTERED_MODEL = "flight_delay_pipeline"
SPLIT_SEED = 42

REG_LABEL = "label_delay"
SEVERE_THRESHOLD_MIN = SEVERE_DELAY_THRESHOLD
CLF_LABEL = "label_severe"
WEIGHT_COL = "class_weight"

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Only the volatile operational columns get Tukey-clipped. DISTANCE is
# deliberately left alone: its IQR fence lands at ~2,106 mi, which would fold
# every transcontinental flight into a single value and destroy a genuine
# signal. Its skew is handled by the log transform instead.
CLIP_COLS = ["DEPARTURE_DELAY", "TAXI_OUT"]
CLIP_OUT = ["dep_delay_clip", "taxi_out_clip"]

# DEPARTURE_DELAY is passed through RAW as well (see NUMERIC_FEATURES), and that
# is not an oversight - it is the single most important feature decision here.
# Arrival delay is very nearly linear in departure delay (corr 0.947, slope ~1),
# so both of the "tidy" treatments destroy signal for a linear model: Tukey
# clipping to [-23, 25] min truncates 12.8% of flights that average +72.9 min of
# arrival delay, and the signed log breaks the linearity outright. Measured on a
# 2% sample with LinearRegression:
#
#     clip only .......... R2 0.390
#     log only ........... R2 0.409
#     clip + log ......... R2 0.409
#     raw ................ R2 0.937
#     raw + clip + log ... R2 0.937
#
# So the raw column carries essentially all of it, and the derived views add
# nothing measurable. dep_delay_clip is kept because a bounded view is still
# useful to the linear models and it exercises the IQR truncator on a real
# column; the log view is dropped as it earned +0.0001.
LOG_COLS = ["DISTANCE", "SCHEDULED_TIME", "GC_DISTANCE_MI"]
LOG_OUT = ["log_distance", "log_sched_time", "log_gc_distance"]

TARGET_ENC_COLS = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]
TARGET_ENC_OUT = ["te_origin", "te_dest", "te_route"]

WHEELSOFF_NUMERIC = (
    CLIP_OUT + LOG_OUT + TARGET_ENC_OUT
    + ["DEPARTURE_DELAY",  # raw and untransformed - see the note above
       "SCHEDULE_SPEED_MPH", "ROUTE_DETOUR", "SCHED_DEP_MIN", "SCHED_ARR_MIN",
       "WHEELS_OFF_MIN", "IS_WEEKEND", "DAY", "ORIGIN_LAT", "ORIGIN_LON",
       "DEST_LAT", "DEST_LON"]
)
# Imputed from training medians. SCHEDULE_SPEED_MPH is the one that really
# needs it (it divides by scheduled block time), but a fitted median for every
# numeric column also keeps a single malformed streaming record from producing
# a NaN feature vector at inference time.
IMPUTE_COLS = ["SCHEDULE_SPEED_MPH", "ROUTE_DETOUR"]

ONEHOT_NUMERIC = ["MONTH", "DAY_OF_WEEK", "DEP_HOUR"]

# ---------------------------------------------------------------------------
# Pre-departure feature sets (the current model)
# ---------------------------------------------------------------------------
# The wheels-off model above scored R2 0.93, and 0.89 of that is one straight
# line on DEPARTURE_DELAY (r = 0.9445): legal at wheels-off, but it predicts a
# delay from a delay that has already happened. The useful question is the one
# a passenger or a dispatcher asks before the aircraft leaves the gate, so
# nothing observed at or after pushback may enter the vector.
PRE_DEPARTURE_FORBIDDEN = [
    "DEPARTURE_DELAY", "dep_delay_clip", "DEPARTURE_TIME",
    "TAXI_OUT", "taxi_out_clip", "WHEELS_OFF", "WHEELS_OFF_MIN",
    "te_origin", "te_dest", "te_route",      # label means, whatever the horizon
    REG_LABEL, CLF_LABEL, "label_glm", WEIGHT_COL,
]

# Airport busyness, target-free. See custom_transformers.FrequencyEncoder.
FREQ_ENC_COLS = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE"]
FREQ_ENC_OUT = ["freq_origin", "freq_dest", "freq_route"]

# Every plausible pre-departure column: the ablation's upper reference.
PREDEP_ALL_NUMERIC = [
    "log_distance", "log_sched_time", "log_gc_distance", "SCHEDULE_SPEED_MPH",
    "ROUTE_DETOUR", "SCHED_DEP_MIN", "SCHED_ARR_MIN", "IS_WEEKEND", "DAY",
    "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON",
] + FREQ_ENC_OUT

# The pruned set, from the EDA on the training split (notebook section 11):
#   dropped as duplicates   log_sched_time (r 0.97 with log_distance),
#                           log_gc_distance (1.00), SCHEDULE_SPEED_MPH (0.94),
#                           SCHED_DEP_MIN (0.998 with the DEP_HOUR one-hot),
#                           IS_WEEKEND (a function of DAY_OF_WEEK)
#   dropped as empty        ROUTE_DETOUR, DAY (|r| < 0.01 with both targets and
#                           no mechanism), freq_route (flat across deciles)
#   kept                    log_distance, SCHED_ARR_MIN (the arrival bank, not
#                           the same as the departure hour), the four
#                           coordinates (geography, for the trees), freq_origin,
#                           freq_dest; plus the one-hot AIRLINE, MONTH,
#                           DAY_OF_WEEK and DEP_HOUR, which carry most of the
#                           signal (severe-delay rate 3.9% to 16.9% by hour,
#                           4.1% to 19.3% by airline, 6.7% to 14.9% by month,
#                           over all curated flights).
PREDEP_NUMERIC = [
    "log_distance", "SCHED_ARR_MIN",
    "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON",
    "freq_origin", "freq_dest",
]

# ---------------------------------------------------------------------------
# Aircraft rotation: what the plane did before this flight
# ---------------------------------------------------------------------------
# The same aircraft (TAIL_NUMBER) flies several legs a day, and a late inbound
# aircraft makes the next departure late. These features look at the previous
# leg of the same aircraft on the same day, grouped by WHEN they become known:
#
#   schedule   leg_of_day, has_prev, turn_slack (scheduled minutes on the ground
#              between the inbound arrival and this departure, both local times
#              at the same airport). Known when the schedule is published.
#   inbound departed    + prev_dep_delay. Known once the inbound leg has left.
#   inbound arrived     + prev_arr_delay, inbound_overrun (= max(0,
#              prev_arr_delay - turn_slack): minutes the inbound lands past the
#              moment this flight should leave).
#              Known at the gate, once the inbound has landed or has an ETA.
#
# They are another flight's observables, never this flight's own outcome. A
# previous leg counts only if it landed where this one departs; otherwise the
# chain is broken (a cancelled or diverted leg was removed) and the features
# are left empty for the Imputer, with has_prev = 0.
FLIGHT_KEY = ["AIRLINE", "FLIGHT_NUMBER", "MONTH", "DAY", "ORIGIN_AIRPORT", "SCHED_DEP_MIN"]
ROT_SCHEDULE = ["leg_of_day", "has_prev", "turn_slack"]
ROT_INBOUND_DEP = ["prev_dep_delay"]
ROT_INBOUND_ARR = ["prev_arr_delay", "inbound_overrun"]
# prev_arr_delay - prev_dep_delay: what the inbound leg made up (negative) or lost
# in the air. Carries what prev_dep_delay adds over prev_arr_delay without being
# a near-copy of it (r 0.93 between those two).
ROT_AIR_GAIN = ["prev_air_gain"]
ROTATION_COLS = ROT_SCHEDULE + ROT_INBOUND_DEP + ROT_INBOUND_ARR + ROT_AIR_GAIN


def rotation_table(df):
    """One row per flight (FLIGHT_KEY is unique) with the rotation columns.

    Computed over every curated flight, so a sampled or split frame still sees
    its true previous leg. Joined on after the split, so the split is unchanged.
    """
    from pyspark.sql import Window
    w = (Window.partitionBy("TAIL_NUMBER", "MONTH", "DAY")
         .orderBy("SCHED_DEP_MIN", "FLIGHT_NUMBER"))       # tie-break: 595 dup minutes
    linked = F.lag("DESTINATION_AIRPORT").over(w) == F.col("ORIGIN_AIRPORT")
    slack = (F.col("SCHED_DEP_MIN") - F.lag("SCHED_ARR_MIN").over(w)).cast("double")
    prev_arr = F.lag(REG_LABEL).over(w).cast("double")
    return (df.select(*FLIGHT_KEY, "TAIL_NUMBER", "DESTINATION_AIRPORT", "SCHED_ARR_MIN",
                      "DEPARTURE_DELAY", REG_LABEL)
            .withColumn("leg_of_day", F.row_number().over(w).cast("double"))
            .withColumn("_linked", F.coalesce(linked, F.lit(False)))
            .withColumn("has_prev", F.col("_linked").cast("double"))
            .withColumn("turn_slack", F.when(F.col("_linked"), slack))
            .withColumn("prev_dep_delay",
                        F.when(F.col("_linked"), F.lag("DEPARTURE_DELAY").over(w).cast("double")))
            .withColumn("prev_arr_delay", F.when(F.col("_linked"), prev_arr))
            .withColumn("prev_air_gain", F.col("prev_arr_delay") - F.col("prev_dep_delay"))
            # Only the part past the ground time propagates; slack left over does not.
            .withColumn("inbound_overrun",
                        F.greatest(F.col("prev_arr_delay") - F.col("turn_slack"), F.lit(0.0)))
            .select(*FLIGHT_KEY, *ROTATION_COLS))


# ---------------------------------------------------------------------------
# Scheduled congestion: how busy the airports are meant to be
# ---------------------------------------------------------------------------
# Counted from the PUBLISHED schedule, i.e. flights_raw with cancelled and
# diverted flights included. Counting the curated table instead would leak:
# a stormy day cancels flights, so a low count there means bad weather, which
# the schedule could not have known.
#
#   sched_origin_hour   log1p(departures scheduled at the origin that hour)
#   sched_dest_hour     log1p(arrivals scheduled at the destination that hour)
#   origin_day_ratio    departures scheduled at the origin that day over its
#                       average day: >1 on peak days such as holidays
CONGESTION_COLS = ["sched_origin_hour", "sched_dest_hour", "origin_day_ratio"]
RAW = path("data", "parquet", "flights_raw")


def congestion_tables(spark):
    """Three small lookup tables keyed on airport and date (and hour)."""
    from data_prep import _repair_airport_codes, hhmm_to_minutes
    raw = _repair_airport_codes(spark, spark.read.parquet(RAW))
    sched = (raw.withColumn("_dh", (hhmm_to_minutes("SCHEDULED_DEPARTURE") / 60).cast("int"))
                .withColumn("_ah", (hhmm_to_minutes("SCHEDULED_ARRIVAL") / 60).cast("int")))
    origin_hour = (sched.groupBy("ORIGIN_AIRPORT", "MONTH", "DAY", F.col("_dh").alias("DEP_HOUR"))
                   .agg(F.log1p(F.count(F.lit(1))).alias("sched_origin_hour")))
    dest_hour = (sched.groupBy("DESTINATION_AIRPORT", "MONTH", "DAY", F.col("_ah").alias("_arr_hour"))
                 .agg(F.log1p(F.count(F.lit(1))).alias("sched_dest_hour")))
    per_day = sched.groupBy("ORIGIN_AIRPORT", "MONTH", "DAY").agg(F.count(F.lit(1)).alias("_n"))
    from pyspark.sql import Window
    origin_day = (per_day.withColumn("origin_day_ratio",
                                     F.col("_n") / F.avg("_n").over(Window.partitionBy("ORIGIN_AIRPORT")))
                  .drop("_n"))
    return origin_hour, dest_hour, origin_day


def add_context_features(spark, full_df, frame, numeric):
    """Join the row-external features a feature set needs onto ``frame``.

    Both kinds look at OTHER flights (the aircraft's previous leg, the airport's
    schedule), so they cannot be pipeline stages: a stage sees one row, and a
    streaming micro-batch does not hold the rest of the day. They are lookups,
    computed once from the full data and joined by key. A left join keeps every
    row of ``frame``, so a train/test split made before this call is unchanged.
    """
    need = set(numeric)
    joined = bool(need & (set(ROTATION_COLS) | set(CONGESTION_COLS)))
    if need & set(ROTATION_COLS):
        frame = frame.join(rotation_table(full_df), FLIGHT_KEY, "left")
    if need & set(CONGESTION_COLS):
        origin_hour, dest_hour, origin_day = congestion_tables(spark)
        frame = (frame.withColumn("_arr_hour", (F.col("SCHED_ARR_MIN") / 60).cast("int"))
                 .join(origin_hour, ["ORIGIN_AIRPORT", "MONTH", "DAY", "DEP_HOUR"], "left")
                 .join(dest_hour, ["DESTINATION_AIRPORT", "MONTH", "DAY", "_arr_hour"], "left")
                 .join(origin_day, ["ORIGIN_AIRPORT", "MONTH", "DAY"], "left")
                 .drop("_arr_hour"))
    if joined:
        # A join leaves spark.sql.shuffle.partitions (64) partitions where the
        # read had one per core (16). Every later job then runs 4x the tasks,
        # each tiny: the rotation ablation sets took 3x as long as the others
        # at 1%. coalesce only merges partitions; no row moves between halves.
        frame = frame.coalesce(spark.sparkContext.defaultParallelism)
    return frame


FEATURE_SETS = {
    # The September tournament's features, kept to reproduce it as a reference.
    "wheelsoff": dict(numeric=WHEELSOFF_NUMERIC, encoder="target"),
    "predeparture_all": dict(numeric=PREDEP_ALL_NUMERIC, encoder="frequency"),
    "predeparture": dict(numeric=PREDEP_NUMERIC, encoder="frequency"),
    # Ablation: does the frequency encoder earn its stage?
    "predeparture_nofreq": dict(numeric=[c for c in PREDEP_NUMERIC
                                         if c not in FREQ_ENC_OUT], encoder="frequency"),
    # Aircraft rotation, by horizon (see ROTATION_COLS above).
    "predeparture_sched": dict(numeric=PREDEP_NUMERIC + ROT_SCHEDULE, encoder="frequency"),
    "predeparture_inbound_dep": dict(numeric=PREDEP_NUMERIC + ROT_SCHEDULE + ROT_INBOUND_DEP,
                                     encoder="frequency"),
    "predeparture_inbound": dict(
        numeric=PREDEP_NUMERIC + ROT_SCHEDULE + ROT_INBOUND_DEP + ROT_INBOUND_ARR,
        encoder="frequency"),
    # One change against predeparture_inbound: prev_dep_delay -> prev_air_gain.
    "predeparture_inbound_gain": dict(
        numeric=PREDEP_NUMERIC + ROT_SCHEDULE + ["prev_arr_delay", "prev_air_gain",
                                                 "inbound_overrun"],
        encoder="frequency"),
    # The tournament candidate: inbound_gain without the frequency encoder, which
    # tied its absence (the coordinates already identify the airport). has_prev
    # stays: it tells the model that prev_* on a first leg is an imputed median.
    "predeparture_final": dict(
        numeric=[c for c in PREDEP_NUMERIC if c not in FREQ_ENC_OUT]
        + ROT_SCHEDULE + ["prev_arr_delay", "prev_air_gain", "inbound_overrun"],
        encoder="frequency"),
    # The same without the near-duplicates: prev_dep_delay (r 0.93 with
    # prev_arr_delay, which supersedes it once the inbound has landed) and
    # has_prev (r 0.62 with leg_of_day, nearly "not the first leg").
    "predeparture_inbound_lean": dict(
        numeric=PREDEP_NUMERIC + ["leg_of_day", "turn_slack", "prev_arr_delay", "inbound_overrun"],
        encoder="frequency"),
    # Scheduled congestion, alone and on top of the full rotation set.
    "predeparture_congestion": dict(numeric=PREDEP_NUMERIC + CONGESTION_COLS,
                                    encoder="frequency"),
    "predeparture_inbound_congestion": dict(
        numeric=PREDEP_NUMERIC + ROT_SCHEDULE + ROT_INBOUND_DEP + ROT_INBOUND_ARR + CONGESTION_COLS,
        encoder="frequency"),
    "predeparture_sched_congestion": dict(
        numeric=PREDEP_NUMERIC + ROT_SCHEDULE + CONGESTION_COLS, encoder="frequency"),
}
# Chosen by the trials in experiments/LOG.md (2026-09-26): pre-departure,
# aircraft rotation with prev_air_gain, no frequency encoder, no congestion.
DEFAULT_FEATURE_SET = "predeparture_final"

# Kept for the scripts that import it; it names the current feature set.
NUMERIC_FEATURES = FEATURE_SETS[DEFAULT_FEATURE_SET]["numeric"]


def _predeparture_stages(numeric: list) -> list:
    """Stages up to the one-hot encoders for a pre-departure feature set.

    Deliberately short. No outlier clip: the pre-departure numerics have no
    outliers worth the name (implied schedule speed tops out at 543 mph,
    distance is a real quantity whose long tail is transcontinental flights).
    No target encoding: nothing here is computed from the label.
    """
    stages = []
    derived = {"log_gc_distance", "SCHEDULE_SPEED_MPH", "ROUTE_DETOUR"}
    if derived & set(numeric):
        stages.append(HaversineTransformer())
    log_pairs = [(src, dst) for src, dst in zip(LOG_COLS, LOG_OUT) if dst in numeric]
    if log_pairs:
        stages.append(SignedLog1pTransformer(inputCols=[s for s, _ in log_pairs],
                                             outputCols=[d for _, d in log_pairs]))
    freq_pairs = [(src, dst) for src, dst in zip(FREQ_ENC_COLS, FREQ_ENC_OUT)
                  if dst in numeric]
    if freq_pairs:
        stages.append(FrequencyEncoder(inputCols=[s for s, _ in freq_pairs],
                                       outputCols=[d for _, d in freq_pairs]))
    # A training median for every numeric input, in place, so one malformed
    # streaming record gets a typical value instead of turning the whole vector
    # into NaN (and the prediction into a constant).
    stages.append(Imputer(inputCols=numeric, outputCols=numeric, strategy="median"))
    return stages


def build_feature_stages(label_col: str, use_pca: bool, pca_k: int,
                         svd_mode: str = "local-eigs",
                         feature_set: str = DEFAULT_FEATURE_SET,
                         pca_variance: float = 0.0):
    """Shared preprocessing. Returns (stages, assembler_input_names)."""
    # svd_mode must stay in step with the --svd-mode CLI default. They disagreed
    # once, and the only caller that omits the argument is
    # scripts/profile_stages.py, which therefore profiled a mode the pipeline
    # itself never ran.
    spec = FEATURE_SETS[feature_set]
    numeric = list(spec["numeric"])
    stages = []

    if spec["encoder"] == "frequency":
        leaked = [c for c in numeric if c in PRE_DEPARTURE_FORBIDDEN]
        assert not leaked, f"pre-departure feature set uses {leaked}"
        stages += _predeparture_stages(numeric)
    else:
        stages.append(OutlierIQRTruncator(inputCols=CLIP_COLS, outputCols=CLIP_OUT,
                                          iqrMultiplier=1.5))
        stages.append(HaversineTransformer())
        stages.append(SignedLog1pTransformer(inputCols=LOG_COLS, outputCols=LOG_OUT))
        stages.append(Imputer(inputCols=IMPUTE_COLS, outputCols=IMPUTE_COLS,
                              strategy="median"))
        # Learned inside _fit, so a Pipeline fitted on the training split (and
        # refitted per CV fold) can never see a held-out target.
        stages.append(TargetEncoder(inputCols=TARGET_ENC_COLS, outputCols=TARGET_ENC_OUT,
                                    labelCol=label_col, smoothing=20.0))

    # AIRLINE is a string; the rest are already small integer codes.
    stages.append(StringIndexer(inputCol="AIRLINE", outputCol="airline_idx",
                                handleInvalid="keep"))
    ohe_in = ["airline_idx"] + ONEHOT_NUMERIC
    ohe_out = [f"{c}_ohe" for c in ohe_in]
    stages.append(OneHotEncoder(inputCols=ohe_in, outputCols=ohe_out,
                                handleInvalid="keep", dropLast=True))

    # The upstream fits (encoders, imputer, indexer) are behind us; cache here so StandardScaler, PCA and the estimator do not
    # each re-derive it. See MaterializeCache for the measurement.
    stages.append(MaterializeCache())

    assembler_inputs = numeric + ohe_out
    if spec["encoder"] == "frequency":
        assert not set(assembler_inputs) & set(PRE_DEPARTURE_FORBIDDEN), assembler_inputs
    stages.append(VectorAssembler(inputCols=assembler_inputs,
                                  outputCol="features_raw", handleInvalid="keep"))

    # withMean=False is deliberate, not a default left untouched. The assembled
    # vector is mostly one-hot and therefore sparse; subtracting a mean would
    # make every zero non-zero and densify it. withStd rescales in place and
    # preserves sparsity. See docs/REPORT.md for the full argument.
    stages.append(StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                                 withMean=False, withStd=True))

    # Three of the 80 assembled slots are constant, for two different reasons.
    # (a) StringIndexer(handleInvalid="keep") reserves index 14 for an unseen
    #     AIRLINE; training contains all 14 carriers, so it never fires here.
    # (b) MONTH and DAY_OF_WEEK are 1-INDEXED, and OneHotEncoder sizes itself
    #     from the largest value it sees - a max of 12 means it assumes indices
    #     0..12, i.e. 13 slots for 12 months, reserving one for a month that
    #     cannot exist. Same for DAY_OF_WEEK (7 -> 8). DEP_HOUR is unaffected
    #     because hour 0 is real. Verified: slots 34, 35, 48.
    # Either way the column is constant, and StandardScaler maps a zero-variance
    # column to all zeros, which leaves the covariance matrix singular - and
    # Spark's PCA runs a Breeze SVD over exactly that matrix, so it fails
    # outright with NotConvergedException rather than degrading. Dropping
    # zero-variance columns first is both the fix and the right thing to do:
    # a constant feature carries no information.
    stages.append(VarianceThresholdSelector(
        featuresCol="features_scaled", outputCol="features_selected",
        varianceThreshold=0.0))

    if use_pca:
        # RowMatrixPCA, not spark.ml's PCA. The brief asks for PCA computed
        # "using RowMatrix SVD without collecting full covariance matrices to
        # the Driver node", and spark.ml's PCA does the opposite: it forms the
        # whole 77x77 covariance on the driver. RowMatrixPCA calls
        # RowMatrix.computeSVD with an explicit mode, so dist-eigs drives ARPACK
        # Lanczos on distributed A^T(Av) products and no n x n matrix is ever
        # materialised. See custom_transformers.RowMatrixPCA for the caveats -
        # in particular that computeSVD does not centre.
        # With pca_variance > 0, k is chosen from the data: the fewest
        # components that reach that share of the variance.
        stages.append(RowMatrixPCA(k=pca_k, svdMode=svd_mode,
                                   varianceThreshold=pca_variance,
                                   inputCol="features_selected",
                                   outputCol="features"))
    else:
        # Both arms must end with a column literally named "features" so that
        # a single estimator definition serves either one.
        stages.append(SQLTransformer(
            statement="SELECT *, features_selected AS features FROM __THIS__"))

    return stages, assembler_inputs


def resolve_feature_names(fitted_model, sample_df, use_pca: bool, pca_k: int):
    """True per-slot feature names for the fitted pipeline.

    The assembler's input list is not the feature list: one-hot columns expand
    to many vector slots each (24 input columns become 80 features here), and
    the variance selector then drops some. VectorAssembler records the expanded
    names as ML attribute metadata, so read them from there and re-index with
    the selector's own choices rather than trying to reconstruct them.
    """
    if use_pca:
        pca = next((s for s in fitted_model.stages if hasattr(s, "explainedVariance")), None)
        k = len(pca.explainedVariance) if pca is not None else pca_k
        return [f"PC{i + 1}" for i in range(k)]
    try:
        transformed = fitted_model.transform(sample_df.limit(10))
        # There is no AttributeGroup in the Python API (it is Scala-only), but
        # the same information is on the field as raw metadata: ml_attr.attrs
        # groups the slots into numeric/binary/nominal, each carrying its own
        # index and name.
        meta = transformed.schema["features_raw"].metadata.get("ml_attr", {})
        names = [""] * int(meta.get("num_attrs", 0))
        for group in meta.get("attrs", {}).values():
            for attr in group:
                names[attr["idx"]] = attr.get("name", f"f{attr['idx']}")
        selector = next((s for s in fitted_model.stages
                         if hasattr(s, "selectedFeatures")), None)
        if selector is not None and names:
            return [names[i] for i in selector.selectedFeatures]
        return names
    except Exception:  # noqa: BLE001 - names are cosmetic; never fail the run
        return []


# ---------------------------------------------------------------------------
# Model tournament definitions
# ---------------------------------------------------------------------------
# Defined in model_specs but left out of --models all; name them explicitly.
OPT_IN_MODELS = {"glm_gaussian_identity"}


def model_specs(args):
    """Each entry: estimator, grid, task, label column, primary metric."""
    grid = {}
    # Only 11% of flights are severely late. Unweighted, the classifiers learn
    # that "never late" is 89% accurate: before departure the signal is weak, and
    # LinearSVC predicted not-late for every test flight. Balanced weights (each
    # class carries half the total weight, from the training split) make a
    # missed delay cost as much as a false alarm. AUC is barely affected; F1 and
    # the confusion matrix become meaningful.
    clf_w = {"weightCol": WEIGHT_COL} if args.class_weight == "balanced" else {}

    lr = LinearRegression(featuresCol="features", labelCol=REG_LABEL,
                          elasticNetParam=0.5, regParam=0.1, maxIter=50)
    grid["linear_regression"] = dict(
        estimator=lr, task="regression", label=REG_LABEL,
        grid=(ParamGridBuilder()
              .addGrid(lr.regParam, [0.01, 0.1])
              .addGrid(lr.elasticNetParam, [0.0, 0.5])
              .build()))

    # A GLM with a log link needs a strictly positive response, but arrival
    # delay is negative for early flights. The label is shifted by a constant
    # computed on train; RMSE, MAE and R^2 are all invariant to a common shift
    # of prediction and target, so the numbers stay comparable to the other
    # regressors without any inverse transform.
    #
    # Poisson rather than Gamma, chosen after measurement. Gamma's variance
    # function is V(mu) = mu^2, and combined with the log link's exp(X.beta)
    # response it overflowed on extreme feature combinations: the full-data run
    # produced a sane MAE of 10.80 alongside an RMSE of 568.90 - a 53x ratio
    # where ~1.3x is normal - i.e. a handful of astronomically large
    # predictions dragging R^2 to -207. Poisson's V(mu) = mu grows far more
    # slowly, and the heavier regularisation grid below damps the linear
    # predictor further. The brief asks for "Gamma/Poisson family with log
    # link", so this stays within spec.
    glm = GeneralizedLinearRegression(featuresCol="features", labelCol="label_glm",
                                      family="poisson", link="log", maxIter=25,
                                      regParam=0.1)
    grid["glm_poisson_log"] = dict(
        estimator=glm, task="regression", label="label_glm",
        grid=(ParamGridBuilder().addGrid(glm.regParam, [0.1, 1.0]).build()))

    # The common-sense GLM for an additive target: Gaussian family, identity
    # link, i.e. least squares fitted by IRLS. The log link above models a
    # multiplicative response and has no reason to fit a delay in minutes.
    # Opt-in (not in --models all) until a trial shows it earns a place.
    glm_id = GeneralizedLinearRegression(featuresCol="features", labelCol=REG_LABEL,
                                         family="gaussian", link="identity",
                                         maxIter=25, regParam=0.01)
    grid["glm_gaussian_identity"] = dict(
        estimator=glm_id, task="regression", label=REG_LABEL,
        grid=(ParamGridBuilder().addGrid(glm_id.regParam, [0.01, 0.1]).build()))

    svc = LinearSVC(featuresCol="features", labelCol=CLF_LABEL, maxIter=30, **clf_w)
    grid["linear_svc"] = dict(
        estimator=svc, task="classification", label=CLF_LABEL,
        grid=(ParamGridBuilder()
              .addGrid(svc.regParam, [0.01, 0.1])
              .build()))

    rfr = RandomForestRegressor(featuresCol="features", labelCol=REG_LABEL,
                                numTrees=40, maxDepth=8, maxBins=64, seed=7)
    grid["random_forest_regressor"] = dict(
        estimator=rfr, task="regression", label=REG_LABEL,
        grid=(ParamGridBuilder()
              .addGrid(rfr.maxDepth, [6, 10])
              .addGrid(rfr.numTrees, [40, 80])
              .build()))

    rfc = RandomForestClassifier(featuresCol="features", labelCol=CLF_LABEL,
                                 numTrees=40, maxDepth=8, maxBins=64, seed=7, **clf_w)
    grid["random_forest_classifier"] = dict(
        estimator=rfc, task="classification", label=CLF_LABEL,
        grid=(ParamGridBuilder()
              .addGrid(rfc.maxDepth, [6, 10])
              .addGrid(rfc.numTrees, [40, 80])
              .build()))

    gbtr = GBTRegressor(featuresCol="features", labelCol=REG_LABEL,
                        maxIter=40, maxDepth=5, maxBins=64, seed=7)
    grid["gbt_regressor"] = dict(
        estimator=gbtr, task="regression", label=REG_LABEL,
        grid=(ParamGridBuilder()
              .addGrid(gbtr.maxDepth, [4, 6])
              .addGrid(gbtr.maxBins, [32, 64])
              .build()))

    gbtc = GBTClassifier(featuresCol="features", labelCol=CLF_LABEL,
                         maxIter=40, maxDepth=5, maxBins=64, seed=7, **clf_w)
    grid["gbt_classifier"] = dict(
        estimator=gbtc, task="classification", label=CLF_LABEL,
        grid=(ParamGridBuilder()
              .addGrid(gbtc.maxDepth, [4, 6])
              .build()))

    return grid


def make_split(df, sample_fraction: float = 1.0):
    """The single definition of the train/test split.

    Anything that needs the *same* test rows as the tournament - the residual
    plot, an ad-hoc error analysis - must call this rather than re-typing
    ``randomSplit``. Re-deriving it elsewhere looks harmless and is not: the
    optional sample runs BEFORE the split, so a caller that forgets it gets a
    different partition of the data and silently mixes training rows into what
    it calls the test set.
    """
    if sample_fraction < 1.0:
        df = df.sample(False, sample_fraction, seed=SPLIT_SEED)
    return df.randomSplit([0.8, 0.2], seed=SPLIT_SEED)


def evaluators(task: str, label: str):
    if task == "regression":
        return {m: RegressionEvaluator(labelCol=label, predictionCol="prediction",
                                       metricName=m)
                for m in ("rmse", "mae", "r2")}
    return {
        "areaUnderROC": BinaryClassificationEvaluator(
            labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderROC"),
        "areaUnderPR": BinaryClassificationEvaluator(
            labelCol=label, rawPredictionCol="rawPrediction", metricName="areaUnderPR"),
        "f1": MulticlassClassificationEvaluator(
            labelCol=label, predictionCol="prediction", metricName="f1"),
        "accuracy": MulticlassClassificationEvaluator(
            labelCol=label, predictionCol="prediction", metricName="accuracy"),
    }


# ---------------------------------------------------------------------------
# Fault tolerance for long runs
# ---------------------------------------------------------------------------
# A full tournament is 14 arms over several hours. Without these, a failure in
# arm 12 throws away everything: results were only written after the whole loop
# finished, and any exception propagated out and killed the run. Three
# independent protections, so a crash costs one arm rather than an afternoon.


def completed_arms(experiment: str) -> dict:
    """Arms already finished, read back from the tracking store.

    MLflow is the source of truth for what has completed - it is written as
    each arm ends, so it survives a crash that never reached the summary file.
    A run with no metrics was started but did not finish, and is not counted.
    """
    try:
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            return {}
        done = {}
        # Ascending start_time, so when an arm was run more than once - a
        # resume after an interrupted run leaves an orphan behind - the newest
        # row overwrites the older ones. Status is checked too: a run killed
        # between log_metrics and log_model has metrics but never finished,
        # and its saved model is truncated, so it must not count as complete.
        for r in client.search_runs([exp.experiment_id], max_results=1000,
                                    order_by=["attributes.start_time ASC"]):
            run_name = r.data.tags.get("mlflow.runName", "")
            if "__" not in run_name or not r.data.metrics:
                continue
            if r.info.status != "FINISHED":
                continue
            done[run_name] = dict(
                r.data.metrics,
                model=r.data.params.get("model", run_name.split("__")[0]),
                arm=r.data.params.get("arm", run_name.split("__")[-1]),
                task=r.data.params.get("task", ""),
                label=r.data.params.get("label", ""))
        return done
    except Exception as exc:  # noqa: BLE001 - never let bookkeeping stop a run
        print(f"  (could not read previous runs: {exc})")
        return {}


def checkpoint(results: dict, failures: dict, out_dir: str = BENCH_DIR) -> None:
    """Persist progress after every arm, not just at the end of the loop."""
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tournament_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        if failures:
            with open(os.path.join(out_dir, "tournament_failures.json"), "w") as fh:
                json.dump(failures, fh, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (checkpoint failed: {exc})")


# ---------------------------------------------------------------------------
# Safe pause
# ---------------------------------------------------------------------------
# A full tournament runs for hours on a laptop. To stop it without losing work:
#
#     touch .pause_tournament          (PowerShell: ni .pause_tournament)
#
# The run stops at the next safe point - before an arm starts, or right after an
# arm's cross-validation, whose result is saved first - and exits cleanly. Delete
# the file and rerun the SAME command to continue: finished arms are skipped
# (--resume, the default) and a checkpointed arm goes straight to its refit.
# Ctrl+C also stops cleanly, but loses the arm in progress.
PAUSE_FILE = path(".pause_tournament")


class SparkDied(BaseException):
    """The JVM behind the session is gone (typically an OutOfMemoryError)."""


def spark_alive(frame) -> bool:
    try:
        frame.sparkSession.range(1).count()
        return True
    except Exception:  # noqa: BLE001
        return False


class Paused(BaseException):
    """Raised at a safe point when PAUSE_FILE exists. BaseException, so the
    per-arm ``except Exception`` does not record it as a failure."""


def pause_requested() -> bool:
    return os.path.exists(PAUSE_FILE)


def cv_checkpoint_path(args, run_name: str) -> str:
    return os.path.join(args.out_dir, "cv_checkpoints", f"{run_name}.json")


def cv_checkpoint_key(args) -> dict:
    """What must match for a saved CV result to be reused."""
    return {"feature_set": args.feature_set, "sample_fraction": args.sample_fraction,
            "tune_fraction": args.tune_fraction, "folds": args.folds,
            "pca_variance": args.pca_variance, "pca_k": args.pca_k,
            "class_weight": args.class_weight, "experiment": args.experiment}


def load_cv_checkpoint(args, run_name: str):
    f = cv_checkpoint_path(args, run_name)
    if not (args.resume and os.path.exists(f)):
        return None
    with open(f) as fh:
        saved = json.load(fh)
    if saved.get("key") != cv_checkpoint_key(args):
        print(f"  (ignoring stale CV checkpoint for {run_name}: settings differ)")
        return None
    return saved


def save_cv_checkpoint(args, run_name: str, best_params: dict, primary: str,
                       value: float) -> None:
    f = cv_checkpoint_path(args, run_name)
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "w") as fh:
        json.dump({"key": cv_checkpoint_key(args), "best_params": best_params,
                   "primary": primary, "value": value}, fh, indent=2)


def log_baselines(train, test, n_test: int, args) -> dict:
    """The "no model" rows every result is read against.

    Regression: predict the training mean delay for every flight. Classification:
    predict "not severe" for every flight, with one constant score, so AUC is
    0.5 and AUC-PR is the positive rate. Logged as ordinary runs in the same
    experiment, so the results table and the MLflow UI show them next to the
    models. Cheap (one pass over test each), so they are recomputed every run.
    """
    out = {}
    mean_delay = float(train.agg(F.avg(REG_LABEL)).first()[0])
    severe_rate = float(train.agg(F.avg(CLF_LABEL)).first()[0])
    from pyspark.ml.functions import array_to_vector
    for task, label, preds, note in (
        ("regression", REG_LABEL,
         test.withColumn("prediction", F.lit(mean_delay)),
         f"predict the training mean, {mean_delay:.3f} min"),
        ("classification", CLF_LABEL,
         test.withColumn("prediction", F.lit(0.0))
             .withColumn("rawPrediction",
                         array_to_vector(F.array(F.lit(1.0), F.lit(0.0)))),
         f"predict not-severe for every flight (training rate {severe_rate:.4f})"),
    ):
        run_name = f"baseline_{task}__none"
        metrics = {m: float(e.evaluate(preds)) for m, e in evaluators(task, label).items()}
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"model": f"baseline_{task}", "arm": "none", "task": task,
                               "label": label, "feature_set": args.feature_set,
                               "baseline": note, "test_rows": n_test,
                               "sample_fraction": args.sample_fraction})
            mlflow.log_metrics(metrics)
        out[run_name] = dict(metrics, model=f"baseline_{task}", arm="none", task=task)
        print(f"  baseline {task:<14} " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
              + f"   ({note})")

    # One-line rules on the strongest legitimate input, when the feature set has
    # one. For the wheels-off set: arrival delay = departure delay + the average
    # time made up (about -5 min). Whatever a model scores above its rule is
    # what the model adds; the rule alone is what "looks like a leak".
    numeric = FEATURE_SETS[args.feature_set]["numeric"]
    for col in [c for c in ("DEPARTURE_DELAY", "prev_arr_delay") if c in numeric]:
        stats = train.agg(F.avg(col).alias("mx"), F.avg(REG_LABEL).alias("my"),
                          F.covar_samp(col, REG_LABEL).alias("cxy"),
                          F.var_samp(col).alias("vx")).first()
        slope = float(stats["cxy"] / stats["vx"]) if stats["vx"] else 0.0
        icpt = float(stats["my"] - slope * stats["mx"])
        x = F.coalesce(F.col(col).cast("double"), F.lit(float(stats["mx"])))
        reg_pred = F.lit(icpt) + F.lit(slope) * x
        for task, label, preds, note in (
            ("regression", REG_LABEL, test.withColumn("prediction", reg_pred),
             f"{icpt:+.2f} + {slope:.3f} * {col}, fitted on train"),
            ("classification", CLF_LABEL,
             test.withColumn("prediction", (reg_pred > SEVERE_THRESHOLD_MIN).cast("double"))
                 .withColumn("rawPrediction", array_to_vector(F.array(-x, x))),
             f"score = {col}; late if the regression rule says > {SEVERE_THRESHOLD_MIN} min"),
        ):
            run_name = f"rule_{col.lower()}_{task}__none"
            metrics = {m: float(e.evaluate(preds)) for m, e in evaluators(task, label).items()}
            with mlflow.start_run(run_name=run_name):
                mlflow.log_params({"model": f"rule_{col.lower()}_{task}", "arm": "none",
                                   "task": task, "label": label,
                                   "feature_set": args.feature_set, "baseline": note,
                                   "test_rows": n_test,
                                   "sample_fraction": args.sample_fraction})
                mlflow.log_metrics(metrics)
            out[run_name] = dict(metrics, model=f"rule_{col.lower()}_{task}", arm="none",
                                 task=task)
            print(f"  rule {col:<16} {task:<14} "
                  + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + f"   ({note})")
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-fraction", type=float, default=1.0,
                    help="fraction of the curated table to use overall")
    ap.add_argument("--tune-fraction", type=float, default=0.1,
                    help="fraction of TRAIN used for cross-validation search")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--feature-set", default=DEFAULT_FEATURE_SET,
                    choices=sorted(FEATURE_SETS),
                    help="predeparture (default): only what is known before the "
                         "aircraft leaves the gate. predeparture_all: every candidate, "
                         "the ablation reference. wheelsoff: the September tournament's "
                         "features, DEPARTURE_DELAY and target encoding included.")
    ap.add_argument("--pca-variance", type=float, default=0.90,
                    help="PCA arms keep the fewest components reaching this share of "
                         "the variance. Ignored when --pca-k is given.")
    ap.add_argument("--pca-k", type=int, default=None,
                    help="fixed number of PCA components (overrides --pca-variance)")
    ap.add_argument("--svd-mode", default="local-eigs",
                    choices=["auto", "local-svd", "local-eigs", "dist-eigs"],
                    help="RowMatrix.computeSVD mode for the PCA arms. dist-eigs is "
                         "the mode the brief describes - ARPACK Lanczos on distributed "
                         "A^T(Av) products, never materialising the n x n covariance - "
                         "and it converges here, but it costs one pass over the rows "
                         "per ARPACK product (90 to 105 of them on this covariance) "
                         "where local-eigs costs one. local-eigs forms the 77x77 "
                         "Gramian on the driver (46 KB) and returns identical components "
                         "at |cos| = 1.0, so that is the default and what the registered "
                         "model was fitted with. dist-eigs failed in the tournament "
                         "because parallel CV ran several ARPACK solves at once, which "
                         "Spark's ARPACK cannot survive; RowMatrixPCA now serialises "
                         "the solve (custom_transformers._ARPACK_LOCK).")
    ap.add_argument("--models", default="all")
    ap.add_argument("--max-dep-delay", type=float, default=None,
                    help="wheels-off feature sets only: keep flights whose departure "
                         "delay is at most this many minutes, in train AND test. It "
                         "defines the model's scope (above it, arrival ~ departure "
                         "delay - 6 is a rule, not a prediction problem); legal because "
                         "departure delay is known at wheels-off. 2%% of flights are "
                         "over 120 min and carry 62%% of the target's sum of squares.")
    ap.add_argument("--class-weight", default="balanced", choices=["balanced", "none"],
                    help="weight the classifiers' training rows so each class carries "
                         "half the weight (default), or not at all")
    ap.add_argument("--arms", default="both", choices=["both", "pca", "nopca"],
                    help="run only one arm per model, e.g. nopca for a quick ablation")
    ap.add_argument("--register-arm", default="pca",
                    choices=["pca", "nopca", "any"],
                    help="which arm may be promoted to Production. The brief's "
                         "Preprocessing spec names PCA(k=10) as part of the Pipeline, "
                         "so the shipped artifact must contain a PCA stage - hence the "
                         "default. 'any' restores pure best-RMSE selection, which picks "
                         "the no-PCA arm here (RMSE 10.52 against 21.48).")
    ap.add_argument("--experiment", default="flight-delay-predeparture",
                    help="flight-delay-mllib holds the two wheels-off tournaments")
    ap.add_argument("--out-dir", default=None,
                    help="where results and plots go; default spark_session.results_dir: "
                         "docs/benchmarks/... for the two official tournaments, "
                         "experiments/<experiment>/ for every trial")
    ap.add_argument("--note", default="",
                    help="what this run is testing; stored as the MLflow experiment "
                         "description and on each run. Also write it into experiments/LOG.md")
    ap.add_argument("--no-register", action="store_true",
                    help="skip register_winner: nothing in the Model Registry or in "
                         "models/ changes. Use for every run you do not mean to ship.")
    ap.add_argument("--register-only", action="store_true",
                    help="train nothing: register the best finished arm of --experiment "
                         "(after a --no-register tournament), archiving the current "
                         "Production version. Back up mlflow.db and models/ first.")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="skip arms already completed in the tracking store (default)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="re-run every arm even if results already exist")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop on the first failing arm instead of continuing")
    args = ap.parse_args()
    if args.pca_k is not None:
        args.pca_variance = 0.0
    else:
        args.pca_k = 10          # unused while pca_variance > 0
    if args.out_dir is None:
        args.out_dir = results_dir(args.experiment)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.register_only:
        mlflow.set_tracking_uri(TRACKING_URI)
        results = completed_arms(args.experiment)
        if not results:
            print(f"no finished arms in {args.experiment!r}; nothing to register")
            return
        spark = build_spark("mllib-register")      # load_model needs a session
        try:
            register_winner(results, args)
        finally:
            spark.stop()
        return
    if pause_requested():
        print(f"{PAUSE_FILE} exists, so the run would pause at once. Delete it to run.")
        return
    spark = build_spark("mllib-pipeline")

    # SQLite rather than the default file store, for two reasons: the MLflow
    # Model Registry is only available on a database-backed store (the brief
    # requires registering the winner and moving it Staging -> Production),
    # and the file store races when CrossValidator's parallel folds make
    # autologging create nested runs from several threads at once.
    mlflow.set_tracking_uri(TRACKING_URI)
    exp = mlflow.set_experiment(args.experiment)
    if args.note:
        # Shown as the experiment's description in the MLflow UI, and kept on
        # every run as a tag, so a trial explains itself wherever it is opened.
        MlflowClient().set_experiment_tag(exp.experiment_id, "mlflow.note.content", args.note)
    # The brief asks for automatic hyperparameter capture via mlflow.pyspark.ml.
    # Models are logged explicitly below so we control which artifact is
    # registered, hence log_models=False here.
    mlflow.pyspark.ml.autolog(log_models=False, log_datasets=False)

    try:
        df = spark.read.parquet(CURATED)

        # Split FIRST, then derive the GLM shift from the training half only.
        # The offset is a single order statistic, and every metric here is
        # invariant to a common shift of prediction and target, so taking it
        # from the full frame could not have inflated a result - but B1.1
        # claims nothing fitted ever sees test data, and min() over test is
        # fitted. Cheaper to be consistent than to caveat it.
        #
        # randomSplit assigns rows from the input's existing partitioning, and
        # adding a projection afterwards is a narrow transformation, so moving
        # withColumn below the split leaves the split itself unchanged. That
        # matters: the 14 completed arms are only comparable while the split
        # is byte-identical, and scripts/check_split.py asserts it.
        train, test = make_split(df, args.sample_fraction)
        # Rotation and congestion lookups, joined after the split so the rows in
        # each half do not move. A no-op for feature sets that use neither.
        numeric = FEATURE_SETS[args.feature_set]["numeric"]
        train = add_context_features(spark, df, train, numeric)
        test = add_context_features(spark, df, test, numeric)
        if args.max_dep_delay is not None:
            if "DEPARTURE_DELAY" not in numeric:
                raise SystemExit("--max-dep-delay needs a feature set that knows the "
                                 "departure delay (wheelsoff); before departure it is unknown")
            # Filtered after the split: each half keeps the same rows it had,
            # minus the out-of-scope ones.
            in_scope = F.col("DEPARTURE_DELAY") <= F.lit(float(args.max_dep_delay))
            train, test = train.filter(in_scope), test.filter(in_scope)

        min_delay = train.agg(F.min(REG_LABEL)).first()[0]
        glm_offset = float(abs(min(min_delay, 0.0)) + 1.0)
        train = train.withColumn("label_glm", F.col(REG_LABEL) + F.lit(glm_offset))
        test = test.withColumn("label_glm", F.col(REG_LABEL) + F.lit(glm_offset))

        # Balanced class weights from the training split's positive rate. Test
        # carries the column only so both frames share a schema; no evaluator
        # reads it, so every test metric stays unweighted.
        pos = float(train.agg(F.avg(CLF_LABEL)).first()[0])
        w_pos, w_neg = 0.5 / pos, 0.5 / (1.0 - pos)
        weight = F.when(F.col(CLF_LABEL) == 1, F.lit(w_pos)).otherwise(F.lit(w_neg))
        train = train.withColumn(WEIGHT_COL, weight)
        test = test.withColumn(WEIGHT_COL, weight)

        train = train.cache()
        test = test.cache()
        n_train, n_test = train.count(), test.count()
        print(f"train={n_train:,}  test={n_test:,}  glm_offset={glm_offset:.1f}")

        # Pin down the contract the streaming job has to satisfy. Publishing
        # the schema alongside the model is what lets inference.py declare a
        # readStream schema without re-deriving it (and file-source streaming
        # requires an explicit schema anyway).
        # Written beside the results; register_winner copies it into models/
        # only when it ships a model, so a dev run cannot desync the contract
        # from the model that is actually in Production.
        label_cols = {REG_LABEL, CLF_LABEL, "label_glm", WEIGHT_COL}
        serving = [f for f in train.schema.fields if f.name not in label_cols]
        with open(os.path.join(args.out_dir, "input_schema.json"), "w") as fh:
            json.dump({"fields": [{"name": f.name, "type": f.dataType.simpleString()}
                                  for f in serving],
                       "glm_offset": glm_offset,
                       "feature_set": args.feature_set}, fh, indent=2)

        # Tuning runs on a sample; the winning params are then refitted on the
        # full training set. 7 model families x 5 folds x grid over 4.5M rows
        # is many hours, and the ranking of hyperparameters is stable well
        # before the metric value is.
        tune = (train if args.tune_fraction >= 1.0
                else train.sample(False, args.tune_fraction, seed=SPLIT_SEED).cache())
        n_tune = tune.count()
        print(f"tuning on {n_tune:,} rows ({args.tune_fraction:.1%} of train)\n")

        specs = model_specs(args)
        wanted = ([m for m in specs if m not in OPT_IN_MODELS] if args.models == "all"
                  else [m.strip() for m in args.models.split(",")])

        already_done = completed_arms(args.experiment) if args.resume else {}
        if already_done:
            print("resuming: %d arm(s) already complete" % len(already_done))
            print("")

        results, failures = {}, {}
        results.update(log_baselines(train, test, n_test, args))
        checkpoint(results, failures, args.out_dir)
        try:
            run_arms(wanted, specs, args, train, test, tune, n_train, n_test, n_tune,
                     already_done, results, failures)
        except (Paused, KeyboardInterrupt, SparkDied) as stop:
            if mlflow.active_run():
                mlflow.end_run(status="KILLED")
            checkpoint(results, failures, args.out_dir)
            how = (f"paused {stop}" if isinstance(stop, Paused)
                   else f"stopped: {stop}" if isinstance(stop, SparkDied)
                   else "interrupted (Ctrl+C)")
            print(f"\n{how}. {len(results)} result(s) kept in {args.out_dir}.")
            print(f"To continue: delete {PAUSE_FILE} if present, then rerun the same command.")
            return
        _finish(results, args)
    finally:
        try:
            spark.stop()
        except Exception:  # noqa: BLE001 - a dead JVM cannot be stopped twice
            pass


def run_arms(wanted, specs, args, train, test, tune, n_train, n_test, n_tune,
             already_done, results, failures) -> None:
    """The tournament loop. Pauses by raising Paused at a safe point."""
    for name in wanted:
        spec = specs[name]
        print(f"=== {name} ===")
        for use_pca in {"both": (True, False), "pca": (True,), "nopca": (False,)}[args.arms]:
            arm = "pca" if use_pca else "nopca"
            run_name = f"{name}__{arm}"
            if args.resume and run_name in already_done:
                results[run_name] = already_done[run_name]
                print(f"  {arm:<5} skipped - already complete")
                checkpoint(results, failures, args.out_dir)
                continue
            if pause_requested():
                raise Paused(f"before {run_name}")

            try:
                t0 = time.time()

                stages, feature_names = build_feature_stages(
                    spec["label"], use_pca, args.pca_k, args.svd_mode,
                    args.feature_set, args.pca_variance)
                pipeline = Pipeline(stages=stages + [spec["estimator"]])
                evals = evaluators(spec["task"], spec["label"])
                primary = "rmse" if spec["task"] == "regression" else "areaUnderROC"

                cv = CrossValidator(
                    estimator=pipeline,
                    estimatorParamMaps=spec["grid"],
                    evaluator=evals[primary],
                    numFolds=args.folds,
                    # Evaluate grid points concurrently across executor slots
                    # rather than one after another.
                    parallelism=args.parallelism,
                    seed=SPLIT_SEED,
                    collectSubModels=False)

                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params({
                        "model": name, "arm": arm, "task": spec["task"],
                        "label": spec["label"], "folds": args.folds,
                        "parallelism": args.parallelism,
                        "feature_set": args.feature_set,
                        "class_weight": args.class_weight
                                        if spec["task"] == "classification" else "",
                        "pca_k": (args.pca_k if not args.pca_variance else "auto")
                                 if use_pca else 0,
                        "pca_variance_target": args.pca_variance if use_pca else 0,
                        "svd_mode": args.svd_mode if use_pca else "",
                        "tune_rows": n_tune, "train_rows": n_train,
                        "test_rows": n_test, "grid_size": len(spec["grid"]),
                        "sample_fraction": args.sample_fraction,
                    "max_dep_delay": args.max_dep_delay if args.max_dep_delay is not None else "",
                    })
                    if args.note:
                        mlflow.set_tag("note", args.note)

                    saved = load_cv_checkpoint(args, run_name)
                    cv_model = None
                    if saved is not None:
                        best_params, cv_value = saved["best_params"], saved["value"]
                        print(f"  {arm:<5} CV restored from checkpoint")
                        mlflow.set_tag("cv_restored_from_checkpoint", "true")
                    else:
                        cv_model = cv.fit(tune)
                        # CrossValidator ranks by the evaluator's own direction,
                        # so read that rather than assuming higher-is-better.
                        avg = cv_model.avgMetrics
                        pick = max if evals[primary].isLargerBetter() else min
                        best_idx = pick(range(len(avg)), key=lambda i: avg[i])
                        best_params = {
                            p.name: v
                            for p, v in cv_model.getEstimatorParamMaps()[best_idx].items()}
                        cv_value = float(avg[best_idx])
                        save_cv_checkpoint(args, run_name, best_params, primary, cv_value)
                    mlflow.log_metric("cv_best_" + primary, cv_value)
                    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                    if pause_requested():
                        raise Paused(f"{run_name} after cross-validation (saved)")

                    # Refit the winning configuration on the full training set.
                    final = (cv_model.bestModel
                             if cv_model is not None and args.tune_fraction >= 1.0
                             else None)
                    if final is None:
                        best_est = spec["estimator"].copy(
                            {spec["estimator"].getParam(k): v
                             for k, v in best_params.items()})
                        stages2, _ = build_feature_stages(
                            spec["label"], use_pca, args.pca_k, args.svd_mode,
                            args.feature_set, args.pca_variance)
                        final = Pipeline(stages=stages2 + [best_est]).fit(train)

                    preds = final.transform(test)
                    metrics = {m: float(e.evaluate(preds)) for m, e in evals.items()}
                    metrics["train_seconds"] = time.time() - t0
                    mlflow.log_metrics(metrics)

                    if use_pca:
                        pca_stage = [s for s in final.stages
                                     if hasattr(s, "explainedVariance")][0]
                        ev = list(map(float, pca_stage.explainedVariance.toArray()))
                        mlflow.log_metric("pca_variance_retained", float(sum(ev)))
                        mlflow.log_metric("pca_k_chosen", len(ev))
                        # One file per arm. Every PCA arm used to write the
                        # same filename, so the seven of them overwrote each
                        # other and the survivor was whichever finished last
                        # - a classification arm, while the registered model
                        # is a regression one. They differ for a real reason:
                        # TargetEncoder is fitted on the label, so the matrix
                        # reaching PCA is not the same for the two tasks.
                        # register_winner writes the canonical file.
                        ev_path = os.path.join(
                            args.out_dir, f"pca_explained_variance__{run_name}.json")
                        with open(ev_path, "w") as fh:
                            json.dump(ev, fh)
                        mlflow.log_artifact(ev_path)

                    # Importance for every model (trees and linear alike,
                    # PCA arms carried back to the original features), ROC and
                    # PR curves for classifiers, residual plots for
                    # regressors; all logged under the run's plots/.
                    model_explain.explain_arm(
                        final, train, preds, task=spec["task"], label=spec["label"],
                        n_test=n_test, metrics=metrics, run_name=run_name,
                        out_dir=args.out_dir)

                    results[run_name] = dict(metrics, model=name, arm=arm,
                                             task=spec["task"], **{
                                                 f"best_{k}": v
                                                 for k, v in best_params.items()})
                    print(f"  {arm:<5} " + "  ".join(
                        f"{k}={v:.4f}" for k, v in metrics.items()))

                    mlflow.spark.log_model(final, artifact_path="pipeline_model")
            except Exception as exc:  # noqa: BLE001
                # One arm failing must not cost the other thirteen. The
                # failure is recorded so the summary can report it, and the
                # arm can be retried on its own with --models later.
                failures[run_name] = f"{type(exc).__name__}: {exc}"
                print(f"  {arm:<5} FAILED - {type(exc).__name__}: {exc}")
                if args.fail_fast:
                    raise
                if mlflow.active_run():
                    mlflow.end_run(status="FAILED")
                checkpoint(results, failures, args.out_dir)
                if not spark_alive(train):
                    # An OutOfMemoryError in the JVM shuts the SparkContext down.
                    # Every later arm would then "fail" in milliseconds for no
                    # reason of its own; stop instead, so a rerun of the same
                    # command resumes cleanly from the arms that finished.
                    raise SparkDied(f"SparkContext is gone after {run_name}")
                continue

            # Release what MaterializeCache persisted for this arm's fits.
            # train/test/tune are cached separately and are untouched.
            freed = custom_transformers.release_caches()
            if freed:
                print(f"  {'':<5} released {freed} cached frame(s)")

            checkpoint(results, failures, args.out_dir)


def _finish(results: dict, args) -> None:
    with open(os.path.join(args.out_dir, "tournament_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {os.path.join(args.out_dir, 'tournament_results.json')}")

    if args.no_register:
        print("--no-register: Model Registry and models/ left untouched")
    else:
        register_winner(results, args)


def register_winner(results: dict, args) -> None:
    """Register the best regression pipeline and promote Staging -> Production."""
    reg = {k: v for k, v in results.items() if v["task"] == "regression"
           and v.get("arm") != "none"}          # baselines and rules
    if not reg:
        print("no regression runs to register")
        return
    # The brief specifies PCA(k=10) in the Pipeline's preprocessing, so the
    # artifact we ship has to contain a PCA stage. Pure best-RMSE selection
    # registers the no-PCA arm (10.52 against 21.48) and hands a marker a
    # SQLTransformer passthrough where PCA should be. The no-PCA arms still
    # run and are still logged - they are simply not what gets promoted.
    want = getattr(args, "register_arm", "any")
    if want != "any":
        eligible = {k: v for k, v in reg.items() if v.get("arm") == want}
        if eligible:
            reg = eligible
        else:
            print(f"warning: no regression arm matched --register-arm {want}; "
                  f"falling back to all {len(reg)} regression arms")

    best_name = min(reg, key=lambda k: reg[k]["rmse"])
    print(f"\nbest regression pipeline: {best_name} (rmse={reg[best_name]['rmse']:.4f})")

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)

    # Take the newest run for this arm that actually FINISHED *and* still has a
    # readable model artifact. Without both checks a failed re-attempt shadows
    # the good earlier run (it is newer), gets promoted to Production, and then
    # cannot be downloaded - which is exactly how a working model was lost.
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.mlflow.runName = '{best_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"], max_results=10)
    win = None
    for r in runs:
        try:
            names = {a.path for a in client.list_artifacts(r.info.run_id, "pipeline_model")}
            if any(n.endswith("MLmodel") for n in names):
                win = r
                break
            print(f"  skipping {r.info.run_id[:8]}: no pipeline_model artifact")
        except Exception as exc:  # noqa: BLE001
            print(f"  skipping {r.info.run_id[:8]}: {type(exc).__name__}: {exc}")
    if win is None:
        print(f"no FINISHED run named {best_name} still has a model artifact; "
              f"leaving Production and {MODEL_DIR} untouched")
        return

    uri = f"runs:/{win.info.run_id}/pipeline_model"

    # Stage the new model on disk BEFORE destroying anything. If the download or
    # the write fails we return with the previous model still in place.
    best_dir = os.path.join(MODEL_DIR, "best_pipeline")
    staging = best_dir + ".staging"
    previous = best_dir + ".previous"
    os.makedirs(MODEL_DIR, exist_ok=True)
    shutil.rmtree(staging, ignore_errors=True)
    try:
        mlflow.spark.load_model(uri).write().overwrite().save(staging.replace("\\", "/"))
    except Exception as exc:  # noqa: BLE001
        shutil.rmtree(staging, ignore_errors=True)
        print(f"could not materialise {uri}: {type(exc).__name__}: {exc}")
        print(f"Production and {best_dir} left untouched")
        return

    mv = mlflow.register_model(uri, REGISTERED_MODEL)
    # MLflow 2.x stage transitions, exactly as the brief specifies.
    client.transition_model_version_stage(REGISTERED_MODEL, mv.version, "Staging")
    client.transition_model_version_stage(REGISTERED_MODEL, mv.version, "Production",
                                          archive_existing_versions=True)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "task", "regression")
    print(f"registered {REGISTERED_MODEL} v{mv.version} -> Production")

    # Record WHICH label the winner was trained on, into the serving contract.
    # The GLM arms train on label_glm, which is ARRIVAL_DELAY + glm_offset, so
    # their predictions come back on the shifted scale. Without this, a GLM
    # winner would make inference.py emit every prediction glm_offset minutes
    # too high - silently, because a shifted delay is still a plausible delay.
    won_label = win.data.params.get("label", REG_LABEL)
    meta_path = os.path.join(MODEL_DIR, "input_schema.json")
    try:
        if os.path.exists(meta_path):
            shutil.copy2(meta_path, meta_path + ".previous")
        shutil.copy2(os.path.join(args.out_dir, "input_schema.json"), meta_path)
        with open(meta_path) as fh:
            meta = json.load(fh)
        meta["label"] = won_label
        meta["winner"] = best_name
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"serving contract: label={won_label}")
    except Exception as exc:  # noqa: BLE001
        print(f"could not update the serving contract: {exc}")

    # Swap last, and keep the old tree until the new one is in place. Only the
    # model directory moves: wiping MODEL_DIR would take input_schema.json with
    # it - the serving contract inference.py and make_stream_events.py read.
    shutil.rmtree(previous, ignore_errors=True)
    if os.path.exists(best_dir):
        os.rename(best_dir, previous)
    try:
        os.rename(staging, best_dir)
    except Exception:  # noqa: BLE001 - put the old model back, then re-raise
        if os.path.exists(previous) and not os.path.exists(best_dir):
            os.rename(previous, best_dir)
        raise
    shutil.rmtree(previous, ignore_errors=True)
    print(f"saved {best_dir}")

    # The canonical explained-variance artifact is written here, from the model
    # that was actually registered, so it can never describe a different arm.
    try:
        from pyspark.ml import PipelineModel
        staged = PipelineModel.load(best_dir.replace("\\", "/"))
        pca = [st for st in staged.stages if hasattr(st, "explainedVariance")]
        if pca:
            ev = list(map(float, pca[0].explainedVariance.toArray()))
            with open(os.path.join(args.out_dir, "pca_explained_variance.json"), "w") as fh:
                json.dump(ev, fh)
            print(f"wrote pca_explained_variance.json from the registered model "
                  f"({100 * sum(ev):.3f}% retained)")
        else:
            print("registered model has no PCA stage; canonical variance file left alone")
    except Exception as exc:  # noqa: BLE001 - reporting only, never fail the run
        print(f"could not refresh pca_explained_variance.json: "
              f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
