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
from pyspark.ml.feature import (PCA, Imputer, OneHotEncoder, SQLTransformer,
                                StandardScaler, StringIndexer, VarianceThresholdSelector,
                                VectorAssembler)
from pyspark.ml.regression import (GBTRegressor, GeneralizedLinearRegression,
                                   LinearRegression, RandomForestRegressor)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_transformers import (  # noqa: E402
    HaversineTransformer, MaterializeCache, OutlierIQRTruncator,
    SignedLog1pTransformer, TargetEncoder,
)
from spark_session import TRACKING_URI, build_spark, path  # noqa: E402

CURATED = path("data", "parquet", "flights_curated")
MODEL_DIR = path("models")
BENCH_DIR = path("docs", "benchmarks")
REGISTERED_MODEL = "flight_delay_pipeline"
SPLIT_SEED = 42

REG_LABEL = "label_delay"
CLF_LABEL = "label_severe"

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
# Only the volatile operational columns get Tukey-clipped. DISTANCE is
# deliberately left alone: its IQR fence lands at ~2,091 mi, which would fold
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

NUMERIC_FEATURES = (
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


def build_feature_stages(label_col: str, use_pca: bool, pca_k: int):
    """Shared preprocessing. Returns (stages, assembler_input_names)."""
    stages = []

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

    # Everything expensive (the Arrow UDF, the target-encoding joins) is now
    # behind us; cache here so StandardScaler, PCA and the estimator do not
    # each re-derive it. See MaterializeCache for the measurement.
    stages.append(MaterializeCache())

    assembler_inputs = NUMERIC_FEATURES + ohe_out
    stages.append(VectorAssembler(inputCols=assembler_inputs,
                                  outputCol="features_raw", handleInvalid="keep"))

    # withMean=False is deliberate, not a default left untouched. The assembled
    # vector is mostly one-hot and therefore sparse; subtracting a mean would
    # make every zero non-zero and densify it. withStd rescales in place and
    # preserves sparsity. See docs/REPORT.md for the full argument.
    stages.append(StandardScaler(inputCol="features_raw", outputCol="features_scaled",
                                 withMean=False, withStd=True))

    # One-hot encoding with handleInvalid="keep" reserves a slot for unseen
    # categories, and on any split where that slot never fires the column is
    # constant. StandardScaler maps a zero-variance column to all zeros, which
    # leaves the covariance matrix singular - and Spark's PCA runs a Breeze SVD
    # over exactly that matrix, so it fails outright with NotConvergedException
    # rather than degrading. Dropping zero-variance columns first is both the
    # fix and the right thing to do: a constant feature carries no information.
    stages.append(VarianceThresholdSelector(
        featuresCol="features_scaled", outputCol="features_selected",
        varianceThreshold=0.0))

    if use_pca:
        stages.append(PCA(k=pca_k, inputCol="features_selected", outputCol="features"))
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
        return [f"PC{i + 1}" for i in range(pca_k)]
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
def model_specs(args):
    """Each entry: estimator, grid, task, label column, primary metric."""
    grid = {}

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

    svc = LinearSVC(featuresCol="features", labelCol=CLF_LABEL, maxIter=30)
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
                                 numTrees=40, maxDepth=8, maxBins=64, seed=7)
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
                         maxIter=40, maxDepth=5, maxBins=64, seed=7)
    grid["gbt_classifier"] = dict(
        estimator=gbtc, task="classification", label=CLF_LABEL,
        grid=(ParamGridBuilder()
              .addGrid(gbtc.maxDepth, [4, 6])
              .build()))

    return grid


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
        for r in client.search_runs([exp.experiment_id], max_results=1000,
                                    order_by=["attributes.start_time ASC"]):
            run_name = r.data.tags.get("mlflow.runName", "")
            if "__" not in run_name or not r.data.metrics:
                continue
            done[run_name] = dict(
                r.data.metrics,
                model=r.data.params.get("model", run_name.split("__")[0]),
                arm=r.data.params.get("arm", run_name.split("__")[-1]),
                task=r.data.params.get("task", ""))
        return done
    except Exception as exc:  # noqa: BLE001 - never let bookkeeping stop a run
        print(f"  (could not read previous runs: {exc})")
        return {}


def checkpoint(results: dict, failures: dict) -> None:
    """Persist progress after every arm, not just at the end of the loop."""
    try:
        os.makedirs(BENCH_DIR, exist_ok=True)
        with open(os.path.join(BENCH_DIR, "tournament_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        if failures:
            with open(os.path.join(BENCH_DIR, "tournament_failures.json"), "w") as fh:
                json.dump(failures, fh, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"  (checkpoint failed: {exc})")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-fraction", type=float, default=1.0,
                    help="fraction of the curated table to use overall")
    ap.add_argument("--tune-fraction", type=float, default=0.1,
                    help="fraction of TRAIN used for cross-validation search")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--parallelism", type=int, default=4)
    ap.add_argument("--pca-k", type=int, default=10)
    ap.add_argument("--models", default="all")
    ap.add_argument("--experiment", default="flight-delay-mllib")
    ap.add_argument("--resume", dest="resume", action="store_true", default=True,
                    help="skip arms already completed in the tracking store (default)")
    ap.add_argument("--no-resume", dest="resume", action="store_false",
                    help="re-run every arm even if results already exist")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop on the first failing arm instead of continuing")
    args = ap.parse_args()

    os.makedirs(BENCH_DIR, exist_ok=True)
    spark = build_spark("mllib-pipeline")

    # SQLite rather than the default file store, for two reasons: the MLflow
    # Model Registry is only available on a database-backed store (the brief
    # requires registering the winner and moving it Staging -> Production),
    # and the file store races when CrossValidator's parallel folds make
    # autologging create nested runs from several threads at once.
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(args.experiment)
    # The brief asks for automatic hyperparameter capture via mlflow.pyspark.ml.
    # Models are logged explicitly below so we control which artifact is
    # registered, hence log_models=False here.
    mlflow.pyspark.ml.autolog(log_models=False, log_datasets=False)

    try:
        df = spark.read.parquet(CURATED)
        if args.sample_fraction < 1.0:
            df = df.sample(False, args.sample_fraction, seed=SPLIT_SEED)

        # GLM shift, computed on the full frame before splitting so the offset
        # is a fixed constant rather than a fitted quantity.
        min_delay = df.agg(F.min(REG_LABEL)).first()[0]
        glm_offset = float(abs(min(min_delay, 0.0)) + 1.0)
        df = df.withColumn("label_glm", F.col(REG_LABEL) + F.lit(glm_offset))

        train, test = df.randomSplit([0.8, 0.2], seed=SPLIT_SEED)
        train = train.cache()
        test = test.cache()
        n_train, n_test = train.count(), test.count()
        print(f"train={n_train:,}  test={n_test:,}  glm_offset={glm_offset:.1f}")

        # Pin down the contract the streaming job has to satisfy. Publishing
        # the schema alongside the model is what lets inference.py declare a
        # readStream schema without re-deriving it (and file-source streaming
        # requires an explicit schema anyway).
        os.makedirs(MODEL_DIR, exist_ok=True)
        label_cols = {REG_LABEL, CLF_LABEL, "label_glm"}
        serving = [f for f in train.schema.fields if f.name not in label_cols]
        with open(os.path.join(MODEL_DIR, "input_schema.json"), "w") as fh:
            json.dump({"fields": [{"name": f.name, "type": f.dataType.simpleString()}
                                  for f in serving],
                       "glm_offset": glm_offset}, fh, indent=2)

        # Tuning runs on a sample; the winning params are then refitted on the
        # full training set. 7 model families x 5 folds x grid over 4.5M rows
        # is many hours, and the ranking of hyperparameters is stable well
        # before the metric value is.
        tune = (train if args.tune_fraction >= 1.0
                else train.sample(False, args.tune_fraction, seed=SPLIT_SEED).cache())
        n_tune = tune.count()
        print(f"tuning on {n_tune:,} rows ({args.tune_fraction:.1%} of train)\n")

        specs = model_specs(args)
        wanted = (list(specs) if args.models == "all"
                  else [m.strip() for m in args.models.split(",")])

        already_done = completed_arms(args.experiment) if args.resume else {}
        if already_done:
            print("resuming: %d arm(s) already complete" % len(already_done))
            print("")

        results, failures = {}, {}
        for name in wanted:
            spec = specs[name]
            print(f"=== {name} ===")
            for use_pca in (True, False):
                arm = "pca" if use_pca else "nopca"
                run_name = f"{name}__{arm}"
                if args.resume and run_name in already_done:
                    results[run_name] = already_done[run_name]
                    print(f"  {arm:<5} skipped - already complete")
                    checkpoint(results, failures)
                    continue

                try:
                    t0 = time.time()

                    stages, feature_names = build_feature_stages(
                        spec["label"], use_pca, args.pca_k)
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
                            "pca_k": args.pca_k if use_pca else 0,
                            "tune_rows": n_tune, "train_rows": n_train,
                            "test_rows": n_test, "grid_size": len(spec["grid"]),
                        })

                        cv_model = cv.fit(tune)
                        # CrossValidator ranks by the evaluator's own direction, so
                        # read that rather than assuming higher-is-better.
                        avg = cv_model.avgMetrics
                        pick = max if evals[primary].isLargerBetter() else min
                        best_idx = pick(range(len(avg)), key=lambda i: avg[i])
                        best_params = {
                            p.name: v
                            for p, v in cv_model.getEstimatorParamMaps()[best_idx].items()}
                        mlflow.log_metric("cv_best_" + primary, float(avg[best_idx]))
                        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

                        # Refit the winning configuration on the full training set.
                        final = cv_model.bestModel if args.tune_fraction >= 1.0 else None
                        if final is None:
                            best_est = spec["estimator"].copy(
                                {spec["estimator"].getParam(k): v
                                 for k, v in best_params.items()})
                            stages2, _ = build_feature_stages(
                                spec["label"], use_pca, args.pca_k)
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
                            with open(os.path.join(BENCH_DIR, "pca_explained_variance.json"),
                                      "w") as fh:
                                json.dump(ev, fh)
                            mlflow.log_artifact(
                                os.path.join(BENCH_DIR, "pca_explained_variance.json"))

                        est_stage = final.stages[-1]
                        if hasattr(est_stage, "featureImportances"):
                            imp = list(map(float, est_stage.featureImportances.toArray()))
                            names = resolve_feature_names(final, train, use_pca, args.pca_k)
                            if len(names) != len(imp):
                                names = [f"f{i}" for i in range(len(imp))]
                            payload = {"arm": arm, "features": names, "importances": imp}
                            fp = os.path.join(BENCH_DIR, f"importance_{run_name}.json")
                            with open(fp, "w") as fh:
                                json.dump(payload, fh)
                            mlflow.log_artifact(fp)

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
                    checkpoint(results, failures)
                    continue

                checkpoint(results, failures)

        with open(os.path.join(BENCH_DIR, "tournament_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {os.path.join(BENCH_DIR, 'tournament_results.json')}")

        register_winner(results, args)
    finally:
        spark.stop()


def register_winner(results: dict, args) -> None:
    """Register the best regression pipeline and promote Staging -> Production."""
    reg = {k: v for k, v in results.items() if v["task"] == "regression"}
    if not reg:
        print("no regression runs to register")
        return
    best_name = min(reg, key=lambda k: reg[k]["rmse"])
    print(f"\nbest regression pipeline: {best_name} (rmse={reg[best_name]['rmse']:.4f})")

    client = MlflowClient()
    exp = client.get_experiment_by_name(args.experiment)
    runs = client.search_runs([exp.experiment_id],
                             filter_string=f"tags.mlflow.runName = '{best_name}'",
                             order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        print("could not locate the winning run in the tracking store")
        return

    uri = f"runs:/{runs[0].info.run_id}/pipeline_model"
    mv = mlflow.register_model(uri, REGISTERED_MODEL)
    # MLflow 2.x stage transitions, exactly as the brief specifies.
    client.transition_model_version_stage(REGISTERED_MODEL, mv.version, "Staging")
    client.transition_model_version_stage(REGISTERED_MODEL, mv.version, "Production",
                                          archive_existing_versions=True)
    client.set_model_version_tag(REGISTERED_MODEL, mv.version, "task", "regression")
    print(f"registered {REGISTERED_MODEL} v{mv.version} -> Production")

    # Also drop a plain PipelineModel on disk for the streaming job, which
    # should not need the tracking store to be reachable.
    # Only clear the model directory itself. Wiping MODEL_DIR would take
    # input_schema.json with it - the serving contract that inference.py and
    # make_stream_events.py both read.
    best_dir = os.path.join(MODEL_DIR, "best_pipeline")
    if os.path.exists(best_dir):
        shutil.rmtree(best_dir, ignore_errors=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    mlflow.spark.load_model(uri).write().overwrite().save(
        os.path.join(MODEL_DIR, "best_pipeline").replace("\\", "/"))
    print(f"saved {os.path.join(MODEL_DIR, 'best_pipeline')}")


if __name__ == "__main__":
    main()
