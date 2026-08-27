"""Custom PySpark ML stages for the flight-delay pipeline.

Contents
--------
Arrow-vectorised UDFs
    ``haversine_miles_udf``   great-circle distance from lat/lon pairs
    ``schedule_speed_udf``    implied ground speed of the published schedule

Pipeline stages
    ``HaversineTransformer``      runs the two pandas_udfs *inside* the Pipeline
    ``SignedLog1pTransformer``    skew compression that tolerates negatives
    ``OutlierIQRTruncator``       Estimator -> ``OutlierIQRTruncatorModel``
    ``TargetEncoder``             Estimator -> ``TargetEncoderModel``

Two design decisions are worth stating up front, because both are places
where the obvious implementation is subtly wrong.

**Why the IQR truncator is an Estimator and not a plain Transformer.**
The brief says "subclass ``pyspark.ml.Transformer`` to compute Q1/Q3 bounds
and clip". A Transformer, though, only has ``_transform`` - so it would have
to recompute its quantiles from whatever DataFrame it is handed. Applied to
the test split (or to a single streaming micro-batch, where the quantiles are
meaningless) it would fit itself to that data. Clipping bounds are learned
parameters, so they belong to the fit stage: ``OutlierIQRTruncator`` computes
them on the training split only and freezes them into
``OutlierIQRTruncatorModel``. That is the leak-proof reading of the
requirement, and it is what makes the saved model behave identically offline
and in streaming.

**Why target encoding is safe here.**
``TargetEncoder`` learns its category means inside ``_fit``, so when it sits
in a Pipeline that is fitted on the training split only, the test split can
never contribute to the encoding. Means are smoothed toward the global prior
so that a category seen twice does not get a confident encoding, and unseen
categories fall back to the prior instead of producing nulls.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pyspark import keyword_only
from pyspark.ml import Estimator, Model, Transformer
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.param.shared import HasInputCols, HasOutputCols
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.types import DoubleType

EARTH_RADIUS_MI = 3958.7613


# ---------------------------------------------------------------------------
# Arrow-vectorised UDFs
# ---------------------------------------------------------------------------
@F.pandas_udf(DoubleType())
def haversine_miles_udf(lat1: pd.Series, lon1: pd.Series,
                        lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Great-circle distance in statute miles.

    A Series-to-Series pandas_udf: Spark hands whole Arrow record batches to
    Python, so the trigonometry runs once over a numpy array at C speed
    instead of once per row through the Python interpreter.
    """
    p1 = np.radians(lat1.to_numpy(dtype="float64"))
    p2 = np.radians(lat2.to_numpy(dtype="float64"))
    dphi = p2 - p1
    dlam = np.radians(lon2.to_numpy(dtype="float64") - lon1.to_numpy(dtype="float64"))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    return pd.Series(EARTH_RADIUS_MI * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


@F.pandas_udf(DoubleType())
def schedule_speed_udf(distance: pd.Series, scheduled_minutes: pd.Series) -> pd.Series:
    """Average ground speed the published schedule implies, in mph.

    A domain-specific pressure metric: a leg scheduled at an unusually high
    implied speed has little slack, so any departure delay propagates
    straight through to arrival. Short legs are dominated by taxi and climb,
    which is exactly the signal we want the model to see.
    """
    d = distance.to_numpy(dtype="float64")
    m = scheduled_minutes.to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        mph = np.where(m > 0, d / (m / 60.0), np.nan)
    return pd.Series(mph)


# ---------------------------------------------------------------------------
# HaversineTransformer
# ---------------------------------------------------------------------------
class HaversineTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """Adds geodesic and schedule-pressure features via the Arrow UDFs above.

    The UDFs are deliberately invoked from a Pipeline stage rather than as
    loose DataFrame operations, so the feature engineering travels with the
    serialized ``PipelineModel`` and is reproduced identically at streaming
    inference time.
    """

    originLatCol = Param(Params._dummy(), "originLatCol", "origin latitude column",
                         typeConverter=TypeConverters.toString)
    originLonCol = Param(Params._dummy(), "originLonCol", "origin longitude column",
                         typeConverter=TypeConverters.toString)
    destLatCol = Param(Params._dummy(), "destLatCol", "destination latitude column",
                       typeConverter=TypeConverters.toString)
    destLonCol = Param(Params._dummy(), "destLonCol", "destination longitude column",
                       typeConverter=TypeConverters.toString)
    distanceCol = Param(Params._dummy(), "distanceCol", "published route distance column",
                        typeConverter=TypeConverters.toString)
    scheduledMinutesCol = Param(Params._dummy(), "scheduledMinutesCol",
                                "scheduled block time in minutes",
                                typeConverter=TypeConverters.toString)
    gcDistanceCol = Param(Params._dummy(), "gcDistanceCol", "output: great-circle miles",
                          typeConverter=TypeConverters.toString)
    routeDetourCol = Param(Params._dummy(), "routeDetourCol",
                           "output: published distance / great-circle distance",
                           typeConverter=TypeConverters.toString)
    scheduleSpeedCol = Param(Params._dummy(), "scheduleSpeedCol",
                             "output: implied schedule speed in mph",
                             typeConverter=TypeConverters.toString)

    @keyword_only
    def __init__(self, originLatCol="ORIGIN_LAT", originLonCol="ORIGIN_LON",
                 destLatCol="DEST_LAT", destLonCol="DEST_LON",
                 distanceCol="DISTANCE", scheduledMinutesCol="SCHEDULED_TIME",
                 gcDistanceCol="GC_DISTANCE_MI", routeDetourCol="ROUTE_DETOUR",
                 scheduleSpeedCol="SCHEDULE_SPEED_MPH"):
        super().__init__()
        self._setDefault(originLatCol="ORIGIN_LAT", originLonCol="ORIGIN_LON",
                         destLatCol="DEST_LAT", destLonCol="DEST_LON",
                         distanceCol="DISTANCE", scheduledMinutesCol="SCHEDULED_TIME",
                         gcDistanceCol="GC_DISTANCE_MI", routeDetourCol="ROUTE_DETOUR",
                         scheduleSpeedCol="SCHEDULE_SPEED_MPH")
        self._set(**self._input_kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        gc_col = self.getOrDefault(self.gcDistanceCol)
        out = dataset.withColumn(
            gc_col,
            haversine_miles_udf(
                F.col(self.getOrDefault(self.originLatCol)).cast("double"),
                F.col(self.getOrDefault(self.originLonCol)).cast("double"),
                F.col(self.getOrDefault(self.destLatCol)).cast("double"),
                F.col(self.getOrDefault(self.destLonCol)).cast("double"),
            ),
        )
        out = out.withColumn(
            self.getOrDefault(self.scheduleSpeedCol),
            schedule_speed_udf(
                F.col(self.getOrDefault(self.distanceCol)).cast("double"),
                F.col(self.getOrDefault(self.scheduledMinutesCol)).cast("double"),
            ),
        )
        # Ratio of published route distance to the geodesic: captures airway
        # dog-legs and congested terminal routings. Guarded against very short
        # legs, where the ratio explodes.
        out = out.withColumn(
            self.getOrDefault(self.routeDetourCol),
            F.when(F.col(gc_col) > 1.0,
                   F.col(self.getOrDefault(self.distanceCol)) / F.col(gc_col))
             .otherwise(F.lit(1.0)),
        )
        return out


# ---------------------------------------------------------------------------
# SignedLog1pTransformer
# ---------------------------------------------------------------------------
class SignedLog1pTransformer(Transformer, HasInputCols, HasOutputCols,
                             DefaultParamsReadable, DefaultParamsWritable):
    """Compresses heavy right skew while tolerating negative values.

    Applies ``sign(x) * log(1 + |x|)``. For non-negative columns this is
    exactly ``log1p(x)``; for columns such as DEPARTURE_DELAY, where early
    departures are legitimately negative, plain ``log1p`` would produce NaN.
    The signed form keeps the transform monotonic across zero, which is what
    the linear models need.
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None):
        super().__init__()
        self._set(**self._input_kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        for src, dst in zip(self.getInputCols(), self.getOutputCols()):
            x = F.col(src).cast("double")
            dataset = dataset.withColumn(
                dst, F.signum(x) * F.log1p(F.abs(x)))
        return dataset


# ---------------------------------------------------------------------------
# MaterializeTransformer
# ---------------------------------------------------------------------------
# DataFrames persisted by MaterializeCache, awaiting release. Module level
# because the persisting stage is buried inside a Pipeline the caller does not
# hold a reference to; release_caches() is called between tournament arms.
_PERSISTED: list = []


def release_caches() -> int:
    """Unpersist everything MaterializeCache has cached. Returns the count."""
    n = 0
    for df in _PERSISTED:
        try:
            df.unpersist()
            n += 1
        except Exception:  # noqa: BLE001 - a dead session must not stop cleanup
            pass
    _PERSISTED.clear()
    return n


class MaterializeCacheModel(Model, DefaultParamsReadable, DefaultParamsWritable):
    """Pass-through half of :class:`MaterializeCache`."""

    @keyword_only
    def __init__(self):
        super().__init__()

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset


class MaterializeCache(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    """Caches the intermediate DataFrame for the duration of ``Pipeline.fit``.

    ``Pipeline.fit`` fits each stage against the *lazy* output of the stages
    before it and never caches in between, so every fitted stage downstream
    re-executes the whole upstream chain - and the iterative learners
    (LinearRegression, PCA's SVD, LinearSVC, GBT) re-execute it once per
    iteration. Measured here on 114k rows, fitting without a cache took 51s
    against 35s with one.

    Caching during ``transform`` is the opposite trade: every stage after this
    point is a plain projection consumed in a single pass, so persisting only
    adds a materialisation the query never needed. The same measurement put a
    cached transform at 6.5s against 0.28s uncached - a 23x penalty paid on
    every scoring call, including each cross-validation fold's evaluation.

    Splitting it into an Estimator whose ``_fit`` persists and a Model whose
    ``_transform`` is a no-op gets both: ``persist`` marks the logical plan, so
    the stages fitted afterwards hit the cache, while the saved model carries
    no caching behaviour into scoring or streaming at all.
    """

    @keyword_only
    def __init__(self):
        super().__init__()

    def _fit(self, dataset: DataFrame) -> MaterializeCacheModel:
        if not dataset.isStreaming:
            from pyspark import StorageLevel
            dataset.persist(StorageLevel.MEMORY_AND_DISK)
            # Registered so the caller can release it. Nothing here knows when
            # the fit that wanted the cache is finished, and a tournament fits
            # this stage once per fold per grid point - without a release those
            # blocks accumulate for the whole run. MEMORY_AND_DISK means Spark
            # evicts rather than failing, so the symptom is memory pressure and
            # GC time, not a crash.
            _PERSISTED.append(dataset)
        return MaterializeCacheModel()


# ---------------------------------------------------------------------------
# OutlierIQRTruncator
# ---------------------------------------------------------------------------
class _IQRParams(Params):
    iqrMultiplier = Param(Params._dummy(), "iqrMultiplier",
                          "Tukey fence multiplier, conventionally 1.5",
                          typeConverter=TypeConverters.toFloat)
    relativeError = Param(Params._dummy(), "relativeError",
                          "approxQuantile relative error",
                          typeConverter=TypeConverters.toFloat)


class OutlierIQRTruncatorModel(Model, HasInputCols, HasOutputCols, _IQRParams,
                               DefaultParamsReadable, DefaultParamsWritable):
    """Clips columns to the Tukey fences frozen at fit time."""

    lowerBounds = Param(Params._dummy(), "lowerBounds", "fitted lower fences",
                        typeConverter=TypeConverters.toListFloat)
    upperBounds = Param(Params._dummy(), "upperBounds", "fitted upper fences",
                        typeConverter=TypeConverters.toListFloat)

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, lowerBounds=None,
                 upperBounds=None, iqrMultiplier=1.5, relativeError=0.001):
        super().__init__()
        self._setDefault(iqrMultiplier=1.5, relativeError=0.001)
        self._set(**self._input_kwargs)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        lo = self.getOrDefault(self.lowerBounds)
        hi = self.getOrDefault(self.upperBounds)
        for src, dst, low, high in zip(self.getInputCols(), self.getOutputCols(), lo, hi):
            # greatest/least are native Catalyst expressions - no Python round trip.
            dataset = dataset.withColumn(
                dst, F.least(F.greatest(F.col(src).cast("double"), F.lit(float(low))),
                             F.lit(float(high))))
        return dataset

    def bounds(self) -> dict[str, tuple[float, float]]:
        """Fitted fences, for reporting."""
        return dict(zip(self.getInputCols(),
                        zip(self.getOrDefault(self.lowerBounds),
                            self.getOrDefault(self.upperBounds))))


class OutlierIQRTruncator(Estimator, HasInputCols, HasOutputCols, _IQRParams,
                          DefaultParamsReadable, DefaultParamsWritable):
    """Learns Tukey fences [Q1 - k*IQR, Q3 + k*IQR] on the training split.

    Quantiles come from ``approxQuantile``, a distributed Greenwald-Khanna
    sketch: it needs one pass and O(1/eps) memory per partition rather than
    sorting 5.7M rows, and all columns are computed in a single pass.
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, iqrMultiplier=1.5,
                 relativeError=0.001):
        super().__init__()
        self._setDefault(iqrMultiplier=1.5, relativeError=0.001)
        self._set(**self._input_kwargs)

    def _fit(self, dataset: DataFrame) -> OutlierIQRTruncatorModel:
        cols = self.getInputCols()
        k = float(self.getOrDefault(self.iqrMultiplier))
        eps = float(self.getOrDefault(self.relativeError))

        casted = dataset.select([F.col(c).cast("double").alias(c) for c in cols])
        quantiles = casted.approxQuantile(cols, [0.25, 0.75], eps)

        lower, upper = [], []
        for col, qs in zip(cols, quantiles):
            # approxQuantile returns [] for a column that is entirely null,
            # so unpacking straight into (q1, q3) raises ValueError. An
            # all-null column has no fences to learn; pass it through
            # unclipped rather than failing the whole fit.
            if len(qs) < 2:
                print(f"  (OutlierIQRTruncator: {col} has no quantiles; "
                      f"passing through unclipped)")
                lower.append(float("-inf"))
                upper.append(float("inf"))
                continue
            q1, q3 = qs[0], qs[1]
            iqr = q3 - q1
            lower.append(float(q1 - k * iqr))
            upper.append(float(q3 + k * iqr))

        return OutlierIQRTruncatorModel(
            inputCols=cols, outputCols=self.getOutputCols(),
            lowerBounds=lower, upperBounds=upper,
            iqrMultiplier=k, relativeError=eps)


# ---------------------------------------------------------------------------
# TargetEncoder
# ---------------------------------------------------------------------------
class _TargetEncoderParams(Params):
    labelCol = Param(Params._dummy(), "labelCol", "target column used for encoding",
                     typeConverter=TypeConverters.toString)
    smoothing = Param(Params._dummy(), "smoothing",
                      "pseudo-count m pulling rare categories toward the prior",
                      typeConverter=TypeConverters.toFloat)


class TargetEncoderModel(Model, HasInputCols, HasOutputCols, _TargetEncoderParams,
                         DefaultParamsReadable, DefaultParamsWritable):
    """Applies frozen category means learned on the training split.

    The fitted mapping is carried as a JSON string Param. That keeps the model
    inside ``DefaultParamsWritable``, so ``PipelineModel.save()`` and
    ``.load()`` round-trip with no custom ``MLWriter``/``MLReader`` - which
    matters because the streaming job deserializes this model. A Parquet
    side-car would scale to far larger vocabularies, at the cost of custom IO
    code; here the vocabulary is a few thousand keys and JSON is comfortable.
    """

    mappingJson = Param(Params._dummy(), "mappingJson",
                        "JSON {column: {category: encoded_value}}",
                        typeConverter=TypeConverters.toString)
    prior = Param(Params._dummy(), "prior", "global target mean, used for unseen keys",
                  typeConverter=TypeConverters.toFloat)

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, labelCol="label",
                 mappingJson="{}", prior=0.0, smoothing=20.0):
        super().__init__()
        self._setDefault(labelCol="label", mappingJson="{}", prior=0.0, smoothing=20.0)
        self._set(**self._input_kwargs)

    def mapping(self) -> dict[str, dict[str, float]]:
        return json.loads(self.getOrDefault(self.mappingJson))

    # Above this many categories a literal map expression stops being the
    # cheaper option and the broadcast join is used instead.
    MAX_INLINE_CATEGORIES = 20000

    def _map_column(self, src: str, table: dict):
        """Cached ``create_map`` expression for one column.

        Encoding via a literal map rather than a join is ~60x cheaper to
        execute (0.10s vs 6s per pass over 114k rows here), because it is a
        plain projection: no broadcast exchange, no extra job, and nothing
        that behaves differently under Structured Streaming. The cost is
        driver-side plan construction - building ~8,500 literal Columns for
        the route vocabulary takes about 6s.

        Column expressions are independent of any particular DataFrame, so
        that construction cost is paid once per model and reused for every
        subsequent transform, which is what makes cross-validation affordable.
        """
        cache = getattr(self, "_map_cache", None)
        if cache is None:
            cache = self._map_cache = {}
        if src not in cache:
            pairs = []
            for k, v in table.items():
                pairs.append(F.lit(str(k)))
                pairs.append(F.lit(float(v)))
            cache[src] = F.create_map(*pairs)
        return cache[src]

    def _lookup_frame(self, spark, src: str, dst: str, table: dict) -> DataFrame:
        """Broadcast-join fallback for vocabularies too large to inline.

        Keyed on the session's own id rather than ``id(spark)``: CPython
        reuses object ids after garbage collection, so a new session could
        collide with a dead one's entry and hand back a DataFrame belonging to
        a stopped context. The frame is registered for release along with the
        MaterializeCache persists.
        """
        cache = getattr(self, "_lookup_cache", None)
        if cache is None:
            cache = self._lookup_cache = {}
        # applicationId is stable for a session and never reused across them.
        key = (spark.sparkContext.applicationId, src, dst)
        if key not in cache:
            frame = spark.createDataFrame(
                [(str(k), float(v)) for k, v in table.items()],
                schema=f"__te_key string, {dst} double")
            cache[key] = frame.persist()
            _PERSISTED.append(frame)
        return cache[key]

    def _transform(self, dataset: DataFrame) -> DataFrame:
        mapping = self.mapping()
        prior = float(self.getOrDefault(self.prior))

        for src, dst in zip(self.getInputCols(), self.getOutputCols()):
            table = mapping.get(src, {})
            if not table:
                dataset = dataset.withColumn(dst, F.lit(prior))
                continue

            key = F.col(src).cast("string")
            if len(table) <= self.MAX_INLINE_CATEGORIES:
                encoded = F.element_at(self._map_column(src, table), key)
                dataset = dataset.withColumn(
                    dst, F.coalesce(encoded, F.lit(prior)))
            else:
                # A stream-static left join with the static side on the right
                # is supported in Structured Streaming, so this path is still
                # valid at inference time.
                lookup = self._lookup_frame(dataset.sparkSession, src, dst, table)
                dataset = (dataset.join(F.broadcast(lookup),
                                        key == F.col("__te_key"), "left")
                           .drop("__te_key")
                           .withColumn(dst, F.coalesce(F.col(dst), F.lit(prior))))
        return dataset


class TargetEncoder(Estimator, HasInputCols, HasOutputCols, _TargetEncoderParams,
                    DefaultParamsReadable, DefaultParamsWritable):
    r"""Smoothed target encoding for high-cardinality categoricals.

    For category :math:`c` with :math:`n_c` rows and target sum
    :math:`\sum y`, the encoding is

    .. math::
        e_c = \frac{\sum_{i \in c} y_i + m \bar{y}}{n_c + m}

    where :math:`\bar{y}` is the global mean and :math:`m` the smoothing
    pseudo-count. As :math:`n_c \to \infty` the encoding approaches the raw
    category mean; for a category seen once or twice it stays close to the
    prior, which is what stops rare airports from being fitted to noise.

    Because the means are learned in ``_fit``, a Pipeline fitted on the
    training split cannot see test targets. The remaining subtlety is that a
    training row contributes to its own category's mean; with ~1,300 rows per
    category here that self-influence is negligible, but the strict remedy is
    out-of-fold encoding.
    """

    @keyword_only
    def __init__(self, inputCols=None, outputCols=None, labelCol="label",
                 smoothing=20.0):
        super().__init__()
        self._setDefault(labelCol="label", smoothing=20.0)
        self._set(**self._input_kwargs)

    def _fit(self, dataset: DataFrame) -> TargetEncoderModel:
        label = self.getOrDefault(self.labelCol)
        m = float(self.getOrDefault(self.smoothing))

        prior = float(dataset.agg(F.avg(F.col(label).cast("double"))).first()[0] or 0.0)

        mapping: dict[str, dict[str, float]] = {}
        for src in self.getInputCols():
            agg = (dataset.groupBy(F.col(src).cast("string").alias("k"))
                   .agg(F.sum(F.col(label).cast("double")).alias("s"),
                        F.count(F.lit(1)).alias("n"))
                   .withColumn("e", (F.col("s") + F.lit(m) * F.lit(prior))
                               / (F.col("n") + F.lit(m))))
            mapping[src] = {r["k"]: float(r["e"])
                            for r in agg.select("k", "e").collect()
                            if r["k"] is not None}

        return TargetEncoderModel(
            inputCols=self.getInputCols(), outputCols=self.getOutputCols(),
            labelCol=label, mappingJson=json.dumps(mapping), prior=prior,
            smoothing=m)
