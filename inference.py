"""Phase 4: streaming inference against the serialized PipelineModel.

Demonstrates that the fitted pipeline scores live records with no retraining
and no code shared with the training job beyond the transformer definitions:

    # terminal 1
    source ./env.sh && python inference.py
    # terminal 2
    source ./env.sh && python scripts/make_stream_events.py --batches 5

The model is loaded either from disk (default) or straight from the MLflow
Model Registry's Production stage (``--from-registry``), which proves the
registered artifact is genuinely servable rather than just recorded.

Two things make this work that are easy to get wrong:

* ``custom_transformers`` must be importable when the model deserializes,
  because the saved metadata refers to those classes by qualified name. Under
  spark-submit that means passing ``--py-files custom_transformers.py``.
* A file-source stream cannot infer its schema, so the schema is read from
  ``models/input_schema.json``, published by the training run. If an incoming
  record does not match it, the mismatch shows up as nulls rather than as a
  crash - which is the scenario the ``--expect-nulls`` report surfaces.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from pyspark.ml import PipelineModel
from pyspark.sql import functions as F
from pyspark.sql.types import StructType

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import custom_transformers  # noqa: F401,E402  (registers classes for deserialization)
from spark_session import TRACKING_URI, build_spark, path  # noqa: E402

MODEL_DIR = path("models")
STREAM_IN = path("stream_input")
STREAM_OUT = path("stream_output")
CHECKPOINT = path("checkpoints", "inference")


def load_serving_schema() -> tuple[StructType, float]:
    """Rebuild the training-time input schema published next to the model."""
    meta_path = os.path.join(MODEL_DIR, "input_schema.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing {meta_path} - run mllib_pipeline.py first")
    with open(meta_path) as fh:
        meta = json.load(fh)
    ddl = ", ".join(f"`{f['name']}` {f['type']}" for f in meta["fields"])
    return StructType.fromDDL(ddl), float(meta.get("glm_offset", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-registry", action="store_true",
                    help="load models:/flight_delay_pipeline/Production instead of disk")
    ap.add_argument("--await-seconds", type=int, default=120,
                    help="how long to keep the query running; 0 means forever")
    ap.add_argument("--trigger-seconds", type=int, default=5)
    args = ap.parse_args()

    for d in (STREAM_IN, STREAM_OUT, CHECKPOINT):
        os.makedirs(d, exist_ok=True)

    spark = build_spark("streaming-inference", cores="4")
    schema, _ = load_serving_schema()
    print(f"serving schema: {len(schema.fields)} columns")

    if args.from_registry:
        import mlflow
        import mlflow.spark
        mlflow.set_tracking_uri(TRACKING_URI)
        uri = "models:/flight_delay_pipeline/Production"
        print(f"loading {uri}")
        model = mlflow.spark.load_model(uri)
    else:
        model_path = os.path.join(MODEL_DIR, "best_pipeline").replace("\\", "/")
        print(f"loading {model_path}")
        model = PipelineModel.load(model_path)
    print(f"loaded PipelineModel with {len(model.stages)} stages")

    events = (spark.readStream
              .schema(schema)
              .option("maxFilesPerTrigger", 1)
              .json(STREAM_IN.replace("\\", "/")))

    # The identical PipelineModel used for batch scoring, applied unchanged to
    # a streaming DataFrame: stateless, so every micro-batch is independent.
    scored = model.transform(events)

    keep = [c for c in ("AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "ROUTE",
                        "DEPARTURE_DELAY", "DISTANCE", "prediction")
            if c in scored.columns]
    scored = scored.select(*keep).withColumn(
        "scored_at", F.current_timestamp())

    console = (scored.writeStream
               .format("console")
               .outputMode("append")
               .option("truncate", "false")
               .option("numRows", 5)
               .trigger(processingTime=f"{args.trigger_seconds} seconds")
               .start())

    sink = (scored.writeStream
            .format("parquet")
            .outputMode("append")
            .option("path", STREAM_OUT.replace("\\", "/"))
            .option("checkpointLocation", CHECKPOINT.replace("\\", "/"))
            .trigger(processingTime=f"{args.trigger_seconds} seconds")
            .start())

    print(f"\nwatching {STREAM_IN} - drop JSON files there to score them")
    print(f"parquet sink: {STREAM_OUT}")
    try:
        if args.await_seconds:
            sink.awaitTermination(args.await_seconds)
            console.awaitTermination(1)
        else:
            spark.streams.awaitAnyTermination()
    finally:
        for q in (console, sink):
            if q.isActive:
                q.stop()
        total = sum(p["numInputRows"] for p in sink.recentProgress) if sink.recentProgress else 0
        print(f"\nrows scored this session: {total}")
        spark.stop()


if __name__ == "__main__":
    main()
