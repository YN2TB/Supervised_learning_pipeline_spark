"""Feed held-out flights into the streaming inference job as JSON files.

    source ./env.sh && python scripts/make_stream_events.py --batches 5 --rows 40

When curated data is present, rows are drawn from the *test* split using the
same seed as training, so the records being scored are genuinely unseen by the
model. On a clean submission checkout, a deterministic, schema-compatible
SEA-to-MCO fallback is emitted so streaming can be demonstrated before the
592 MB Kaggle download. Each batch is written
as one JSON-lines file into stream_input/, which is what the file-source
stream in inference.py picks up (one file per trigger).

Written with plain Python file IO rather than Spark's JSON writer so that each
batch lands as a single complete file: a partitioned Spark write would drop
several part-files plus _SUCCESS into the watched directory at once, and the
stream would treat every one of them as a separate batch.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spark_session import build_spark, path  # noqa: E402

STREAM_IN = path("stream_input")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=5)
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--interval", type=float, default=4.0)
    ap.add_argument("--clean", action="store_true",
                    help="empty stream_input/ first")
    args = ap.parse_args()

    os.makedirs(STREAM_IN, exist_ok=True)
    if args.clean:
        for f in os.listdir(STREAM_IN):
            os.remove(os.path.join(STREAM_IN, f))
        print(f"cleared {STREAM_IN}")

    meta_path = os.path.join(path("models"), "input_schema.json")
    if not os.path.exists(meta_path):
        raise SystemExit(f"missing {meta_path} - run mllib_pipeline.py first")
    with open(meta_path) as fh:
        fields = json.load(fh)["fields"]
    cols = [f["name"] for f in fields]

    from mllib_pipeline import CURATED, SPLIT_SEED
    spark = None
    if os.path.exists(CURATED):
        spark = build_spark("make-stream-events", cores="4")
        try:
            df = spark.read.parquet(CURATED)
            # Same seed and ratio as training, so this really is the held-out side.
            _, test = df.randomSplit([0.8, 0.2], seed=SPLIT_SEED)

            need = args.batches * args.rows
            frac = min(1.0, (need * 40.0) / max(test.count(), 1))
            pool = test.sample(False, frac, seed=int(time.time()) % 10000) \
                       .select(*cols).limit(need).collect()
            pool = [row.asDict() for row in pool]
            print(f"drew {len(pool)} held-out rows across {len(cols)} columns")
        finally:
            spark.stop()
    else:
        base = {
            "DAY": 15, "DAY_OF_WEEK": 3, "AIRLINE": "AS", "FLIGHT_NUMBER": 123,
            "TAIL_NUMBER": "N123AS", "ORIGIN_AIRPORT": "SEA",
            "DESTINATION_AIRPORT": "MCO", "SCHEDULED_DEPARTURE": "0830",
            "DEPARTURE_TIME": "0842", "DEPARTURE_DELAY": 12, "TAXI_OUT": 16,
            "WHEELS_OFF": "0858", "SCHEDULED_TIME": 330, "DISTANCE": 2554,
            "SCHEDULED_ARRIVAL": "1700", "MONTH": 7, "ORIGIN_LAT": 47.45,
            "ORIGIN_LON": -122.31, "DEST_LAT": 28.43, "DEST_LON": -81.31,
            "AIRLINE_NAME": "Alaska Airlines", "SCHED_DEP_MIN": 510,
            "SCHED_ARR_MIN": 1020, "WHEELS_OFF_MIN": 538, "DEP_HOUR": 8,
            "IS_WEEKEND": 0, "ROUTE": "SEA-MCO",
        }
        need = args.batches * args.rows
        pool = []
        for i in range(need):
            row = {name: base.get(name) for name in cols}
            row["DEPARTURE_DELAY"] = 5 + i * 3
            row["TAXI_OUT"] = 12 + i % 8
            pool.append(row)
        print(f"curated data absent; generated {len(pool)} schema-compatible demo rows")

    for b in range(args.batches):
        chunk = pool[b * args.rows:(b + 1) * args.rows]
        if not chunk:
            break
        fname = os.path.join(STREAM_IN, f"events_{int(time.time() * 1000)}_{b}.json")
        tmp = fname + ".tmp"
        # Write to a temp name and rename: the stream must never see a
        # half-written file.
        with open(tmp, "w", encoding="utf-8") as fh:
            for row in chunk:
                fh.write(json.dumps(row, default=str) + "\n")
        os.replace(tmp, fname)
        print(f"  batch {b + 1}/{args.batches}: {len(chunk)} rows -> "
              f"{os.path.basename(fname)}")
        if b < args.batches - 1:
            time.sleep(args.interval)
    print("\ndone - watch the inference.py console for predictions")


if __name__ == "__main__":
    main()
