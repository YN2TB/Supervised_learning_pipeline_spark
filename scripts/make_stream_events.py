"""Feed held-out flights into the streaming inference job as JSON files.

    source ./env.sh && python scripts/make_stream_events.py --batches 5 --rows 40

Rows are drawn from the *test* split using the same seed as training, so the
records being scored are genuinely unseen by the model. Each batch is written
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
        cols = [f["name"] for f in json.load(fh)["fields"]]

    spark = build_spark("make-stream-events", cores="4")
    try:
        from mllib_pipeline import CURATED, SPLIT_SEED
        df = spark.read.parquet(CURATED)
        # Same seed and ratio as training, so this really is the held-out side.
        _, test = df.randomSplit([0.8, 0.2], seed=SPLIT_SEED)

        need = args.batches * args.rows
        frac = min(1.0, (need * 40.0) / max(test.count(), 1))
        pool = test.sample(False, frac, seed=int(time.time()) % 10000) \
                   .select(*cols).limit(need).collect()
        print(f"drew {len(pool)} held-out rows across {len(cols)} columns")

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
                    fh.write(json.dumps(row.asDict(), default=str) + "\n")
            os.replace(tmp, fname)
            print(f"  batch {b + 1}/{args.batches}: {len(chunk)} rows -> "
                  f"{os.path.basename(fname)}")
            if b < args.batches - 1:
                time.sleep(args.interval)
    finally:
        spark.stop()
    print("\ndone - watch the inference.py console for predictions")


if __name__ == "__main__":
    main()
