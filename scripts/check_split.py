#!/usr/bin/env python
"""Assert the train/test split did not move.

mllib_pipeline used to add ``label_glm`` to the frame BEFORE calling
randomSplit; it now splits first and derives the offset from the training half
(so nothing fitted touches test data). randomSplit draws per row from the
input's existing partitioning and a projection is a narrow transformation, so
the split should be unchanged - but "should be" is not good enough here. The
14 completed tournament arms and every number in docs/REPORT.md are only
comparable to each other while the split is identical, so this checks it
directly rather than reasoning about it.

    source ./env.sh && python scripts/check_split.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import functions as F  # noqa: E402

from mllib_pipeline import CURATED, REG_LABEL, SPLIT_SEED, make_split  # noqa: E402
from spark_session import build_spark  # noqa: E402


def fingerprint(df) -> tuple[int, int]:
    """Order-independent identity of a row set.

    xxhash64 over every column, summed. Sensitive to which rows are present,
    blind to the order they arrive in - which is what we want, since partition
    ordering is not guaranteed stable between plans.
    """
    cols = [c for c in df.columns if c != "label_glm"]
    h = df.select(F.xxhash64(*[F.col(c).cast("string") for c in cols]).alias("h"))
    row = h.agg(F.sum(F.col("h").cast("decimal(38,0)")).alias("s"),
                F.count("*").alias("n")).first()
    return int(row["s"] or 0), int(row["n"])


def main() -> int:
    spark = build_spark("check-split", cores="8")
    spark.sparkContext.setLogLevel("ERROR")
    try:
        df = spark.read.parquet(CURATED)

        # OLD behaviour: label added first, then split.
        min_all = df.agg(F.min(REG_LABEL)).first()[0]
        off_all = float(abs(min(min_all, 0.0)) + 1.0)
        old = df.withColumn("label_glm", F.col(REG_LABEL) + F.lit(off_all))
        old_train, old_test = old.randomSplit([0.8, 0.2], seed=SPLIT_SEED)

        # NEW behaviour: split first, offset from train only.
        new_train, new_test = make_split(df, 1.0)
        min_train = new_train.agg(F.min(REG_LABEL)).first()[0]
        off_train = float(abs(min(min_train, 0.0)) + 1.0)

        ok = True
        for name, a, b in (("train", old_train, new_train),
                           ("test", old_test, new_test)):
            fa, na = fingerprint(a)
            fb, nb = fingerprint(b)
            same = (fa, na) == (fb, nb)
            ok &= same
            print(f"  {name}: old n={na:,} new n={nb:,}  "
                  f"{'IDENTICAL' if same else 'DIFFERENT - RESULTS INVALIDATED'}")

        print(f"\n  glm_offset  full-frame={off_all:.1f}  train-only={off_train:.1f}"
              f"  {'(unchanged)' if off_all == off_train else '(changed)'}")
        if off_all != off_train:
            print("  Note: a different offset is harmless - RMSE, MAE and R^2 are")
            print("  all invariant to a common shift of prediction and target.")

        print("\nSPLIT UNCHANGED - existing arms remain valid" if ok
              else "\nSPLIT MOVED - the completed arms are no longer comparable")
        return 0 if ok else 1
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
