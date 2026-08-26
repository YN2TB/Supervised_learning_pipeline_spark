"""Time each pipeline stage in isolation to find what is actually slow."""
from __future__ import annotations

import os
import sys
import time

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mllib_pipeline import CURATED, REG_LABEL, build_feature_stages  # noqa: E402
from spark_session import build_spark, path  # noqa: E402


def main() -> None:
    spark = build_spark("profile-stages", cores="8")
    try:
        df = spark.read.parquet(CURATED).sample(False, 0.02, seed=42).cache()
        n = df.count()
        print(f"rows: {n:,}\n")

        stages, _ = build_feature_stages(REG_LABEL, use_pca=True, pca_k=10)

        # Fit and apply one stage at a time, materialising after each so the
        # timing is attributed to the right stage rather than to a lazy plan.
        cur = df
        for st in stages:
            name = type(st).__name__
            t0 = time.time()
            model = st.fit(cur) if hasattr(st, "fit") else st
            t_fit = time.time() - t0

            t0 = time.time()
            out = model.transform(cur)
            cnt = out.count()
            t_tx = time.time() - t0
            print(f"{name:<28} fit={t_fit:7.2f}s  transform+count={t_tx:7.2f}s  rows={cnt:,}")
            cur = out.cache() if name in ("TargetEncoder",) else out

        t0 = time.time()
        lr = LinearRegression(featuresCol="features", labelCol=REG_LABEL, maxIter=50)
        lr.fit(cur)
        print(f"{'LinearRegression':<28} fit={time.time() - t0:7.2f}s")

        # Now the same thing as one Pipeline, which is how CrossValidator runs
        # it: any large gap between the two is plan-rebuild overhead.
        t0 = time.time()
        stages2, _ = build_feature_stages(REG_LABEL, use_pca=True, pca_k=10)
        m = Pipeline(stages=stages2 + [lr]).fit(df)
        print(f"\nwhole Pipeline.fit  = {time.time() - t0:7.2f}s")

        t0 = time.time()
        m.transform(df).count()
        print(f"whole transform+count = {time.time() - t0:7.2f}s")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
