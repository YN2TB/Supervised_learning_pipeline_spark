#!/usr/bin/env python
"""Reproduce why every dist-eigs PCA arm failed in the tournament, and check the fix.

The failures recorded in docs/benchmarks/tournament_failures.json were ARPACK
``info = 3`` ("No shifts could be applied") and ``Index 1540 out of bounds for
length 1540``. They were first blamed on the spectrum: two eigenvalues close
together at the cut, with ncv fixed at 20. That is wrong, and this script is the
evidence.

The operator is a 77-row matrix A_small = diag(sqrt(lambda)) V^T whose Gramian is
exactly the tournament covariance (the registered model's stages 1 to 11 on the
training split). Same spectrum, same Spark ARPACK, same ncv, tolerance and
iteration cap; the only thing varied is how many solves run at once.

  1. raw RowMatrix.computeSVD, one solve at a time, then four at once from four
     threads, which is what CrossValidator(parallelism=4) does to the PCA stage
  2. RowMatrixPCA, which now serialises the solve behind _ARPACK_LOCK, fitted
     from four threads at once and compared with a single-threaded fit. This
     uses local-eigs: the same lock and the same ARPACK, and part 1 shows it
     breaks under concurrency too. dist-eigs through RowMatrixPCA re-serialises
     every row through a Python worker on each ARPACK product (the Python-side
     cache does not reach the JVM rows), so four concurrent fits take hours.

Spark's ARPACK (dev.ludovic.netlib F2jARPACK) keeps its state in static fields,
so concurrent solves corrupt each other. Read-only: writes one JSON file.

    source ./env.sh && python scripts/arpack_concurrency.py
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from functools import partial

import numpy as np

print = partial(print, flush=True)  # noqa: A001 - progress survives an interrupted run

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.ml.linalg import Vectors as MLVectors  # noqa: E402
from pyspark.mllib.linalg import Vectors as MLLibVectors  # noqa: E402
from pyspark.mllib.linalg.distributed import RowMatrix  # noqa: E402

from custom_transformers import RowMatrixPCA  # noqa: E402
from lanczos_convergence import covariance_of_training_split  # noqa: E402
from spark_session import build_spark, path  # noqa: E402

OUT = path("docs", "benchmarks", "arpack_concurrency.json")
K, MAX_ITER, TOL = 10, 300, 1e-10
THREADS, ROUNDS = 4, 3


def error_kind(exc: Exception) -> str:
    text = str(exc)
    if "info = 3" in text:
        return "info = 3 (No shifts could be applied)"
    if "info = 1" in text:
        return "info = 1 (Maximum number of iterations taken)"
    if "out of bounds" in text:
        return "Index out of bounds"
    java = getattr(exc, "java_exception", None)
    if java is not None:
        return str(java.getClass().getName())
    first = text.split("\n")[0][:80] if text else ""
    return f"{type(exc).__name__}: {first}"


def main() -> None:
    C, m = covariance_of_training_split()
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]
    # C is singular; eigh returns a few eigenvalues of about -3e-13, and sqrt of
    # those is NaN, which would poison the operator. Clip them to zero.
    A = np.sqrt(np.clip(lam, 0.0, None))[:, None] * V.T
    assert np.isfinite(A).all()

    spark = build_spark("arpack-concurrency", cores="8")
    spark.sparkContext.setLogLevel("ERROR")
    sc = spark.sparkContext
    impl = sc._jvm.dev.ludovic.netlib.arpack.ARPACK.getInstance().getClass().getName()
    print(f"ACC ARPACK implementation: {impl}")

    def solve_raw(mode: str) -> str:
        rm = RowMatrix(sc.parallelize([MLLibVectors.dense(r) for r in A], 4))
        jm = rm._java_matrix_wrapper._java_model   # keep rm alive: py4j frees jm with it
        jm.rows().cache()
        jm.rows().count()
        try:
            svd = jm.computeSVD(K, False, 1e-9, MAX_ITER, TOL, mode)
            s2 = np.array([float(x) for x in svd.s().toArray()]) ** 2
            if len(s2) < K:
                # No exception: Spark logs a warning and returns what converged.
                return f"returned {len(s2)} of {K} values, no error raised"
            err = float(np.max(np.abs(s2 - lam[:K]) / lam[:K]))
            return "ok" if err < 1e-8 else f"wrong answer (rel err {err:.1e})"
        except Exception as exc:  # noqa: BLE001 - classifying the failure is the point
            return error_kind(exc)

    def tally(outcomes: list[str]) -> dict:
        out: dict[str, int] = {}
        for o in outcomes:
            out[o] = out.get(o, 0) + 1
        return out

    raw: dict[str, dict] = {}
    for mode in ("dist-eigs", "local-eigs"):
        alone = [solve_raw(mode) for _ in range(ROUNDS)]
        together: list[str] = []
        for _ in range(ROUNDS):
            res = [""] * THREADS
            ts = [threading.Thread(target=lambda i=i: res.__setitem__(i, solve_raw(mode)))
                  for i in range(THREADS)]
            for t in ts:
                t.start()
            for t in ts:
                t.join()
            together += res
        raw[mode] = {"one_at_a_time": tally(alone), f"{THREADS}_at_once": tally(together)}
        print(f"ACC raw computeSVD {mode:10s} alone {raw[mode]['one_at_a_time']}  "
              f"{THREADS} at once {raw[mode][f'{THREADS}_at_once']}")

    # --- the fix: RowMatrixPCA under concurrency ------------------------------
    df = spark.createDataFrame([(MLVectors.dense(r),) for r in A], ["features_selected"])
    df.cache().count()
    est = RowMatrixPCA(inputCol="features_selected", outputCol="pc", k=K, svdMode="local-eigs")
    ref = est.fit(df)
    ref_ev = ref.explainedVariance.toArray()
    ref_pc = ref.pc.toArray()

    fixed: list[str] = []
    for _ in range(ROUNDS):
        res = [""] * THREADS

        def fit(i: int) -> None:
            try:
                mdl = est.fit(df)
                cos = np.abs(np.sum(mdl.pc.toArray() * ref_pc, axis=0))
                same = (np.max(np.abs(mdl.explainedVariance.toArray() - ref_ev)) < 1e-10
                        and np.min(cos) > 1 - 1e-8)
                res[i] = "ok" if same else "differs from single-threaded fit"
            except Exception as exc:  # noqa: BLE001
                res[i] = error_kind(exc)

        ts = [threading.Thread(target=fit, args=(i,)) for i in range(THREADS)]
        t0 = time.time()
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        fixed += res
        print(f"ACC RowMatrixPCA local-eigs, {THREADS} at once: {tally(res)}  ({time.time() - t0:.0f}s)")
    spark.stop()

    result = {
        "operator": "77-row matrix whose Gramian is the tournament covariance "
                    f"(training split, {m:,} rows)",
        "arpack_implementation": impl,
        "k": K, "ncv": min(2 * K, len(lam)), "tol": TOL, "maxIter": MAX_ITER,
        "raw_computeSVD": raw,
        f"RowMatrixPCA_with_lock_local_eigs_{THREADS}_at_once": tally(fixed),
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print(f"ACC wrote {OUT}")


if __name__ == "__main__":
    main()
