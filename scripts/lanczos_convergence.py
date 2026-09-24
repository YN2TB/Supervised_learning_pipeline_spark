#!/usr/bin/env python
"""Measure how Lanczos and ARPACK converge on the covariance the PCA arms decompose.

docs/REPORT.md A2 explains dist-eigs as ARPACK's implicitly restarted Lanczos,
and slide 25 shows the three-term recurrence. The figures quoted there (how many
products each eigenvalue needs, how many passes ARPACK makes at Spark's fixed
ncv) are measured here and persisted, rather than quoted from a session log.

The matrix is the one the tournament's PCA stage actually sees: the registered
model's stages 1 to 11 applied to the training split, reconstructed with
make_split() at cores="*" exactly as CLAUDE.md requires, then centred. The
covariance is formed once by Spark; everything after that is NumPy and SciPy on a
77 x 77 matrix. scipy.sparse.linalg.eigsh is an independent implementation of
the ARPACK routine (dsaupd) that Spark's EigenValueDecomposition.symmetricEigs
calls, so counting its matrix-vector products counts the passes dist-eigs would
make over the data.

Read-only: loads models/best_pipeline, writes one JSON file.

    source ./env.sh && python scripts/lanczos_convergence.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark import StorageLevel  # noqa: E402
from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.ml.stat import Correlation, Summarizer  # noqa: E402
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, LinearOperator, eigsh  # noqa: E402

from mllib_pipeline import CURATED, make_split  # noqa: E402
from spark_session import build_spark, path  # noqa: E402

OUT = path("docs", "benchmarks", "lanczos_convergence.json")
K = 10                 # the brief's k
TOL, MAX_ITER = 1e-10, 300   # what RowMatrixPCA passes to computeSVD
REL = 1e-6             # "converged" for the hand-rolled comparisons below


def covariance_of_training_split() -> tuple[np.ndarray, int]:
    spark = build_spark("lanczos-convergence")          # cores="*", as the tournament ran
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(path("models", "best_pipeline"))
    # Stages 1 to 11: everything up to and including VarianceThresholdSelector,
    # so the rows are exactly what stage 12 (PCA) was fitted on.
    upto_selector = PipelineModel(stages=model.stages[:11])
    train, _ = make_split(spark.read.parquet(CURATED))
    feats = (upto_selector.transform(train).select("features_selected")
             .persist(StorageLevel.MEMORY_AND_DISK))
    # Stay in the JVM. A Python RDD over 4.56M rows, next to the Arrow UDF's own
    # workers, crashed a Python worker; Correlation.corr runs
    # RowMatrix.computeCovariance on the Scala side and never leaves it.
    stats = feats.select(Summarizer.metrics("variance", "count")
                         .summary(feats["features_selected"]).alias("s")).first()["s"]
    sd = np.sqrt(stats["variance"].toArray())
    m = int(stats["count"])
    corr = Correlation.corr(feats, "features_selected", "pearson").first()[0].toArray()
    feats.unpersist()
    spark.stop()
    return corr * np.outer(sd, sd), m                    # covariance, divided by m - 1


def lanczos(C: np.ndarray, steps: int, seed: int = 0):
    """The three-term recurrence on slide 25, with full re-orthogonalisation.

    Returns the alphas, betas and the Ritz values after every step.
    """
    n = C.shape[0]
    v = np.random.default_rng(seed).standard_normal(n)
    V = [v / np.linalg.norm(v)]
    alphas, betas, ritz = [], [], []
    v_prev, beta = np.zeros(n), 0.0
    for _ in range(steps):
        w = C @ V[-1]                        # M v_j: the one distributed pass
        a = float(V[-1] @ w)                 # alpha_j
        w = w - a * V[-1] - beta * v_prev    # subtract what is already known
        for u in V:                          # ARPACK re-orthogonalises too
            w -= (u @ w) * u
        b = float(np.linalg.norm(w))         # beta_{j+1}
        alphas.append(a)
        T = np.diag(alphas) + np.diag(betas, 1) + np.diag(betas, -1)
        ritz.append(np.sort(np.linalg.eigvalsh(T))[::-1])
        if b < 1e-12:
            break
        betas.append(b)
        v_prev, beta = V[-1], b
        V.append(w / b)
    return alphas, betas, ritz


def arpack_passes(C: np.ndarray, ncv: int, v0=None) -> dict:
    count = [0]

    def matvec(z):
        count[0] += 1
        return C @ z

    op = LinearOperator(C.shape, matvec=matvec, dtype=float)
    try:
        vals = eigsh(op, k=K, ncv=ncv, which="LM", tol=TOL, maxiter=MAX_ITER,
                     return_eigenvectors=False, v0=v0)
        return {"converged": True, "passes": count[0], "eigenvalues": sorted(map(float, vals), reverse=True)}
    except (ArpackNoConvergence, ArpackError) as exc:
        return {"converged": False, "passes": count[0], "error": str(exc)[:200]}


def main() -> None:
    C, m = covariance_of_training_split()
    n = C.shape[0]
    lam = np.sort(np.linalg.eigvalsh(C))[::-1]
    print(f"LCZ rows {m:,}  n {n}  trace {lam.sum():.3f}  top-{K} share {100 * lam[:K].sum() / lam.sum():.3f}%")

    alphas, betas, ritz = lanczos(C, steps=n)

    def steps_to(i: int):
        for j, r in enumerate(ritz):
            if len(r) > i and abs(r[i] - lam[i]) / lam[i] < REL:
                return j + 1
        return None

    x = np.random.default_rng(1).standard_normal(n)
    power = None
    for it in range(1, 5000):
        x = C @ x
        x /= np.linalg.norm(x)
        if abs(x @ C @ x - lam[0]) / lam[0] < REL:
            power = it
            break

    runs = {f"ncv={ncv}": arpack_passes(C, ncv) for ncv in (20, 30, 40)}
    # Spark starts ARPACK from a random residual, so check the fixed-ncv case
    # from several random starts rather than trusting one.
    starts = [arpack_passes(C, 20, v0=np.random.default_rng(s).standard_normal(n)) for s in range(10)]

    result = {
        "source": "models/best_pipeline stages 1-11 on the training split (make_split, cores='*')",
        "rows": m,
        "n": n,
        "k": K,
        "eigenvalues_top12": [round(float(x), 6) for x in lam[:12]],
        "explained_variance_top10_pct": [round(100 * float(x) / float(lam.sum()), 3) for x in lam[:K]],
        "retained_pct": round(100 * float(lam[:K].sum() / lam.sum()), 3),
        "ratio_l1_l2": round(float(lam[0] / lam[1]), 5),
        "ratio_l10_l11": round(float(lam[9] / lam[10]), 5),
        "ratio_l13_l14": round(float(lam[12] / lam[13]), 5),
        "lanczos_first_steps": [{"step": j + 1, "alpha": round(alphas[j], 4), "beta_next": round(betas[j], 4)}
                                for j in range(min(5, len(betas)))],
        "lanczos_products_to_converge": {f"lambda_{i + 1}": steps_to(i) for i in (0, 1, 4, 9)},
        "power_iteration_products_lambda_1": power,
        "convergence_tolerance_relative": REL,
        "arpack": {"tol": TOL, "maxiter": MAX_ITER, "which": "LM", **runs},
        "arpack_ncv20_random_starts": {
            "converged": sum(s["converged"] for s in starts),
            "of": len(starts),
            "passes_min": min(s["passes"] for s in starts),
            "passes_max": max(s["passes"] for s in starts),
        },
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print("LCZ " + json.dumps(result))
    print(f"LCZ wrote {OUT}")


if __name__ == "__main__":
    main()
