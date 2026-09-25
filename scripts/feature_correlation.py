#!/usr/bin/env python
"""Correlation of the 20 numeric features with each other and with the target.

The columns are exactly what the VectorAssembler receives: the registered
model's stages 1 to 5 (clip, haversine, log, impute, target-encode) applied to
the training split, reconstructed with make_split() at cores="*". Scaling is
left out because it cannot change a correlation. The matrix is computed in the
JVM with Correlation.corr; nothing crosses into Python row by row.

Caveat worth keeping: te_origin, te_dest and te_route are fitted on these same
training rows, so their correlation with the target carries the (small,
smoothing-bounded) self-leakage described in REPORT.md.

Read-only: loads models/best_pipeline, writes two files.

    source ./env.sh && python scripts/feature_correlation.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from pyspark import StorageLevel  # noqa: E402
from pyspark.ml import PipelineModel  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.ml.stat import Correlation  # noqa: E402

from mllib_pipeline import CURATED, NUMERIC_FEATURES, REG_LABEL, make_split  # noqa: E402
from spark_session import build_spark, path  # noqa: E402

OUT_JSON = path("docs", "benchmarks", "feature_correlation.json")
OUT_PNG = path("docs", "benchmarks", "feature_correlation.png")

# Reference palette (dataviz skill): diverging blue <-> red, neutral midpoint.
NEG, MID, POS = "#2a78d6", "#f0efec", "#e34948"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1B2430", "#6B7580", "#D8D9D2"


def compute() -> tuple[np.ndarray, list[str], int]:
    spark = build_spark("feature-correlation")           # cores="*", as the tournament ran
    spark.sparkContext.setLogLevel("ERROR")
    model = PipelineModel.load(path("models", "best_pipeline"))
    assert type(model.stages[4]).__name__ == "TargetEncoderModel", type(model.stages[4]).__name__
    upto_te = PipelineModel(stages=model.stages[:5])     # stages 1-5
    train, _ = make_split(spark.read.parquet(CURATED))
    cols = list(NUMERIC_FEATURES) + [REG_LABEL]
    vec = (VectorAssembler(inputCols=cols, outputCol="v")
           .transform(upto_te.transform(train).select(*cols))
           .select("v").persist(StorageLevel.MEMORY_AND_DISK))
    rows = vec.count()
    corr = Correlation.corr(vec, "v", "pearson").first()[0].toArray()
    vec.unpersist()
    spark.stop()
    return corr, cols, rows


def draw(corr: np.ndarray, cols: list[str], rows: int) -> None:
    labels = [c if c != REG_LABEL else "ARRIVAL_DELAY (target)" for c in cols]
    # Strict lower triangle: the diagonal is all 1.00 and says nothing, so drop
    # the first row and last column, which would otherwise hold only diagonal cells.
    sub = corr[1:, :-1]
    row_labels, col_labels = labels[1:], labels[:-1]
    n = sub.shape[0]
    cmap = LinearSegmentedColormap.from_list("diverging", [NEG, MID, POS])
    masked = np.ma.array(sub, mask=np.triu(np.ones_like(sub, dtype=bool), k=1))

    fig, ax = plt.subplots(figsize=(13, 11), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(masked, cmap=cmap, vmin=-1, vmax=1)
    for i in range(n):
        for j in range(i + 1):
            v = sub[i, j]
            text = f"{v:.2f}"
            if text == "-0.00":
                text = "0.00"
            ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                    color="#FAFAF8" if abs(v) >= 0.6 else INK)
    # a 2px surface gap between cells keeps them legible as separate marks
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=8, color=INK, family="monospace")
    ax.set_yticklabels(row_labels, fontsize=8, color=INK, family="monospace")
    ax.get_yticklabels()[-1].set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cb.set_label("Pearson correlation", color=MUTED, fontsize=9)
    ax.set_title(f"Numeric features and the target, training split ({rows:,} rows)",
                 loc="left", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    corr, cols, rows = compute()
    target = cols.index(REG_LABEL)
    feats = [c for c in cols if c != REG_LABEL]
    pairs = [(cols[i], cols[j], float(corr[i, j]))
             for i in range(len(feats)) for j in range(i + 1, len(feats))]
    pairs.sort(key=lambda p: -abs(p[2]))
    abs_pairs = np.array([abs(p[2]) for p in pairs])
    result = {
        "source": "models/best_pipeline stages 1-5 on the training split (make_split, cores='*'); "
                  "pre-scaling values, Pearson, computed with Correlation.corr",
        "rows": rows,
        "columns": cols,
        "matrix": [[round(float(x), 4) for x in r] for r in corr],
        "with_target": sorted(({"feature": c, "r": round(float(corr[cols.index(c), target]), 4)} for c in feats),
                              key=lambda d: -abs(d["r"])),
        "feature_pairs": {"count": len(pairs), "median_abs_r": round(float(np.median(abs_pairs)), 4),
                          "top10": [{"a": a, "b": b, "r": round(r, 4)} for a, b, r in pairs[:10]]},
    }
    with open(OUT_JSON, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    draw(corr, cols, rows)
    print("CORR rows", rows, "| pairs", len(pairs), "| median |r|", result["feature_pairs"]["median_abs_r"])
    print("CORR top with target:", result["with_target"][:5])
    print("CORR top pairs:", result["feature_pairs"]["top10"][:5])
    print(f"CORR wrote {OUT_JSON} and {OUT_PNG}")


if __name__ == "__main__":
    main()
