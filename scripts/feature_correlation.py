#!/usr/bin/env python
"""Correlation matrices of the candidate features and both targets, before and
after the pipeline's transformations.

    --stage raw           the columns as they sit in the curated data: distance,
                          schedule, calendar, coordinates. DEPARTURE_DELAY and
                          TAXI_OUT are included as a reference only, to show
                          what the wheels-off model leaned on; the pre-departure
                          model may not use them.
    --stage transformed   what the VectorAssembler receives for the
                          predeparture_all feature set (5% sample of train):
                          log transforms, the
                          Haversine-derived columns, the frequency encodings,
                          all imputed; stages fitted on the training split.
    --stage wheelsoff     the September tournament's 20 numeric features after
                          its stages 1 to 5 (the original version of this script).

Both targets are in every matrix: label_delay (minutes) and label_severe (the
0/1 "more than 30 minutes late" label). Computed on the training split
(make_split, cores="*") in one Spark SQL aggregation. Scaling is left out
because it cannot change a correlation.

    source ./env.sh && python scripts/feature_correlation.py --stage raw
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from pyspark.ml import Pipeline, PipelineModel  # noqa: E402
from pyspark.ml.feature import VectorAssembler  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402

from mllib_pipeline import (CLF_LABEL, CURATED, FEATURE_SETS, REG_LABEL,  # noqa: E402
                            WHEELSOFF_NUMERIC, _predeparture_stages, make_split)
from spark_session import build_spark, path  # noqa: E402

# Reference palette (dataviz skill): diverging blue <-> red, neutral midpoint.
NEG, MID, POS = "#2a78d6", "#f0efec", "#e34948"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1B2430", "#6B7580", "#D8D9D2"

RAW_COLUMNS = [
    "DISTANCE", "SCHEDULED_TIME", "SCHED_DEP_MIN", "SCHED_ARR_MIN", "DEP_HOUR",
    "MONTH", "DAY", "DAY_OF_WEEK", "IS_WEEKEND",
    "ORIGIN_LAT", "ORIGIN_LON", "DEST_LAT", "DEST_LON",
    "DEPARTURE_DELAY", "TAXI_OUT",          # reference: known only at wheels-off
]
LABELS = [REG_LABEL, CLF_LABEL]
TRANSFORMED_SAMPLE = 0.05
TITLES = {
    "raw": "Candidate columns before transformation",
    "transformed": "Pre-departure features after transformation",
    "wheelsoff": "Wheels-off features (September model)",
}


def out_paths(stage: str) -> tuple[str, str]:
    stem = "feature_correlation" if stage == "wheelsoff" else f"feature_correlation_{stage}"
    return (path("docs", "benchmarks", stem + ".json"),
            path("docs", "benchmarks", stem + ".png"))


def frame_for(stage: str, train):
    if stage == "raw":
        cols = RAW_COLUMNS
        return train.select(*[F.col(c).cast("double") for c in cols + LABELS]), cols
    if stage == "transformed":
        # A 5% sample of train: ~228k rows, standard error of r about 0.002,
        # ample for judging redundancy. At 25% and at full size the Python
        # worker running the Haversine Arrow UDF died mid-stage every time
        # (EOFError / connection reset), under Correlation.corr and under a
        # plain SQL aggregation alike; at 5% it runs. Unresolved; the pruned
        # feature set the model uses does not include that UDF.
        train = train.sample(False, TRANSFORMED_SAMPLE, seed=1)
        cols = list(FEATURE_SETS["predeparture_all"]["numeric"])
        model = Pipeline(stages=_predeparture_stages(cols)).fit(train)
        return model.transform(train).select(*cols, *LABELS), cols
    model = PipelineModel.load(path("models", "best_pipeline"))
    assert type(model.stages[4]).__name__ == "TargetEncoderModel", type(model.stages[4]).__name__
    cols = list(WHEELSOFF_NUMERIC)
    return PipelineModel(stages=model.stages[:5]).transform(train).select(*cols, *LABELS), cols


def compute(stage: str):
    spark = build_spark("feature-correlation")           # cores="*", as the tournament ran
    spark.sparkContext.setLogLevel("ERROR")
    train, _ = make_split(spark.read.parquet(CURATED))
    frame, feats = frame_for(stage, train)
    cols = feats + LABELS
    # One SQL aggregation: every pairwise Pearson r plus the row count, in a
    # single pass. Correlation.corr (which goes through an RDD) was tried first
    # and crashed the Python worker of the Haversine Arrow UDF three times out
    # of three on this plan, at full size and on a 25% sample alike.
    clean = frame.select(*[F.col(c).cast("double") for c in cols]).dropna()
    pairs = [(i, j) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    row = clean.agg(F.count(F.lit(1)).alias("n"),
                    *[F.corr(cols[i], cols[j]).alias(f"c{i}_{j}") for i, j in pairs]).first()
    rows = int(row["n"])
    corr = np.eye(len(cols))
    for i, j in pairs:
        corr[i, j] = corr[j, i] = row[f"c{i}_{j}"]
    spark.stop()
    return corr, cols, feats, rows


def draw(corr: np.ndarray, cols: list[str], rows: int, stage: str, png: str) -> None:
    rename = {REG_LABEL: "ARRIVAL_DELAY (min)", CLF_LABEL: "SEVERE (> 30 min, 0/1)"}
    labels = [rename.get(c, c) for c in cols]
    # Strict lower triangle: the diagonal is all 1.00 and says nothing, so drop
    # the first row and last column, which would otherwise hold only diagonal cells.
    sub = corr[1:, :-1]
    row_labels, col_labels = labels[1:], labels[:-1]
    n = sub.shape[0]
    cmap = LinearSegmentedColormap.from_list("diverging", [NEG, MID, POS])
    masked = np.ma.array(sub, mask=np.triu(np.ones_like(sub, dtype=bool), k=1))

    size = 4.5 + 0.48 * n
    fig, ax = plt.subplots(figsize=(size + 1.5, size), dpi=150)
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
    for t in ax.get_yticklabels()[-2:]:
        t.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=MUTED, length=0)
    cb.set_label("Pearson correlation", color=MUTED, fontsize=9)
    where = ("5% sample of the training split" if stage == "transformed"
             else "training split")
    ax.set_title(f"{TITLES[stage]}, {where} ({rows:,} rows)",
                 loc="left", fontsize=12, color=INK, pad=12)
    fig.tight_layout()
    fig.savefig(png, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(TITLES), default="transformed")
    stage = ap.parse_args().stage
    out_json, out_png = out_paths(stage)

    corr, cols, feats, rows = compute(stage)
    pairs = [(cols[i], cols[j], float(corr[i, j]))
             for i in range(len(feats)) for j in range(i + 1, len(feats))]
    pairs.sort(key=lambda p: -abs(p[2]))
    abs_pairs = np.array([abs(p[2]) for p in pairs])
    with_target = {
        lab: sorted(({"feature": c, "r": round(float(corr[cols.index(c), cols.index(lab)]), 4)}
                     for c in feats), key=lambda d: -abs(d["r"]))
        for lab in LABELS}
    result = {
        "source": f"stage={stage}, "
                  + ("5% sample of the " if stage == "transformed" else "")
                  + "training split (make_split, cores='*'); pre-scaling "
                  "values, Pearson, one Spark SQL aggregation",
        "rows": rows,
        "columns": cols,
        "matrix": [[round(float(x), 4) for x in r] for r in corr],
        "with_target": with_target[REG_LABEL],
        "with_severe": with_target[CLF_LABEL],
        "feature_pairs": {"count": len(pairs), "median_abs_r": round(float(np.median(abs_pairs)), 4),
                          "top10": [{"a": a, "b": b, "r": round(r, 4)} for a, b, r in pairs[:10]]},
    }
    with open(out_json, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    draw(corr, cols, rows, stage, out_png)
    print("CORR", stage, "rows", rows, "| pairs", len(pairs),
          "| median |r|", result["feature_pairs"]["median_abs_r"])
    print("CORR top with delay:", with_target[REG_LABEL][:5])
    print("CORR top with severe:", with_target[CLF_LABEL][:5])
    print("CORR top pairs:", result["feature_pairs"]["top10"][:6])
    print(f"CORR wrote {out_json} and {out_png}")


if __name__ == "__main__":
    main()
