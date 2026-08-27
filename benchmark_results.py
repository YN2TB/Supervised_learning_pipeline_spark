"""Phase 5: pull the tournament results out of MLflow and render the figures.

    source ./env.sh && python benchmark_results.py

Reads the tracking store rather than retraining, so every number in the report
traces back to a logged run. Writes docs/benchmarks/*.png and a markdown table
that docs/REPORT.md includes verbatim.

Chart conventions follow one house style: one measure per axis (never a second
y-scale), a legend whenever two series share a plot plus direct value labels,
recessive grid and axes, and text in ink colours rather than the series colour.
Colours are the first two slots of a CVD-validated categorical palette
(blue/orange), which clear both the colourblind and normal-vision separation
floors against this surface.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spark_session import TRACKING_URI, path  # noqa: E402

BENCH_DIR = path("docs", "benchmarks")

# --- palette (validated: adjacent CVD dE 9.2, normal-vision dE 27.6, light) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
SERIES = ["#2a78d6", "#eb6834"]      # blue, orange
SEQ = "#2a78d6"

REG_METRICS = ["rmse", "mae", "r2"]
CLF_METRICS = ["areaUnderROC", "areaUnderPR", "f1", "accuracy"]


def style_axes(ax, xlabel="", ylabel="", title=""):
    """Recessive chrome: the data should be the only assertive thing on screen."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=12)


def save(fig, name: str) -> str:
    out = os.path.join(BENCH_DIR, name)
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# ---------------------------------------------------------------------------
def load_runs(experiment: str) -> dict:
    """Prefer the MLflow store; fall back to the JSON the training run wrote."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()
        exp = client.get_experiment_by_name(experiment)
        if exp is None:
            raise RuntimeError(f"experiment {experiment!r} not found")
        rows = {}
        for r in client.search_runs([exp.experiment_id], max_results=500):
            name = r.data.tags.get("mlflow.runName", "")
            if "__" not in name or not r.data.metrics:
                continue          # skip autolog's nested CV children
            rows[name] = dict(r.data.metrics,
                              model=r.data.params.get("model", name.split("__")[0]),
                              arm=r.data.params.get("arm", name.split("__")[1]),
                              task=r.data.params.get("task", ""))
        if rows:
            print(f"loaded {len(rows)} runs from the MLflow store")
            return rows
        raise RuntimeError("no completed runs in the store")
    except Exception as exc:  # noqa: BLE001
        print(f"MLflow unavailable ({exc}); falling back to tournament_results.json")
        with open(os.path.join(BENCH_DIR, "tournament_results.json")) as fh:
            return json.load(fh)


# ---------------------------------------------------------------------------
def plot_explained_variance() -> None:
    src = os.path.join(BENCH_DIR, "pca_explained_variance.json")
    if not os.path.exists(src):
        print("  (no PCA variance file; skipping)")
        return
    with open(src) as fh:
        ev = np.asarray(json.load(fh), dtype=float)
    cum = np.cumsum(ev) * 100
    ks = np.arange(1, len(ev) + 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ks, cum, color=SEQ, linewidth=2.0, marker="o", markersize=6,
            markerfacecolor=SEQ, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.bar(ks, ev * 100, color=SEQ, alpha=0.18, width=0.55)

    # Label only the endpoint rather than every marker.
    ax.annotate(f"{cum[-1]:.1f}% at k={len(ev)}",
                xy=(ks[-1], cum[-1]), xytext=(-8, -18),
                textcoords="offset points", ha="right",
                color=INK, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks(ks)
    style_axes(ax, "principal component k", "variance explained (%)",
               "Cumulative variance retained by distributed PCA")
    ax.text(0.0, -0.18, "bars: individual component   line: cumulative",
            transform=ax.transAxes, color=INK_2, fontsize=8)
    save(fig, "pca_explained_variance.png")


def plot_importances() -> None:
    files = sorted(f for f in os.listdir(BENCH_DIR)
                   if f.startswith("importance_") and f.endswith("nopca.json"))
    if not files:
        print("  (no no-PCA importance files; skipping)")
        return
    for fname in files:
        with open(os.path.join(BENCH_DIR, fname)) as fh:
            payload = json.load(fh)
        names = payload["features"]
        vals = np.asarray(payload["importances"], dtype=float)
        n = min(15, len(vals))
        idx = np.argsort(vals)[::-1][:n][::-1]

        fig, ax = plt.subplots(figsize=(7.6, 0.36 * n + 1.6))
        ax.barh(range(n), vals[idx], color=SEQ, height=0.62)
        ax.set_yticks(range(n))
        ax.set_yticklabels([names[i] for i in idx], fontsize=9)
        span = vals[idx].max() or 1.0
        for y, v in enumerate(vals[idx]):
            ax.text(v + span * 0.012, y, f"{v:.3f}", va="center",
                    color=INK_2, fontsize=8)
        ax.set_xlim(0, span * 1.16)
        model = fname[len("importance_"):-len("__nopca.json")]
        style_axes(ax, "Gini importance", "",
                   f"Feature importance - {model.replace('_', ' ')} (no PCA)")
        save(fig, f"feature_importance_{model}.png")


def plot_model_comparison(runs: dict) -> None:
    """Regression and classification get their own figure - never one dual axis."""
    for task, metric, better, fname in (
        ("regression", "rmse", "lower is better", "model_comparison_regression.png"),
        ("classification", "areaUnderROC", "higher is better",
         "model_comparison_classification.png"),
    ):
        rows = {k: v for k, v in runs.items()
                if v.get("task") == task and metric in v}
        if not rows:
            continue
        models = sorted({v["model"] for v in rows.values()})
        arms = ["pca", "nopca"]
        vals = {a: [next((v[metric] for v in rows.values()
                          if v["model"] == m and v["arm"] == a), np.nan)
                    for m in models] for a in arms}

        y = np.arange(len(models))
        h = 0.36
        # Floor the height: with only one or two models the axes get so short
        # that thick bars and the legend fight for the same space.
        fig, ax = plt.subplots(figsize=(8.4, max(3.2, 0.72 * len(models) + 2.0)))
        for i, arm in enumerate(arms):
            off = (i - 0.5) * (h + 0.04)   # 2px-equivalent gap between bars
            ax.barh(y + off, vals[arm], height=h, color=SERIES[i],
                    label="with PCA(k=10)" if arm == "pca" else "no PCA")
        finite = [v for a in arms for v in vals[a] if np.isfinite(v)]
        span = max(finite) if finite else 1.0
        for i, arm in enumerate(arms):
            off = (i - 0.5) * (h + 0.04)
            for j, v in enumerate(vals[arm]):
                if np.isfinite(v):
                    ax.text(v + span * 0.01, y[j] + off, f"{v:.3f}",
                            va="center", color=INK_2, fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([m.replace("_", " ") for m in models], fontsize=9)
        ax.set_xlim(0, span * 1.18)
        style_axes(ax, f"{metric}  ({better})", "",
                   f"Model tournament - {task}")
        # Legend sits above the axes, not inside them: anchored in the data
        # area it overlapped the right-hand value labels.
        leg = ax.legend(frameon=False, fontsize=9, loc="lower right",
                        bbox_to_anchor=(1.0, 1.01), ncol=2,
                        handlelength=1.2, handleheight=1.0, columnspacing=1.4)
        for t in leg.get_texts():
            t.set_color(INK_2)
        save(fig, fname)


def plot_residuals(runs: dict) -> None:
    """Predicted vs actual and the residual distribution for the winner."""
    model_path = os.path.join(path("models"), "best_pipeline")
    if not os.path.exists(model_path):
        print("  (no saved best_pipeline; skipping residuals)")
        return
    try:
        from pyspark.ml import PipelineModel
        import custom_transformers  # noqa: F401
        from mllib_pipeline import CURATED, REG_LABEL, SPLIT_SEED
        from spark_session import build_spark
        spark = build_spark("benchmark-residuals", cores="8")
        try:
            df = spark.read.parquet(CURATED)
            _, test = df.randomSplit([0.8, 0.2], seed=SPLIT_SEED)
            sample = test.sample(False, 0.02, seed=1)
            preds = (PipelineModel.load(model_path.replace("\\", "/"))
                     .transform(sample).select(REG_LABEL, "prediction").toPandas())
        finally:
            spark.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"  (residual plot skipped: {exc})")
        return

    actual = preds[REG_LABEL].to_numpy(float)
    pred = preds["prediction"].to_numpy(float)
    resid = actual - pred

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4))
    lo, hi = np.percentile(actual, [0.5, 99.5])
    ax = axes[0]
    ax.hexbin(actual, pred, gridsize=45, extent=(lo, hi, lo, hi),
              cmap="Blues", mincnt=1, linewidths=0)
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.2, linestyle="--")
    style_axes(ax, "actual arrival delay (min)", "predicted (min)",
               "Predicted vs actual")

    ax = axes[1]
    rlo, rhi = np.percentile(resid, [1, 99])
    ax.hist(resid, bins=60, range=(rlo, rhi), color=SEQ, alpha=0.85)
    ax.axvline(0, color=INK_2, linewidth=1.2, linestyle="--")
    style_axes(ax, "residual: actual - predicted (min)", "flights",
               "Residual distribution")
    ax.text(0.98, 0.94, f"mean {resid.mean():+.2f}\nsd {resid.std():.2f}",
            transform=ax.transAxes, ha="right", va="top", color=INK, fontsize=9)
    fig.tight_layout()
    save(fig, "residuals.png")


def write_table(runs: dict) -> None:
    """Markdown table that REPORT.md includes."""
    lines = []
    for task, metrics in (("regression", REG_METRICS),
                          ("classification", CLF_METRICS)):
        rows = {k: v for k, v in runs.items() if v.get("task") == task}
        if not rows:
            continue
        lines.append(f"\n### {task.capitalize()} arm\n")
        head = ["model", "arm"] + metrics + ["train_s"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        key = "rmse" if task == "regression" else "areaUnderROC"
        order = sorted(rows.items(),
                       key=lambda kv: (kv[1].get(key, float("inf"))
                                       * (1 if task == "regression" else -1)))
        for name, v in order:
            cells = [v.get("model", name), v.get("arm", "")]
            cells += [f"{v[m]:.4f}" if m in v else "-" for m in metrics]
            cells.append(f"{v.get('train_seconds', float('nan')):.0f}")
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")

    out = os.path.join(BENCH_DIR, "results_table.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="flight-delay-mllib")
    ap.add_argument("--skip-residuals", action="store_true")
    ap.add_argument("--from-json", action="store_true",
                    help="use committed full tournament_results.json instead of MLflow")
    args = ap.parse_args()

    os.makedirs(BENCH_DIR, exist_ok=True)
    if args.from_json:
        with open(os.path.join(BENCH_DIR, "tournament_results.json")) as fh:
            runs = json.load(fh)
        print(f"loaded {len(runs)} runs from tournament_results.json")
    else:
        runs = load_runs(args.experiment)
    print(f"\nrendering figures into {BENCH_DIR}")
    plot_explained_variance()
    plot_importances()
    plot_model_comparison(runs)
    if not args.skip_residuals:
        plot_residuals(runs)
    write_table(runs)
    print("\ndone")


if __name__ == "__main__":
    main()
