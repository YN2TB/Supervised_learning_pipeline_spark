"""Phase 5: pull the tournament results out of MLflow and render the figures.

    source ./env.sh && python benchmark_results.py                      # pre-departure
    source ./env.sh && python benchmark_results.py --experiment flight-delay-mllib

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
from spark_session import TRACKING_URI, path, results_dir  # noqa: E402

BENCH_DIR = path("docs", "benchmarks")
DEFAULT_EXPERIMENT = "flight-delay-predeparture"
REFERENCE_EXPERIMENT = "flight-delay-mllib"     # the wheels-off tournaments


def bench_dir_for(experiment: str) -> str:
    """Same rule as mllib_pipeline --out-dir (spark_session.results_dir)."""
    return results_dir(experiment)

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
        # Ascending start_time so a re-run overwrites the older row, and
        # FINISHED only. An arm interrupted between log_metrics and log_model
        # leaves a run that HAS metrics but never completed; without both
        # guards it can win the dict slot and put a stale number into the
        # report - a plausible-looking number, with nothing marking it wrong.
        for r in client.search_runs([exp.experiment_id], max_results=1000,
                                    order_by=["attributes.start_time ASC"]):
            name = r.data.tags.get("mlflow.runName", "")
            if "__" not in name or not r.data.metrics:
                continue          # skip autolog's nested CV children
            if r.info.status != "FINISHED":
                continue
            sf = r.data.params.get("sample_fraction")
            rows[name] = dict(r.data.metrics,
                              model=r.data.params.get("model", name.split("__")[0]),
                              arm=r.data.params.get("arm", name.split("__")[1]),
                              task=r.data.params.get("task", ""),
                              sample_fraction=float(sf) if sf is not None else None,
                              test_rows=int(r.data.params["test_rows"])
                              if r.data.params.get("test_rows") else None)
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
        # No registered model in this experiment yet: use a regression arm's own
        # file (every PCA arm writes one).
        cands = sorted(f for f in os.listdir(BENCH_DIR)
                       if f.startswith("pca_explained_variance__") and "regressor" in f)
        if not cands:
            print("  (no PCA variance file; skipping)")
            return
        src = os.path.join(BENCH_DIR, cands[0])
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
    ax.set_xticks(ks if len(ks) <= 20 else ks[::max(1, len(ks) // 12)])
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
        vals = np.abs(np.asarray(payload["importances"], dtype=float))
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
        style_axes(ax, payload.get("kind", "Gini importance"), "",
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
                if v.get("task") == task and metric in v
                and not str(v.get("model", "")).startswith("baseline")}
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
                    label="with PCA" if arm == "pca" else "no PCA")
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
        from mllib_pipeline import CURATED, REG_LABEL, make_split
        from spark_session import build_spark
        # cores="*" is NOT cosmetic. randomSplit draws per input partition, and
        # a parquet read produces one partition per core, so the same seed
        # yields a DIFFERENT split at a different core count: 16 cores gives
        # train=4,564,168 here, 8 gives 4,563,188, 4 gives 4,562,867. This ran
        # at cores="8" while the tournament ran at "*", so the "held-out" rows
        # plotted below were partly training rows. Match the training session.
        spark = build_spark("benchmark-residuals", cores="*")
        try:
            df = spark.read.parquet(CURATED)
            frac = float(next((r.get("sample_fraction", 1.0)
                               for r in runs.values()
                               if r.get("sample_fraction") is not None), 1.0))
            _, test = make_split(df, frac)

            # Matching core counts is necessary but not sufficient - a different
            # machine has a different core count. So verify against the row
            # count the tournament actually recorded, and refuse to draw a
            # residual plot over the wrong rows rather than drawing a wrong one.
            expected = next((r.get("test_rows") for r in runs.values()
                             if r.get("test_rows")), None)
            if expected is not None:
                actual = test.count()
                if int(actual) != int(expected):
                    print(f"  (residual plot skipped: reconstructed test set has "
                          f"{actual:,} rows but the tournament used {expected:,}. "
                          f"The split depends on read partitioning; rerun this on "
                          f"the machine that trained, or retrain.)")
                    return
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


def plot_importance_grid() -> None:
    """Top ten features of every no-PCA model, side by side, one scale each."""
    files = sorted(f for f in os.listdir(BENCH_DIR)
                   if f.startswith("importance_") and f.endswith("__nopca.json"))
    if not files:
        return
    cols = 3 if len(files) > 4 else 2
    rows_ = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(5.2 * cols, 3.7 * rows_), squeeze=False)
    for ax in axes.flat[len(files):]:
        ax.axis("off")
    for ax, fname in zip(axes.flat, files):
        with open(os.path.join(BENCH_DIR, fname)) as fh:
            payload = json.load(fh)
        vals = np.abs(np.asarray(payload["importances"], dtype=float))
        share = vals / (vals.sum() or 1.0)
        idx = np.argsort(share)[::-1][:10][::-1]
        ax.barh(range(len(idx)), share[idx], color=SEQ, height=0.62)
        ax.set_yticks(range(len(idx)))
        ax.set_yticklabels([payload["features"][i] for i in idx], fontsize=8)
        model = fname[len("importance_"):-len("__nopca.json")].replace("_", " ")
        style_axes(ax, "share of total |importance|", "", model)
        ax.title.set_fontsize(10)
    fig.suptitle("What each model uses (no-PCA arms)", x=0.01, ha="left",
                 color=INK, fontsize=13)
    fig.tight_layout()
    save(fig, "feature_importance_grid.png")


def plot_curves_combined(runs: dict) -> None:
    """Every classifier's ROC and PR curve on one pair of axes."""
    curves = {}
    for name, v in runs.items():
        if v.get("task") != "classification" or str(v.get("model", "")).startswith("baseline"):
            continue
        f = os.path.join(BENCH_DIR, f"{name}_curves.json")
        if os.path.exists(f):
            with open(f) as fh:
                curves[name] = (json.load(fh), v)
    if not curves:
        print("  (no classifier curves; skipping)")
        return
    models = sorted({v["model"] for _, v in curves.values()})
    palette = ["#2a78d6", "#eb6834", "#1f9e89", "#8a5cc2"]
    colour = {m: palette[i % len(palette)] for i, m in enumerate(models)}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    base = None
    for name, (c, v) in sorted(curves.items()):
        ls = "--" if v.get("arm") == "pca" else "-"
        lab = f"{v['model'].replace('_', ' ')} ({'PCA' if v.get('arm') == 'pca' else 'no PCA'})"
        axes[0].plot(c["fpr"], c["tpr"], color=colour[v["model"]], linestyle=ls, linewidth=1.8,
                     label=f"{lab}, {v.get('areaUnderROC', float('nan')):.3f}")
        axes[1].plot(c["recall"], c["precision"], color=colour[v["model"]], linestyle=ls,
                     linewidth=1.8, label=f"{lab}, {v.get('areaUnderPR', float('nan')):.3f}")
        base = c.get("positive_rate", base)
    axes[0].plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle=":", label="chance, 0.500")
    if base is not None:
        axes[1].axhline(base, color=INK_2, linewidth=1, linestyle=":",
                        label=f"chance, {base:.3f}")
    style_axes(axes[0], "false positive rate", "true positive rate", "ROC, test split")
    style_axes(axes[1], "recall", "precision", "Precision-recall, test split")
    for ax, loc in ((axes[0], "lower right"), (axes[1], "upper right")):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.01)
        leg = ax.legend(frameon=False, fontsize=8, loc=loc, title="model, area under curve",
                        title_fontsize=8)
        for t in leg.get_texts():
            t.set_color(INK_2)
    fig.tight_layout()
    save(fig, "roc_pr_all_classifiers.png")


def write_comparison(runs: dict, reference: str, current: str) -> None:
    """Per model and arm: the reference experiment against this one."""
    try:
        ref = load_runs(reference)
    except Exception as exc:  # noqa: BLE001
        print(f"  (comparison skipped: {exc})")
        return
    lines = [f"Reference `{reference}` against `{current}`, test split, same seed.\n"]
    for task, key, others in (("regression", "rmse", ["mae", "r2"]),
                              ("classification", "areaUnderROC", ["areaUnderPR", "f1"])):
        names = sorted({n for n, v in list(runs.items()) + list(ref.items())
                        if v.get("task") == task})
        if not names:
            continue
        mets = [key] + others
        lines.append(f"\n### {task.capitalize()}\n")
        head = ["model", "arm"] + [f"{m} ({lab})" for m in mets for lab in ("ref", "now")]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "|".join(["---"] * len(head)) + "|")
        for n in names:
            a, b = ref.get(n, {}), runs.get(n, {})
            v = a or b
            cells = [v.get("model", n), v.get("arm", "")]
            for m in mets:
                for src in (a, b):
                    cells.append(f"{src[m]:.4f}" if m in src else "-")
            lines.append("| " + " | ".join(str(c) for c in cells) + " |")
    out = os.path.join(BENCH_DIR, "comparison_table.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote {out}")


def main() -> None:
    global BENCH_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    ap.add_argument("--compare", default=REFERENCE_EXPERIMENT,
                    help="experiment to set side by side with this one ('' to skip)")
    ap.add_argument("--skip-residuals", action="store_true")
    args = ap.parse_args()

    BENCH_DIR = bench_dir_for(args.experiment)
    os.makedirs(BENCH_DIR, exist_ok=True)
    runs = load_runs(args.experiment)
    print(f"\nrendering figures into {BENCH_DIR}")
    plot_explained_variance()
    plot_importances()
    plot_importance_grid()
    plot_model_comparison(runs)
    plot_curves_combined(runs)
    # The residual plot reloads models/best_pipeline, i.e. whatever is in
    # Production; only meaningful for the experiment that registered it. Every
    # arm of the newer experiments already has its own residual plot.
    if not args.skip_residuals and args.experiment == REFERENCE_EXPERIMENT:
        plot_residuals(runs)
    write_table(runs)
    if args.compare and args.compare != args.experiment:
        write_comparison(runs, args.compare, args.experiment)
    print("\ndone")


if __name__ == "__main__":
    main()
