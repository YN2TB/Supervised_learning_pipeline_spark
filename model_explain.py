"""Per-arm explanation artifacts: what each model uses and how well it separates.

Called from the tournament loop for every finished arm. Writes PNG and JSON
files into the run's output directory and logs them to the active MLflow run
under ``plots/``, so every run in the UI carries its own charts.

    importance   every model, not only trees. Trees report featureImportances;
                 linear models (LinearRegression, LinearSVC, GLM) report the
                 coefficient on each scaled feature, which is comparable across
                 features because StandardScaler divided each by its sigma.
                 PCA arms are mapped back to the original features.
    roc / pr     classification arms: ROC and precision-recall curves on the
                 test split, plus the confusion matrix at the model's own
                 threshold. PR matters more here: only 11% of flights are
                 severely late, and ROC flatters a model on imbalanced data.
    residuals    regression arms: predicted against actual, and the residual
                 distribution. ROC has no meaning without a binary output.

Nothing here can fail a run: every public function catches its own errors.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# House style, shared with benchmark_results.py.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e3e2de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

# Test rows pulled to the driver for the curves. A curve over 200k points is
# indistinguishable from one over 1.1M, and the AUC in the legend is the exact
# one Spark's evaluator logged, not recomputed from the sample.
CURVE_SAMPLE_ROWS = 200_000


def _style(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)


def _save(fig, out_dir, name, logged):
    import mlflow
    p = os.path.join(out_dir, name)
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    mlflow.log_artifact(p, artifact_path="plots")
    logged.append(p)


def _dump(obj, out_dir, name, logged):
    import mlflow
    p = os.path.join(out_dir, name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh)
    mlflow.log_artifact(p, artifact_path="plots")
    logged.append(p)


# ---------------------------------------------------------------------------
# Importance
# ---------------------------------------------------------------------------
def _names(fitted_model, sample_df, col: str, selector_out: str) -> list[str]:
    """Slot names of assembled vector `col`, after the selector writing `selector_out`."""
    out = fitted_model.transform(sample_df.limit(5))
    meta = out.schema[col].metadata.get("ml_attr", {})
    names = [""] * int(meta.get("num_attrs", 0))
    for group in meta.get("attrs", {}).values():
        for attr in group:
            names[attr["idx"]] = attr.get("name", f"f{attr['idx']}")
    selector = next((s for s in fitted_model.stages if hasattr(s, "selectedFeatures")
                     and s.getOutputCol() == selector_out), None)
    return [names[i] for i in selector.selectedFeatures] if selector else names


def slot_names(fitted_model, sample_df) -> list[str]:
    """Names of the slots in features_selected."""
    return _names(fitted_model, sample_df, "features_raw", "features_selected")


def importance(fitted_model, sample_df) -> dict:
    """Per-feature importance for any of the tournament's estimators.

    Returns {"kind", "features", "importances", "signed"}, over the
    original (pre-PCA) features. For a PCA arm the estimator only sees the
    components, so its weights are carried back through the loadings:
    for a linear model w_x = V w_pc exactly (the score is w_pc . V^T x); for a
    tree there is no exact inverse, so each component's importance is spread
    over the features by its squared loadings (each column of V has unit norm,
    so this conserves the total).
    """
    est = fitted_model.stages[-1]
    pca = next((s for s in fitted_model.stages if hasattr(s, "explainedVariance")), None)
    names = slot_names(fitted_model, sample_df)

    if hasattr(est, "featureImportances"):
        kind, signed = "tree importance (Gini gain share)", False
        w = est.featureImportances.toArray()
    elif hasattr(est, "coefficients"):
        kind, signed = "coefficient on the scaled feature", True
        w = est.coefficients.toArray()
    else:
        raise ValueError(f"{type(est).__name__} has neither importances nor coefficients")

    if pca is not None and pca.getInputCol() == "features_selected":
        V = pca.pc.toArray()                       # n_features x k
        if signed:
            w = V @ w
            kind += ", carried back through the PCA loadings (exact)"
        else:
            w = (V ** 2) @ w
            kind += ", spread over features by squared PCA loadings (approximate)"
    elif pca is not None:
        # Numeric-only PCA (--pca-scope numeric): the vector is [PCs of the numeric
        # block, one-hot slots]. Carry the PC part back onto the numeric columns and
        # keep the one-hot part as it is. The PCs were rescaled once more after the
        # rotation, so for linear models this is approximate as well.
        num_names = _names(fitted_model, sample_df, "num_raw",
                           selector_out=pca.getInputCol())
        pc_idx = [i for i, n in enumerate(names) if n.startswith(pca.getOutputCol())]
        rest = [i for i in range(len(names)) if i not in set(pc_idx)]
        V = pca.pc.toArray()                       # numeric slots x k
        w_pc = np.asarray(w)[pc_idx]
        mapped = (V @ w_pc) if signed else ((V ** 2) @ w_pc)
        names = list(num_names) + [names[i] for i in rest]
        w = np.concatenate([mapped, np.asarray(w)[rest]])
        kind += ", numeric block carried back through its PCA loadings (approximate)"
    if len(names) != len(w):
        names = [f"f{i}" for i in range(len(w))]
    return {"kind": kind, "features": names, "importances": [float(x) for x in w],
            "signed": signed}


def plot_importance(payload: dict, title: str, out_dir: str, stem: str, logged: list,
                    top: int = 20):
    vals = np.asarray(payload["importances"], dtype=float)
    names = payload["features"]
    n = min(top, len(vals))
    idx = np.argsort(np.abs(vals))[::-1][:n][::-1]

    fig, ax = plt.subplots(figsize=(7.8, 0.34 * n + 1.7))
    colours = [ORANGE if (payload["signed"] and vals[i] < 0) else BLUE for i in idx]
    ax.barh(range(n), np.abs(vals[idx]), color=colours, height=0.62)
    ax.set_yticks(range(n))
    ax.set_yticklabels([names[i] for i in idx], fontsize=8.5, color=INK)
    span = float(np.abs(vals[idx]).max()) or 1.0
    for y, i in enumerate(idx):
        ax.text(abs(vals[i]) + span * 0.012, y, f"{vals[i]:+.3g}" if payload["signed"]
                else f"{vals[i]:.3g}", va="center", color=INK_2, fontsize=7.5)
    ax.set_xlim(0, span * 1.18)
    _style(ax, payload["kind"], "", title)
    if payload["signed"]:
        ax.set_title("blue raises the prediction, orange lowers it", loc="right",
                     color=INK_2, fontsize=8, pad=10)
    _save(fig, out_dir, f"{stem}_importance.png", logged)


# ---------------------------------------------------------------------------
# Classification curves
# ---------------------------------------------------------------------------
def _scores(preds, label: str, n_test: int):
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    col = "probability" if "probability" in preds.columns else "rawPrediction"
    frac = min(1.0, CURVE_SAMPLE_ROWS / max(n_test, 1))
    sample = preds if frac >= 1.0 else preds.sample(False, frac, seed=11)
    pdf = (sample.select(F.col(label).cast("double").alias("y"),
                         vector_to_array(col)[1].alias("s"),
                         F.col("prediction").cast("double").alias("p"))
           .toPandas())
    return pdf["y"].to_numpy(), pdf["s"].to_numpy(), pdf["p"].to_numpy(), col


def classification_plots(preds, label: str, n_test: int, metrics: dict, title: str,
                         out_dir: str, stem: str, logged: list) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
    y, s, p, score_col = _scores(preds, label, n_test)
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    base = float(y.mean())

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax = axes[0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2,
            label=f"model, AUC {metrics.get('areaUnderROC', float('nan')):.3f}")
    ax.plot([0, 1], [0, 1], color=INK_2, linewidth=1, linestyle="--", label="chance, AUC 0.500")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _style(ax, "false positive rate", "true positive rate", "ROC curve")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    ax = axes[1]
    ax.plot(rec, prec, color=BLUE, linewidth=2,
            label=f"model, AUC-PR {metrics.get('areaUnderPR', float('nan')):.3f}")
    ax.axhline(base, color=INK_2, linewidth=1, linestyle="--",
               label=f"chance = positive rate {base:.3f}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    _style(ax, "recall", "precision", "Precision-recall curve")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f"{stem}_roc_pr.png", logged)

    cm = confusion_matrix(y, p, labels=[0.0, 1.0])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}\n{v / cm.sum():.1%}", ha="center", va="center",
                color=SURFACE if v > cm.max() / 2 else INK, fontsize=9)
    ax.set_xticks([0, 1], ["predicted on time", "predicted > 30 min"], fontsize=8.5)
    ax.set_yticks([0, 1], ["on time", "> 30 min late"], fontsize=8.5)
    ax.tick_params(length=0, colors=INK_2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(f"Confusion matrix, test sample of {len(y):,}", color=INK, fontsize=10,
                 loc="left")
    fig.set_facecolor(SURFACE)
    _save(fig, out_dir, f"{stem}_confusion.png", logged)

    # A thinned curve for the combined chart in benchmark_results.py.
    keep = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
    keep_pr = np.unique(np.linspace(0, len(rec) - 1, 300).astype(int))
    curve = {"score_col": score_col, "rows": int(len(y)), "positive_rate": base,
             "fpr": fpr[keep].round(5).tolist(), "tpr": tpr[keep].round(5).tolist(),
             "recall": rec[keep_pr].round(5).tolist(),
             "precision": prec[keep_pr].round(5).tolist(),
             "confusion": cm.tolist()}
    _dump(curve, out_dir, f"{stem}_curves.json", logged)
    return curve


# ---------------------------------------------------------------------------
# Regression plots
# ---------------------------------------------------------------------------
def regression_plots(preds, label: str, n_test: int, title: str,
                     out_dir: str, stem: str, logged: list):
    from pyspark.sql import functions as F
    frac = min(1.0, CURVE_SAMPLE_ROWS / max(n_test, 1))
    sample = preds if frac >= 1.0 else preds.sample(False, frac, seed=11)
    pdf = sample.select(F.col(label).cast("double").alias("y"),
                        F.col("prediction").cast("double").alias("p")).toPandas()
    y, p = pdf["y"].to_numpy(), pdf["p"].to_numpy()
    r = y - p

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    lo, hi = np.percentile(y, [0.5, 99.5])
    ax = axes[0]
    ax.hexbin(y, p, gridsize=45, extent=(lo, hi, lo, hi), cmap="Blues", mincnt=1,
              linewidths=0, bins="log")
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=1.1, linestyle="--")
    _style(ax, "actual (min)", "predicted (min)", "Predicted vs actual")
    ax = axes[1]
    rlo, rhi = np.percentile(r, [1, 99])
    ax.hist(r, bins=60, range=(rlo, rhi), color=BLUE, alpha=0.85)
    ax.axvline(0, color=INK_2, linewidth=1.1, linestyle="--")
    _style(ax, "residual: actual - predicted (min)", "flights", "Residuals")
    ax.text(0.98, 0.94, f"mean {r.mean():+.2f}\nsd {r.std():.2f}", transform=ax.transAxes,
            ha="right", va="top", color=INK, fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, f"{stem}_residuals.png", logged)


# ---------------------------------------------------------------------------
def explain_arm(final, train, preds, *, task: str, label: str, n_test: int,
                metrics: dict, run_name: str, out_dir: str) -> list[str]:
    """Everything for one finished arm. Returns the files written."""
    logged: list[str] = []
    os.makedirs(out_dir, exist_ok=True)
    model, _, arm = run_name.partition("__")
    pretty = f"{model.replace('_', ' ')} ({'PCA' if arm == 'pca' else 'no PCA'})"
    try:
        payload = importance(final, train)
        payload["arm"] = run_name.split("__")[-1]
        _dump(payload, out_dir, f"importance_{run_name}.json", logged)
        plot_importance(payload, f"Feature importance, {pretty}", out_dir, run_name, logged)
    except Exception as exc:  # noqa: BLE001
        print(f"  (importance skipped: {type(exc).__name__}: {exc})")
    try:
        if task == "classification":
            classification_plots(preds, label, n_test, metrics, pretty, out_dir,
                                 run_name, logged)
        else:
            regression_plots(preds, label, n_test, pretty, out_dir, run_name, logged)
    except Exception as exc:  # noqa: BLE001
        print(f"  (curves skipped: {type(exc).__name__}: {exc})")
    return logged
