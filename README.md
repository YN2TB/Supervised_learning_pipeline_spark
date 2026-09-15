# Advanced PySpark MLlib: Flight Delay Prediction

End-to-end supervised learning pipeline on the 2015 US DOT on-time performance
feed: custom PySpark `Estimator`/`Transformer` classes, Arrow-vectorised UDFs,
distributed PCA, a five-algorithm tournament under parallel cross-validation,
MLflow tracking with Model Registry promotion, and streaming inference against
the serialized pipeline.

| | |
|---|---|
| **Data** | 5,819,079 flights (592 MB CSV), 322 airports, 14 carriers |
| **Curated** | 5,704,000 rows (98.0%) after leakage policy and quality filters |
| **Regression** | `ARRIVAL_DELAY` in minutes |
| **Classification** | severe delay, `ARRIVAL_DELAY > 30` (11.08% positive) |
| **Prediction point** | wheels-off, see [Leakage policy](#leakage-policy) |

---

## Quick start

### First time only

Nothing is installed system-wide: the interpreter, the JDK and the Hadoop
natives all live under this directory. [`SETUP.md`](SETUP.md) explains why each
version is pinned; this is just the sequence.

```bash
# 1. Python 3.11 venv + pinned dependencies
uv python install 3.11
"$(uv python find 3.11)" -m venv .venv
./.venv/Scripts/python.exe -m pip install -U pip setuptools
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. Project-local JDK 17  ->  .jdk/
#    Temurin 17 as a zip (no admin/UAC needed), from the Adoptium API.

# 3. Hadoop natives  ->  .hadoop/bin/
#    hadoop.dll + winutils.exe, version 3.3.5, from cdarlint/winutils.
#    Must be 3.3.5 to match PySpark's bundled hadoop-client-*-3.3.4.jar.
```

Then place the three Kaggle CSVs in `data/`: `flights.csv`, `airports.csv`,
`airlines.csv` ([source](https://www.kaggle.com/datasets/usdot/flight-delays)).
They are gitignored for size; `flights.csv` alone is 592 MB.

### Every session

```bash
source ./env.sh                    # Git Bash        (PowerShell: . .\env.ps1)
python scripts/smoke_test.py       # runtime gate: must print 5/5
```

**`source ./env.sh` is not optional, even for a one-line query.** It pins the
JDK, the Hadoop natives and the venv interpreter; without it `python` is the
system 3.14, which cannot import PySpark at all, and which carries MLflow
3.8.1, whose first `MlflowClient` call silently migrates `mlflow.db` to the 3.x
schema and locks the pinned 2.19.0 out of it. See [Environment](#environment).

Then the full sequence:

```bash
python data_prep.py --step raw             # CSV -> Parquet, partitioned by month
python scripts/build_airport_code_map.py   # recover October's airport codes
python data_prep.py --step curate          # joins, labels, leakage policy
python scripts/test_transformers.py        # transformer gate: must print 8/8

./submit_pipeline.sh                       # tournament + MLflow  (long-running)
python benchmark_results.py                # figures + results table

python inference.py &                      # streaming scorer
python scripts/make_stream_events.py --batches 5
```

`submit_pipeline.sh` takes `MODE=train` (default), `MODE=inference` or
`MODE=benchmark`.

Useful flags while iterating:

```bash
# fast end-to-end shakeout on a tiny slice
python mllib_pipeline.py --sample-fraction 0.005 --tune-fraction 0.5 --folds 2

# tune on 10% of train, refit the winner on all of it
python mllib_pipeline.py --tune-fraction 0.1 --folds 5 --parallelism 4

# PCA arms are 6.7-8.0x faster on the driver-side Gramian route; identical
# components, so use it whenever you are not demonstrating the distributed one
python mllib_pipeline.py --svd-mode local-eigs
```

---

## Viewing the results

The tournament is already fitted: `mlflow.db` holds every run and the Model
Registry entry, so none of this needs a retrain.

```bash
source ./env.sh
MLFLOW_URI=$(python -c 'import spark_session; print(spark_session.TRACKING_URI)')
mlflow ui --backend-store-uri "$MLFLOW_URI" --host 127.0.0.1 --port 5000
```

`spark_session.TRACKING_URI` is the one place the store location is defined, so
that form keeps working after the directory is renamed for submission. Note the
URI needs a *Windows* path: `sqlite:///$PWD/mlflow.db` does **not** work in Git
Bash, where `$PWD` is `/d/BigData` and SQLite cannot open it. Use `$(pwd -W)` if
you want to spell it out by hand.

Then open <http://127.0.0.1:5000>. Two things to look at:

- **Experiments -> `flight-delay-mllib`**: the tournament arms as top-level
  runs, each with its cross-validation folds nested underneath.
- **Models -> `flight_delay_pipeline`**: version 1 in stage **Production**,
  which is the `Staging -> Production` transition the brief asks for.

Static copies of the same numbers live in `docs/benchmarks/` (figures plus
`results_table.md`), regenerated from the store by `python benchmark_results.py`
without refitting anything.

> **Two traps around the store.**
> Run `mlflow` only through `env.sh` or `./.venv/Scripts/mlflow.exe`: a bare
> `mlflow`/`python` resolves to system Python 3.14 and its MLflow 3.8.1 will
> migrate the database out from under the pinned 2.19.0. (Recovery, if it
> happens: the 3.x migrations are additive, so
> `UPDATE alembic_version SET version_num='0584bdc529eb'` restores it with no
> data loss.)
> And **any** training run registers a new model version and promotes it with
> `archive_existing_versions=True`, even `--sample-fraction 0.01`, and
> `--experiment` does not protect you, because the registry is global. Back up
> `mlflow.db`, `models/` and `docs/benchmarks/tournament_results.json` before a
> run you do not intend to keep.

---

## Layout

```
custom_transformers.py     Estimator/Model pairs + Arrow pandas_udfs
mllib_pipeline.py          ingest -> pipeline -> CV tournament -> MLflow
inference.py               readStream + PipelineModel.load
benchmark_results.py       metric extraction + figures
data_prep.py               CSV -> Parquet, code repair, joins, labels
flight_schema.py           read schema + the leakage policy
spark_session.py           one place the SparkSession is configured
submit_pipeline.sh         spark-submit entry point (MODE=train|inference|benchmark)
env.sh / env.ps1           JAVA_HOME, HADOOP_HOME, PYSPARK_PYTHON, ...
scripts/
  smoke_test.py            Phase 0 gate: JVM, Arrow, model save/load
  build_airport_code_map.py  October DOT-id -> IATA recovery
  test_transformers.py     Phase 2 gate: leak-safety + serialization
  make_stream_events.py    drip held-out rows into stream_input/
docs/
  REPORT.md                derivations, design decisions, benchmark analysis
  benchmarks/              generated figures and results table
```

---

## Design decisions worth knowing

### Leakage policy

The prediction point is fixed at **wheels-off**: departure delay and taxi-out are
known, and everything observable only after that instant is dropped. The policy
lives in `flight_schema.py: LEAKY_COLUMNS` and is enforced by an assertion in
`data_prep.py`: it is executable, not just documented.

The trap is the five delay-attribution columns (`AIR_SYSTEM_DELAY`,
`SECURITY_DELAY`, `AIRLINE_DELAY`, `LATE_AIRCRAFT_DELAY`, `WEATHER_DELAY`). By the
USDOT definition they **sum to `ARRIVAL_DELAY`**, so a model given them
reconstructs the target by addition and scores R² ≈ 0.99 while being worthless in
production. `ARRIVAL_DELAY` itself is also dropped after the labels are derived,
so no unlabelled copy of the target is left lying in the table.

**Sanity check:** at this horizon expect **R² ≈ 0.85–0.92**. A result near 0.99
means a leak got back in.

### The October schema drift

One month of the feed ships a different key encoding: October rows carry numeric
DOT airport ids (`14747`) where every other month carries IATA codes (`SEA`).
Since `airports.csv` is keyed by IATA only, a naive join silently drops all
**486,165** October rows.

`scripts/build_airport_code_map.py` recovers the mapping from the data itself
using two signals, because neither suffices alone:

1. **Direction-aware flight-number vote**: a flight number flies the same route
   all year. Restricted to keys whose direction is unambiguous, since some
   carriers reuse a number for both legs of a round trip.
2. **Geometric fit**: every row carries its route `DISTANCE`, so an unknown code
   is located by trilateration against already-known partners.

They are complementary: swapping a route's endpoints leaves the great-circle
distance unchanged, so signal 2 is blind to a transposition, exactly what signal
1 guards. Signal 1 in turn is unreliable in the thin tail, where signal 2 decides.

Verification is **per-code, not pooled**: a mis-mapped airport is wrong on every
row it appears in, so it shows as a large median for that code while barely moving
the global median. Pooled stats looked excellent (0.96 mi) while five codes were
badly wrong.

**Result:** 302/307 codes resolved, injective, worst per-code error 3.25 mi,
**99.8% of October rows recovered**. The last five had no confident fit and are
left unmapped rather than guessed: 885 rows, 0.015% of the dataset.

### `withMean=False` is deliberate

The assembled vector is 23 non-zero of 80 slots and therefore sparse. Centering
maps every structural zero to `-μ`, destroying sparsity and forcing a dense
materialisation: **2.0×** the memory here, measured with Spark's `SizeEstimator`
(336 vs 672 bytes per row), and **110×** if `ROUTE` were one-hot encoded instead of
target-encoded. `withStd` is multiplicative, so zero is a fixed point and sparsity
survives. Full argument in [`docs/REPORT.md` §A1.2](docs/REPORT.md).

### Custom stages are Estimator/Model pairs

`OutlierIQRTruncator` computes its Tukey fences in `_fit` on the training split
and freezes them into the model. Implemented as a bare `Transformer` (which the
brief's wording suggests) it would recompute quantiles from whatever DataFrame it
received, refitting itself on test data and behaving differently on every
streaming micro-batch. Clipping fences are learned parameters.

This still meets the requirement literally: `pyspark.ml.Model` *is* a subclass of
`pyspark.ml.Transformer`, so `OutlierIQRTruncatorModel` (the object that does the
clipping inside the fitted pipeline and ships to streaming) **is** a
`Transformer` subclass. `HaversineTransformer` and `SignedLog1pTransformer`
subclass `Transformer` directly.

### PCA runs in two arms

PCA is required, but it rotates features into linear combinations, so
`featureImportances` over principal components says nothing about the original
variables. Every model trains both with `PCA(k=10)` and without; the no-PCA arm
supplies interpretable importances, and the comparison measures what the
compression costs.

The PCA stage is `custom_transformers.RowMatrixPCA`, not `spark.ml`'s `PCA`. The
brief asks for PCA computed "using RowMatrix SVD without collecting full
covariance matrices to the Driver node", and `spark.ml`'s `PCA` forms the whole
77×77 covariance on the driver. `RowMatrixPCA` calls `computeSVD` with the mode
named explicitly, so `--svd-mode dist-eigs` runs ARPACK Lanczos on distributed
`Aᵀ(Av)` products and never materialises an n×n matrix. It centres the rows
first, so both routes match `spark.ml`'s PCA exactly.

`dist-eigs` is the only mode that satisfies that wording: `local-eigs` and
`local-svd` both call `computeGramianMatrix` and so form the covariance on the
driver. It is not the default, because on this data it does not finish. Every PCA
arm fails under it with `ARPACK info = 3, No shifts could be applied` (one with an
`ArrayIndexOutOfBounds` at 1540 = 77 × 20 = n × ncv, the Lanczos basis
overflowing), recorded in `docs/benchmarks/tournament_failures.json`. The spectrum
is flat where we cut it, ARPACK restarts by shifting along spectral gaps, and
Spark hard-codes `ncv = min(2k, n) = 20`.

So **`local-eigs` is the default**, and it is what the registered model was fitted
with; `svd_mode` is logged on every PCA run. Components are identical either way
(|cos| = 1.0). Where `dist-eigs` does run it costs 6.7–8.0× on the PCA stage,
since it makes one distributed pass per Lanczos iteration where the Gramian route
makes one pass total.

The distributed route is still the right design, which is why the fallback is
reported as a finding rather than quietly swapped. The 77×77 covariance is only 46 KB
because `ROUTE` is target-encoded; one-hot it instead and the vector is 4,703 wide
with a 169 MB covariance, and with `TAIL_NUMBER` too, 9,599 wide and 703 MB. The
distributed route's driver footprint is ARPACK's Lanczos basis, `O(n x ncv)` with
`ncv = 20`, so it grows linearly rather than quadratically: 19 KB, 886 KB and
1.76 MB across those same three widths. That is the property worth having in a
pipeline meant to scale. Components are identical
either way, so pass `--svd-mode local-eigs` while iterating; see
[`docs/REPORT.md` §A2.1](docs/REPORT.md).

---

## Performance notes

The first working pipeline took **146 s to fit** on 114k rows while its stages
cost ~4 s in isolation. `Pipeline.fit` fits each stage against the *lazy* output
of the previous ones and never caches between them, so every fitted stage
re-executes the whole upstream chain, and iterative learners re-execute it once
per iteration. Two measured fixes:

- **Target encoding via a cached `create_map` expression instead of a broadcast
  join.** Execution went ~60× cheaper (0.10 s vs 6 s/pass); the 6.3 s of
  driver-side plan construction is paid once, since Column expressions are
  independent of any DataFrame.
- **`MaterializeCache`: an Estimator that persists in `_fit`, whose Model is a
  no-op.** Caching helps fitting (51 s → 35 s) but *costs* 23× on transform
  (0.28 s → 6.5 s), which every CV fold pays. Splitting it captures both, and the
  saved model carries no caching into scoring or streaming.

| | fit | transform |
|---|---|---|
| original | 146 s | 19.7 s |
| optimised | **24.6 s** | **0.32 s** |

Verified semantically neutral: the transformer gate reports byte-identical
column totals before and after.

---

## Environment

The machine's defaults cannot run Spark; everything is pinned project-locally
under this directory and nothing system-wide is modified. See
[`SETUP.md`](SETUP.md) for the full reasoning.

| Component | Choice | Why |
|---|---|---|
| Python | **3.11** (venv) | 3.14 breaks PySpark at import; on **3.12.0** the JVM launches the Python worker, it exits 0 without writing to its socket, and every task dies with `EOFException` |
| Java | **Temurin JDK 17** (`.jdk/`) | `JAVA_HOME` pointed at JDK 26, unsupported by every Spark release |
| PySpark | 3.5.4 | stable pairing with MLflow 2.x |
| MLflow | **2.19.0** | 3.x removes `transition_model_version_stage`, the Staging→Production API the brief requires |
| Tracking store | **SQLite**, not `file:` | the Model Registry is not supported by the filesystem store at all |
| Hadoop natives | 3.3.5 (`.hadoop/`) | `PipelineModel.save()` on Windows needs `hadoop.dll`; must match Spark's bundled Hadoop **3.3.4** jars, not the 3.4.1 tree in `C:\Hadoop` |

---

## Verification gates

Each phase has a gate; nothing downstream is trusted until it passes.

| Gate | Command | Expected |
|---|---|---|
| Runtime | `python scripts/smoke_test.py` | `ALL CHECKS PASSED (5/5)` |
| Data | `python data_prep.py --step curate` | 5,704,000 rows; 0 missing coordinates |
| Transformers | `python scripts/test_transformers.py` | `ALL CHECKS PASSED (8/8)` |
| Tournament | `mlflow ui --backend-store-uri "$(python -c 'import spark_session; print(spark_session.TRACKING_URI)')"` | 14 tournament arms as top-level runs, CV folds nested under each, a version in **Production** |
| Streaming | `inference.py` + `make_stream_events.py` | predictions on the console sink and in `stream_output/` |

---

## Source

Assignment brief: `supervised_learning_pipeline_spark_subject.docx.pdf`.
Dataset: [Kaggle: US Flight Delays and Performance Data](https://www.kaggle.com/datasets/usdot/flight-delays)
(place `flights.csv`, `airports.csv`, `airlines.csv` in `data/`; they are
gitignored for size).
