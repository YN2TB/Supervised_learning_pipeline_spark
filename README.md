# Advanced PySpark MLlib — Flight Delay Prediction

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
| **Prediction point** | wheels-off — see [Leakage policy](#leakage-policy) |

---

## Quick start

```bash
source ./env.sh                    # Git Bash        (PowerShell: . .\env.ps1)
python scripts/smoke_test.py       # runtime gate — must print 5/5
```

Then the full sequence:

```bash
python data_prep.py --step raw             # CSV -> Parquet, partitioned by month
python scripts/build_airport_code_map.py   # recover October's airport codes
python data_prep.py --step curate          # joins, labels, leakage policy
python scripts/test_transformers.py        # transformer gate — must print 8/8

./submit_pipeline.sh                       # tournament + MLflow  (long-running)
python benchmark_results.py                # figures + results table

python inference.py &                      # streaming scorer
python scripts/make_stream_events.py --batches 5
```

Useful flags while iterating:

```bash
# fast end-to-end shakeout on a tiny slice
python mllib_pipeline.py --sample-fraction 0.005 --tune-fraction 0.5 --folds 2

# tune on 10% of train, refit the winner on all of it
python mllib_pipeline.py --tune-fraction 0.1 --folds 5 --parallelism 4
```

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
`data_prep.py` — it is executable, not just documented.

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

1. **Direction-aware flight-number vote** — a flight number flies the same route
   all year. Restricted to keys whose direction is unambiguous, since some
   carriers reuse a number for both legs of a round trip.
2. **Geometric fit** — every row carries its route `DISTANCE`, so an unknown code
   is located by trilateration against already-known partners.

They are complementary: swapping a route's endpoints leaves the great-circle
distance unchanged, so signal 2 is blind to a transposition — exactly what signal
1 guards. Signal 1 in turn is unreliable in the thin tail, where signal 2 decides.

Verification is **per-code, not pooled**: a mis-mapped airport is wrong on every
row it appears in, so it shows as a large median for that code while barely moving
the global median. Pooled stats looked excellent (0.96 mi) while five codes were
badly wrong.

**Result:** 302/307 codes resolved, injective, worst per-code error 3.25 mi,
**99.8% of October rows recovered**. The last five had no confident fit and are
left unmapped rather than guessed — 885 rows, 0.015% of the dataset.

### `withMean=False` is deliberate

The assembled vector is ~70% one-hot and therefore sparse. Centering maps every
structural zero to `-μ`, destroying sparsity and forcing a dense materialisation
(~3.3× memory here, ~100× if `ROUTE` were one-hot encoded). `withStd` is
multiplicative, so zero is a fixed point and sparsity survives. Full argument in
[`docs/REPORT.md` §A1.2](docs/REPORT.md).

### Custom stages are Estimator/Model pairs

`OutlierIQRTruncator` computes its Tukey fences in `_fit` on the training split
and freezes them into the model. Implemented as a bare `Transformer` — which the
brief's wording suggests — it would recompute quantiles from whatever DataFrame it
received, refitting itself on test data and behaving differently on every
streaming micro-batch. Clipping fences are learned parameters.

### PCA runs in two arms

PCA is required, but it rotates features into linear combinations, so
`featureImportances` over principal components says nothing about the original
variables. Every model trains both with `PCA(k=10)` and without; the no-PCA arm
supplies interpretable importances, and the comparison measures what the
compression costs.

---

## Performance notes

The first working pipeline took **146 s to fit** on 114k rows while its stages
cost ~4 s in isolation. `Pipeline.fit` fits each stage against the *lazy* output
of the previous ones and never caches between them, so every fitted stage
re-executes the whole upstream chain — and iterative learners re-execute it once
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

Verified semantically neutral — the transformer gate reports byte-identical
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
| Tournament | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | 5 parent runs, nested CV children, a version in **Production** |
| Streaming | `inference.py` + `make_stream_events.py` | predictions on the console sink and in `stream_output/` |

---

## Source

Assignment brief: `supervised_learning_pipeline_spark_subject.docx.pdf`.
Dataset: [Kaggle — US Flight Delays and Performance Data](https://www.kaggle.com/datasets/usdot/flight-delays)
(place `flights.csv`, `airports.csv`, `airlines.csv` in `data/`; they are
gitignored for size).
