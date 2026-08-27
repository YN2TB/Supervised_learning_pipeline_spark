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

## Final benchmark

The committed benchmark artifacts come from the full curated dataset and the
same deterministic 80/20 split (`seed=42`). Each model is evaluated both with
`PCA(k=10)` and without PCA.

| Task | Winning model | Main metric | Other metrics |
|---|---|---:|---:|
| Regression | ElasticNet Linear Regression, no PCA | **RMSE 10.5171** | MAE 7.1939, R² 0.9289 |
| Classification | GBT Classifier, no PCA | **AUC-ROC 0.9817** | AUC-PR 0.9384, F1 0.9716 |

PCA retains about **28.5%** of variance at `k=10`; the paired experiment shows
why satisfying the PCA requirement does not mean deploying the PCA arm. See
[`docs/benchmarks/results_table.md`](docs/benchmarks/results_table.md).

---

## Hướng dẫn chạy đầy đủ

Tất cả lệnh dưới đây phải được chạy tại thư mục gốc của repository. Kiểm tra bằng
PowerShell:

```powershell
Get-Location
# ...\Supervised_learning_pipeline_spark
```

### 1. Cài môi trường

Khuyến nghị Python 3.11, Java 17 và PySpark 3.5.4. Hướng dẫn giải thích chi tiết
từng phiên bản nằm trong [`SETUP.md`](SETUP.md).

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt

# Không bắt buộc nếu PowerShell chặn script: spark_session.py cũng tự cấu hình
. .\env.ps1
python scripts\smoke_test.py
```

Git Bash trên Windows:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip setuptools
python -m pip install -r requirements.txt
source ./env.sh
python scripts/smoke_test.py
```

Linux/macOS không cài `pywin32` trong Windows lock file; dùng runtime tối thiểu:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install pyspark==3.5.4 pandas pyarrow numpy matplotlib \
  mlflow==2.19.0 scipy scikit-learn
python scripts/smoke_test.py
```

Kết quả runtime gate phải là `ALL CHECKS PASSED (5/5)`. Nếu PowerShell không cho
chạy `env.ps1`, gọi các lệnh `python ...` trực tiếp; project tự tìm `.hadoop/bin`,
tạo `.spark-tmp` và pin đúng Python worker.

### 2. Đặt dữ liệu

Cấu trúc đang dùng:

```text
deep_learning/
├── data_raw/
│   ├── flights.csv       # 5,819,079 rows, khoảng 592 MB
│   ├── airports.csv
│   └── airlines.csv
└── Supervised_learning_pipeline_spark/
    └── data/
        └── dot_to_iata.csv
```

`data_raw/` là dữ liệu gốc bên ngoài repository. `data/dot_to_iata.csv` là bảng
mapping DOT-id → IATA cần giữ trong bài. Có thể dùng vị trí dữ liệu khác bằng
`--raw-dir D:\duong\dan\data_raw` hoặc biến môi trường `FLIGHT_RAW_DIR`.

### 3. Tiền xử lý CSV → Parquet curated

PowerShell:

```powershell
python data_prep.py --step raw --raw-dir ..\data_raw
python data_prep.py --step curate --raw-dir ..\data_raw
```

Git Bash/Linux/macOS:

```bash
python data_prep.py --step raw --raw-dir ../data_raw
python data_prep.py --step curate --raw-dir ../data_raw
```

Kết quả đúng:

```text
raw rows:     5,819,079
curated rows: 5,704,000
severe-delay rate: 0.1108
output: data/parquet/flights_curated/
```

`data/dot_to_iata.csv` đã có sẵn. Chỉ khi muốn tái tạo bảng này mới chạy:

```powershell
# Chạy sau --step raw và trước --step curate
python scripts\build_airport_code_map.py
```

### 4. Chạy các verification gates

```powershell
# Không cần dữ liệu thật; kiểm tra inheritance, Arrow, leakage và serialization
python scripts\ci_transformer_check.py
# Expected: ALL CHECKS PASSED (9/9)

# Dùng khoảng 2% curated data; phải chạy bước tiền xử lý trước
python scripts\test_transformers.py
# Expected: ALL CHECKS PASSED (8/8)

# Tùy chọn: đo thời gian từng feature-engineering stage
python scripts\profile_stages.py
```

### 5. Chạy model tournament

Chạy thử nhanh trước. Dùng tên JSON riêng để không ghi đè benchmark full-data:

```powershell
python mllib_pipeline.py --sample-fraction 0.005 --tune-fraction 0.5 `
  --folds 2 --parallelism 4 --experiment flight-delay-smoke --no-resume `
  --results-name tournament_smoke.json
```

Chạy cấu hình nộp bài: cùng split 80/20, CV 5 folds, bốn trials đồng thời. Đây là
job dài và phụ thuộc cấu hình máy:

```powershell
python mllib_pipeline.py --tune-fraction 0.1 --folds 5 --parallelism 4 --no-resume
```

Git Bash có thể chạy qua `spark-submit`:

```bash
./submit_pipeline.sh --tune-fraction 0.1 --folds 5 --parallelism 4 --no-resume
```

Chỉ chạy một hoặc nhiều model cụ thể:

```powershell
python mllib_pipeline.py --models linear_regression,linear_svc `
  --sample-fraction 0.01 --folds 2 --experiment selected-models --no-resume `
  --results-name selected_models.json
```

Tên hợp lệ cho `--models`:

```text
linear_regression, glm_poisson_log, linear_svc,
random_forest_regressor, random_forest_classifier,
gbt_regressor, gbt_classifier
```

| Flag | Mặc định | Ý nghĩa |
|---|---:|---|
| `--sample-fraction` | `1.0` | Tỷ lệ toàn bộ curated table dùng cho run |
| `--tune-fraction` | `0.1` | Tỷ lệ train split dùng tìm hyperparameter |
| `--folds` | `5` | Số CV folds |
| `--parallelism` | `4` | Số grid trials chạy đồng thời |
| `--pca-k` | `10` | Số principal components |
| `--models` | `all` | Danh sách model hoặc `all` |
| `--experiment` | `flight-delay-mllib` | Tên MLflow experiment |
| `--results-name` | `tournament_results.json` | JSON ghi vào `docs/benchmarks/` |
| `--resume` | bật | Bỏ qua arm đã hoàn thành trong MLflow |
| `--no-resume` | tắt | Chạy lại mọi arm |
| `--fail-fast` | tắt | Dừng ngay khi một arm lỗi |

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
  ci_transformer_check.py  synthetic CI gate, không cần curated data
  build_airport_code_map.py  October DOT-id -> IATA recovery
  test_transformers.py     Phase 2 gate: leak-safety + serialization
  profile_stages.py        profile thời gian từng pipeline stage
  make_stream_events.py    drip held-out rows into stream_input/
docs/
  REPORT.md                derivations, design decisions, benchmark analysis
  presentation.html        browser-ready presentation
  team-guide.html          concise demo/troubleshooting guide
  benchmarks/              generated figures and results table
data/
  dot_to_iata.csv          precomputed October DOT-id -> IATA mapping
models/
  best_pipeline/           serialized winning PipelineModel
  input_schema.json        serving contract for readStream
SETUP.md                    giải thích và cài runtime
requirements.txt           Windows environment lock
supervised_learning_pipeline_spark_subject.docx.pdf  đề bài gốc
```

Runtime outputs are intentionally not part of the submission: `data/parquet/`,
`.spark-tmp/`, `stream_input/`, `stream_output/`, `checkpoints/`, `mlruns/`, and
`mlflow.db` are reproducible and ignored by Git. The source CSVs remain in
`../data_raw/` (or another directory passed through `--raw-dir`).

The current checkout also contains a working local `mlflow.db`/`mlruns` store.
Registry verification is exported in
[`docs/benchmarks/mlflow_registry_evidence.json`](docs/benchmarks/mlflow_registry_evidence.json):
model `flight_delay_pipeline` version 1 has `Staging` and `Production` aliases,
and loading the 13-stage pipeline back from `Production` has been verified.

### 6. Mở và kiểm tra MLflow

Project dùng SQLite tracking store tại `mlflow.db`. Mở UI từ thư mục repository:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Sau đó mở [http://127.0.0.1:5000](http://127.0.0.1:5000). Kiểm tra:

1. Experiment `flight-delay-mllib`.
2. Store hiện tại có ba evidence runs; một full tournament sẽ tạo 14 parent
   PCA/no-PCA arms cùng các nested CrossValidator runs.
3. Metrics: RMSE, MAE, R², AUC-ROC, AUC-PR, F1 và accuracy.
4. Artifacts: evaluation JSON, PCA JSON/PNG, feature-importance metrics/JSON/PNG.
5. Registered Models → `flight_delay_pipeline` → version có `Production`.

Run IDs và bằng chứng Registry hiện tại nằm trong
[`docs/benchmarks/mlflow_registry_evidence.json`](docs/benchmarks/mlflow_registry_evidence.json).
Tắt UI bằng `Ctrl+C` tại terminal đang chạy MLflow.

### 7. Tạo lại và xem benchmark

Sau khi một MLflow tournament đầy đủ hoàn thành, render trực tiếp từ experiment:

```powershell
python benchmark_results.py --experiment flight-delay-mllib
```

Để tái tạo đúng bộ biểu đồ full-data đã commit mà không phụ thuộc các run đang có
trong MLflow, dùng `--from-json`. Thêm `--skip-residuals` nếu curated Parquet
không còn:

```powershell
python benchmark_results.py --from-json --skip-residuals
```

Các file cần xem:

- [Bảng kết quả](docs/benchmarks/results_table.md)
- [Tournament JSON](docs/benchmarks/tournament_results.json)
- [PCA explained variance](docs/benchmarks/pca_explained_variance.png)
- [Regression comparison](docs/benchmarks/model_comparison_regression.png)
- [Classification comparison](docs/benchmarks/model_comparison_classification.png)
- [Residual analysis](docs/benchmarks/residuals.png)
- `docs/benchmarks/feature_importance_*.png`

Git Bash cũng hỗ trợ mode benchmark:

```bash
MODE=benchmark ./submit_pipeline.sh --from-json --skip-residuals
```

| Benchmark flag | Ý nghĩa |
|---|---|
| `--experiment NAME` | Đọc metrics từ MLflow experiment được chỉ định |
| `--from-json` | Bỏ qua MLflow và dùng full `tournament_results.json` đã commit |
| `--skip-residuals` | Không load Spark model/curated data để vẽ residuals |

### 8. Xem report, presentation và team guide

Mở trực tiếp trên Windows:

```powershell
Start-Process .\docs\REPORT.md
Start-Process .\docs\presentation.html
Start-Process .\docs\team-guide.html
Start-Process .\supervised_learning_pipeline_spark_subject.docx.pdf
```

Hoặc phục vụ toàn bộ repository qua HTTP để link/ảnh hoạt động ổn định:

```powershell
python -m http.server 8000
```

Sau đó mở:

- Presentation: [http://127.0.0.1:8000/docs/presentation.html](http://127.0.0.1:8000/docs/presentation.html)
- Team guide: [http://127.0.0.1:8000/docs/team-guide.html](http://127.0.0.1:8000/docs/team-guide.html)
- Benchmark images: [http://127.0.0.1:8000/docs/benchmarks/](http://127.0.0.1:8000/docs/benchmarks/)

### 9. Chạy streaming inference

`inference.py` tạo `stream_input/`, `stream_output/` và `checkpoints/inference/`
tự động. Dùng hai terminal PowerShell.

Terminal 1 — load model đã serialize trên disk và chờ vô hạn:

```powershell
python inference.py --await-seconds 0 --trigger-seconds 5
```

Terminal 2 — đưa năm JSON micro-batches vào stream:

```powershell
python scripts\make_stream_events.py --clean --batches 5 --rows 40 --interval 4
```

Nếu `data/parquet/flights_curated/` tồn tại, event generator lấy dữ liệu từ held-out
split. Nếu không, nó sinh record demo đúng `models/input_schema.json`. Prediction
được in ở terminal 1 và ghi dạng Parquet vào `stream_output/`. Dừng bằng `Ctrl+C`.

Để chứng minh model load trực tiếp từ MLflow Production Registry:

```powershell
python inference.py --from-registry --await-seconds 120 --trigger-seconds 5
```

Git Bash tương đương:

```bash
MODE=inference ./submit_pipeline.sh --from-registry --await-seconds 0
```

| Inference flag | Mặc định | Ý nghĩa |
|---|---:|---|
| `--from-registry` | tắt | Load Production Registry thay vì `models/best_pipeline/` |
| `--await-seconds` | `120` | Thời gian chạy; `0` nghĩa là chờ vô hạn |
| `--trigger-seconds` | `5` | Chu kỳ Structured Streaming trigger |

| Event-generator flag | Mặc định | Ý nghĩa |
|---|---:|---|
| `--batches` | `5` | Số JSON files/micro-batches tạo ra |
| `--rows` | `40` | Số records mỗi batch |
| `--interval` | `4` | Số giây giữa hai batch |
| `--clean` | tắt | Xóa JSON cũ trong `stream_input/` trước khi tạo |

### 10. Các output được tạo ở đâu?

| Output | Vị trí |
|---|---|
| Raw Parquet | `data/parquet/flights_raw/` |
| Curated leak-free Parquet | `data/parquet/flights_curated/` |
| MLflow tracking database | `mlflow.db` |
| MLflow artifacts | `mlruns/` |
| Tournament/plots | `docs/benchmarks/` |
| Winning disk model | `models/best_pipeline/` |
| Serving schema | `models/input_schema.json` |
| Incoming streaming JSON | `stream_input/` |
| Streaming predictions | `stream_output/` |
| Streaming checkpoint | `checkpoints/inference/` |

### 11. Production launcher `submit_pipeline.sh`

Script hỗ trợ ba mode:

```bash
# Train
./submit_pipeline.sh --folds 5 --parallelism 4

# Streaming inference
MODE=inference ./submit_pipeline.sh --await-seconds 0

# Benchmark
MODE=benchmark ./submit_pipeline.sh --skip-residuals
```

Có thể override tài nguyên mà không sửa file:

```bash
MASTER='local[8]' DRIVER_MEM='8g' ./submit_pipeline.sh --folds 5
```

### 12. Troubleshooting nhanh

| Lỗi | Cách xử lý |
|---|---|
| Không tìm thấy `flights.csv` | Truyền đúng `--raw-dir ..\data_raw` |
| Không có `data/parquet/flights_curated` | Chạy lần lượt `--step raw`, rồi `--step curate` |
| PowerShell chặn `env.ps1` | Chạy `python ...` trực tiếp; runtime được auto-configure |
| `winutils.exe` / Hadoop permission error | Kiểm tra `.hadoop/bin/winutils.exe` và `.hadoop/bin/hadoop.dll` |
| Python worker `EOFException` | Dùng Python 3.11 và bảo đảm driver/worker dùng cùng interpreter |
| PCA `NotConvergedException` | Không bỏ `VarianceThresholdSelector` khỏi Pipeline |
| MLflow UI không có run | Kiểm tra đang mở đúng `sqlite:///mlflow.db` và đúng experiment |
| Stream không nhận file | Dùng file mới trong `stream_input/`; có thể chạy generator với `--clean` |
| Registry inference không load | Mở MLflow UI, kiểm tra alias/stage `Production` và model version |

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

### Direct Transformer plus leak-safe fitting

`OutlierIQRTruncator` directly subclasses `Transformer`, exactly as required, and
can compute Q1/Q3 itself when used standalone. In the production Pipeline,
`OutlierIQRTruncatorEstimator` learns the fences from each training/CV fold and
returns the Transformer with frozen bounds. Test targets and streaming
micro-batches therefore never influence Q1/Q3.

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

The reproducible setup is documented in [`SETUP.md`](SETUP.md). Local runtime
helpers live inside the repository and do not change system-wide configuration.

| Component | Choice | Why |
|---|---|---|
| Python | **3.11** (venv) | 3.14 breaks PySpark at import; on **3.12.0** the JVM launches the Python worker, it exits 0 without writing to its socket, and every task dies with `EOFException` |
| Java | **Temurin JDK 17** (`.jdk/`) | `JAVA_HOME` pointed at JDK 26, unsupported by every Spark release |
| PySpark | 3.5.4 | stable pairing with MLflow 2.x |
| MLflow | **2.19.0** in the lock file | Uses rubric-compatible stages on 2.x and Staging/Production aliases plus tags on 3.x |
| Tracking store | **SQLite**, not `file:` | the Model Registry is not supported by the filesystem store at all |
| Hadoop natives | 3.3.5 (`.hadoop/`) | `PipelineModel.save()` on Windows needs `hadoop.dll`; must match Spark's bundled Hadoop **3.3.4** jars, not the 3.4.1 tree in `C:\Hadoop` |

---

## Verification gates

Each phase has a gate; nothing downstream is trusted until it passes.

| Gate | Command | Expected |
|---|---|---|
| Runtime | `python scripts/smoke_test.py` | `ALL CHECKS PASSED (5/5)` |
| Synthetic CI | `python scripts/ci_transformer_check.py` | `ALL CHECKS PASSED (9/9)` |
| Data | `python data_prep.py --step curate` | 5,704,000 rows; 0 missing coordinates |
| Transformers | `python scripts/test_transformers.py` | `ALL CHECKS PASSED (8/8)` |
| Tournament | `mlflow ui --backend-store-uri sqlite:///mlflow.db` | 14 PCA/no-PCA parent arms, nested CV runs, a model in **Production** |
| Streaming | `inference.py` + `make_stream_events.py` | predictions on the console sink and in `stream_output/` |

## Submission checklist

- Source: custom transformers, pipeline, benchmark and streaming inference.
- Theory: `docs/REPORT.md` with scaling, PCA/SVD, ElasticNet, GLM, SVM, RF and GBT mathematics.
- Presentation: `docs/presentation.html`.
- Evidence: full tournament JSON/table/plots and feature importances under `docs/benchmarks/`.
- Deployment: serialized `models/best_pipeline/`, serving schema and `submit_pipeline.sh`.
- Reproducibility: pinned requirements, setup notes, smoke tests and GitHub CI.

---

## Source

Assignment brief: `supervised_learning_pipeline_spark_subject.docx.pdf`.
Dataset: [Kaggle — US Flight Delays and Performance Data](https://www.kaggle.com/datasets/usdot/flight-delays)
(place `flights.csv`, `airports.csv`, `airlines.csv` either in `data/` or the
sibling `../data_raw/` directory; use `--raw-dir PATH` for any other location).
