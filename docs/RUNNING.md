# Running MLflow and the streaming inference demo

Two things you may be asked to show live: the MLflow tracking UI, and the
serialized pipeline scoring a stream. Neither needs training to be re-run.

PowerShell commands are given first because that is the default shell here, and
**Windows PowerShell 5.1 has no `&&`** (it is a parser error, not a warning), so
every step is its own line. Git Bash equivalents follow each block.

---

## 1. MLflow tracking UI

```powershell
cd D:\BigData
. .\env.ps1
.\.venv\Scripts\mlflow.exe ui --backend-store-uri "sqlite:///D:/BigData/mlflow.db" --host 127.0.0.1 --port 5000
```

Git Bash:

```bash
cd /d/BigData
source ./env.sh
./.venv/Scripts/mlflow.exe ui --backend-store-uri "sqlite:///D:/BigData/mlflow.db" --host 127.0.0.1 --port 5000
```

Then open <http://127.0.0.1:5000>. Leave the terminal running; `Ctrl+C` stops it.

### Two things that will bite

**Use `.\.venv\Scripts\mlflow.exe`, never a bare `mlflow` or `python`.** System
Python 3.14 carries MLflow 3.8.1, and any client call from it silently
Alembic-migrates `mlflow.db` to the 3.x schema. The pinned 2.19.0 then refuses to
open it with `CommandError: Can't locate revision identified by
'1bd49d398cd23'`. Recovery is possible because the 3.x migrations are additive:
`UPDATE alembic_version SET version_num='0584bdc529eb'` restores 2.19.0's head
with no data loss. Easier not to trigger it.

**Pass `--backend-store-uri` explicitly, with a Windows-style path.** Without it
MLflow defaults to a `./mlruns` file store, which shows none of the registered
models. `sqlite:///$PWD/mlflow.db` does **not** work; the absolute
`sqlite:///D:/BigData/mlflow.db` does. The same string is defined once in code as
`spark_session.TRACKING_URI`.

### What to point at

| where | what is there |
|---|---|
| Experiments, `flight-delay-mllib` | the 14 tournament arms as top-level runs, each with nested cross-validation children. Sort by `rmse` or `r2`. |
| Models, `flight_delay_pipeline` | **v4 in Production**, `random_forest_regressor__pca`. v1 is Archived. |

The Production run's parameters are worth showing, because they answer questions
before they are asked: `arm=pca`, `svd_mode=local-eigs`, `r2=0.8545`,
`rmse=15.0479`, `pca_variance_retained=0.2846`, `train_seconds=2743` (45.7
minutes on the full training split).

---

## 2. Streaming inference

Two terminals. The first scores, the second feeds it.

### Terminal 1: start the scorer

```powershell
cd D:\BigData
. .\env.ps1
python inference.py --from-registry --await-seconds 0
```

Wait for these three lines before touching terminal 2:

```
loading models:/flight_delay_pipeline/Production
loaded PipelineModel with 13 stages
watching D:\BigData\stream_input - drop JSON files there to score them
```

`--await-seconds 0` means run until stopped, so it will not die mid-demo.
`Ctrl+C` ends it.

`--from-registry` loads `models:/flight_delay_pipeline/Production` through
MLflow, which is what proves the registered artifact is servable rather than
merely recorded. Drop the flag to load `models\best_pipeline` from disk instead:
same model, no tracking server needed, useful if MLflow is not reachable.

### Terminal 2: feed it flights

```powershell
cd D:\BigData
. .\env.ps1
python scripts\make_stream_events.py --clean --batches 5 --rows 40 --interval 3
```

Predictions appear in terminal 1 within a few seconds, one table per file.

### `--clean` matters more than it looks

Without it you will hit a result that looks like failure and is not.
`stream_input\` already holds files the Parquet sink consumed on an earlier run,
and its checkpoint correctly refuses to score them twice, so the job reports:

```
rows scored this session: 0
```

That is the exactly-once guarantee working. Re-processing would have been the
bug. `--clean` empties the folder first so every file is genuinely new.

If a rehearsal has already pushed new files through the checkpoint and you want
the counter to start from zero, delete the checkpoint **before** starting
terminal 1:

```powershell
Remove-Item -Recurse -Force D:\BigData\checkpoints\inference
```

Safe: it holds stream offsets, not model data. `stream_output\` keeps its
existing parquet files and the sink simply starts counting again.

### Three things worth pointing at while it runs

1. `loaded PipelineModel with 13 stages` proves the registry round-trip, and
   that the custom transformers deserialized with it.
2. A row whose `DEPARTURE_DELAY` exceeds 25 while the prediction still tracks it.
   The truncator's upper fence is 25, so this shows the raw column survived
   clipping, which is the single most important feature decision in the project.
3. Run terminal 2 again **without** `--clean`: nothing is re-scored. The
   checkpoint refusing to duplicate is the point.

---

## What neither of these does

Neither command trains anything, registers a model version, or writes to
`models\`. Both are safe to run repeatedly in front of an audience.

`MODE=train` through `submit_pipeline.sh` is the dangerous one: it calls
`register_winner()` unconditionally and promotes with
`archive_existing_versions=True`, so even `--sample-fraction 0.01` will register
a new version, archive the real winner and overwrite `models\best_pipeline`.
Back up `mlflow.db`, `models\` and `docs\benchmarks\tournament_results.json`
before any training run you do not intend to keep. See CLAUDE.md section 2.
