# Runtime setup

The runtime is project-local: `.venv/`, `.jdk/`, `.hadoop/`, and `.spark-tmp/`
all live under the repository root. Both `env.ps1` and `env.sh` resolve that
root dynamically, so the checkout can be moved or renamed safely and nothing
system-wide is modified.

## Why each pin exists

| Component | Choice | Why |
|---|---|---|
| Python | **3.11.15** (via `uv`, project venv) | System Python 3.14 breaks PySpark at import (`typing.io` removed in 3.13). Python **3.12.0 also fails**: the JVM launches the Python worker, the worker exits 0 without writing to its socket, and every task dies with `Python worker exited unexpectedly (crashed)` / `EOFException`. Only the driver-side Arrow path works there, which makes it look like an Arrow problem — it is not. 3.11 is the newest interpreter PySpark 3.5 fully supports. |
| Java | **Temurin JDK 17** (`.jdk/`) | `JAVA_HOME` pointed at JDK 26, which no Spark release supports. Installed as a zip, so no admin/UAC. JDK 26 and JRE 8 are left untouched. |
| PySpark | **3.5.4** | Stable pairing with MLflow 2.x. |
| MLflow | **2.19.0** | Deliberately 2.x: the brief requires registry **stage tags** (`Staging → Production`), and `transition_model_version_stage` is removed in MLflow 3.x in favour of aliases. |
| Hadoop natives | **3.3.5** `winutils.exe` + `hadoop.dll` (`.hadoop/`) | `PipelineModel.save()` on Windows needs `hadoop.dll`; `C:\Hadoop` had `winutils.exe` only. 3.3.5 is chosen to match PySpark's bundled `hadoop-client-*-3.3.4.jar`, **not** the 3.4.1 tree in `C:\Hadoop`. |
| setuptools | required | Supplies the `distutils` shim that `pyspark/ml/image.py` still imports. |

## First-time setup

```bash
uv python install 3.11
"$(uv python find 3.11)" -m venv .venv
./.venv/Scripts/python.exe -m pip install -U pip setuptools
./.venv/Scripts/python.exe -m pip install -r requirements.txt
# JDK 17 -> .jdk/    (Adoptium API, zip)
# hadoop.dll + winutils.exe -> .hadoop/bin/   (cdarlint/winutils, hadoop-3.3.5)
```

## Every session

```bash
source ./env.sh          # Git Bash
. .\env.ps1              # PowerShell (optional; Python also auto-configures runtime)
python scripts/smoke_test.py   # must print ALL CHECKS PASSED (5/5)
```

`scripts/smoke_test.py` is the gate: it proves JVM startup, an Arrow
`pandas_udf`, and a `PipelineModel` save/load round-trip all work before
any real job is run.

If PowerShell execution policy blocks `env.ps1`, run the Python commands
directly. `spark_session.py` discovers `.hadoop/bin`, configures temporary
storage, and pins the active Python interpreter before the JVM starts.
