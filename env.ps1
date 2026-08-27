# Spark runtime environment for the Big Data MLlib assignment (PowerShell).
# Usage:  . .\env.ps1
#
# Resolve from this file, so cloning/moving the repository never invalidates
# the runtime configuration.
$ProjectRoot = $PSScriptRoot
$LocalJdk = Get-ChildItem -Directory -LiteralPath (Join-Path $ProjectRoot '.jdk') -ErrorAction SilentlyContinue | Select-Object -First 1
if ($LocalJdk) { $env:JAVA_HOME = $LocalJdk.FullName }
$LocalHadoop = Join-Path $ProjectRoot '.hadoop'
if (Test-Path -LiteralPath $LocalHadoop) { $env:HADOOP_HOME = $LocalHadoop }
$env:SPARK_LOCAL_DIRS = Join-Path $ProjectRoot '.spark-tmp'
New-Item -ItemType Directory -Force -Path $env:SPARK_LOCAL_DIRS | Out-Null

# Executors and driver must both launch the venv interpreter, otherwise Spark
# picks whatever `python` is first on PATH (here: a broken 3.14 + pyspark 3.4).
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $VenvPython)) {
    $VenvPython = (Get-Command python -ErrorAction Stop).Source
}
$env:PYSPARK_PYTHON        = $VenvPython
$env:PYSPARK_DRIVER_PYTHON = $VenvPython

# So `import custom_transformers` resolves when a saved PipelineModel is
# deserialized.
$env:PYTHONPATH = "$ProjectRoot;$env:PYTHONPATH"

$RuntimeBins = @()
if ($LocalJdk) { $RuntimeBins += (Join-Path $LocalJdk.FullName 'bin') }
if (Test-Path -LiteralPath $LocalHadoop) { $RuntimeBins += (Join-Path $LocalHadoop 'bin') }
if (Test-Path -LiteralPath (Split-Path $VenvPython)) { $RuntimeBins += (Split-Path $VenvPython) }
$env:PATH = (($RuntimeBins + $env:PATH) -join ';')

$env:PYTHONUTF8 = '1'
