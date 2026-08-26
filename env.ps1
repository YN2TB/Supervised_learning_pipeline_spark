# Spark runtime environment for the Big Data MLlib assignment (PowerShell).
# Usage:  . .\env.ps1
#
# Every value here is project-local: nothing outside D:\BigData is modified.
# The system JDK 26 (unsupported by Spark) and the Hadoop 3.4.1 install in
# C:\Hadoop (whose natives do not match Spark's bundled Hadoop 3.3.4 client
# jars) are both deliberately bypassed.

$env:JAVA_HOME       = 'D:\BigData\.jdk\jdk-17.0.20.1+1'
$env:HADOOP_HOME     = 'D:\BigData\.hadoop'
$env:SPARK_LOCAL_DIRS = 'D:\BigData\.spark-tmp'

# Executors and driver must both launch the venv interpreter, otherwise Spark
# picks whatever `python` is first on PATH (here: a broken 3.14 + pyspark 3.4).
$env:PYSPARK_PYTHON        = 'D:\BigData\.venv\Scripts\python.exe'
$env:PYSPARK_DRIVER_PYTHON = 'D:\BigData\.venv\Scripts\python.exe'

# So `import custom_transformers` resolves when a saved PipelineModel is
# deserialized.
$env:PYTHONPATH = "D:\BigData;$env:PYTHONPATH"

$env:PATH = "D:\BigData\.jdk\jdk-17.0.20.1+1\bin;D:\BigData\.hadoop\bin;D:\BigData\.venv\Scripts;$env:PATH"

$env:PYTHONUTF8 = '1'
