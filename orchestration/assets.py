import json
import os
import subprocess
import sys
from pathlib import Path

from dagster import asset, Output, MetadataValue

# always use the same Python that is running Dagster
PYTHON = sys.executable
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# force UTF-8 encoding for all subprocess calls on Windows
SUBPROCESS_ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


@asset(group_name="data", description="Raw creditcard.csv downloaded from Kaggle.")
def raw_data():
    """Checks the raw data file exists before the pipeline starts."""
    path = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
    assert path.exists(), (
        f"Raw data not found at {path}. "
        "Download creditcard.csv from Kaggle and place it in data/raw/"
    )
    return Output(value=str(path), metadata={"path": MetadataValue.path(str(path))})


@asset(group_name="data", deps=["raw_data"],
       description="Validated and cleaned data. Pandera schema checks applied.")
def validated_data():
    """Runs validate.py to clean data and write to data/processed/."""
    result = subprocess.run(
        [PYTHON, "-m", "src.validate"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=SUBPROCESS_ENV,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise Exception(f"Validation failed:\n{result.stderr}")

    path = PROJECT_ROOT / "data" / "processed" / "creditcard_processed.csv"
    return Output(value=str(path), metadata={"path": MetadataValue.path(str(path))})


@asset(group_name="model", deps=["validated_data"],
       description="Trained XGBoost model logged and registered in MLflow.")
def trained_model():
    """Runs train.py — applies SMOTE and logs the run to MLflow."""
    result = subprocess.run(
        [PYTHON, "-m", "src.train"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=SUBPROCESS_ENV,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise Exception(f"Training failed:\n{result.stderr}")

    return Output(
        value="MLflow Model Registry",
        metadata={"note": MetadataValue.text("Promote to Production via MLflow UI or promote_model.py")},
    )


@asset(group_name="model", deps=["trained_model"],
       description="Evaluation metrics for the Production model against the test set.")
def evaluated_model():
    """Runs evaluate.py and surfaces metrics in the Dagster UI."""
    result = subprocess.run(
        [PYTHON, "-m", "src.evaluate"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=SUBPROCESS_ENV,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise Exception(f"Evaluation failed:\n{result.stderr}")

    metrics_path = PROJECT_ROOT / "data" / "processed" / "metrics.json"
    with open(metrics_path) as f:
        metrics = json.load(f)

    return Output(
        value=metrics,
        metadata={k: MetadataValue.float(v) for k, v in metrics.items()},
    )


@asset(group_name="monitoring", deps=["evaluated_model"],
       description="Evidently AI drift report comparing training vs live data.")
def drift_report():
    """Runs the Evidently drift report generator."""
    result = subprocess.run(
        [PYTHON, "-m", "monitoring.generate_report"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=SUBPROCESS_ENV,
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise Exception(f"Drift report failed:\n{result.stderr}")

    return Output(
        value="monitoring/dashboards/",
        metadata={"note": MetadataValue.text("Open the HTML report in a browser to inspect feature drift.")},
    )