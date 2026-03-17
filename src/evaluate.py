import json
import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    model_name = config["mlflow"]["model_name"]
    alias      = config["mlflow"]["registered_model_alias"]

    model_uri = f"models:/{model_name}@{alias}"
    log.info(f"Loading model from MLflow: {model_uri}")
    model = mlflow.xgboost.load_model(model_uri)
    log.info("Model loaded successfully")

    test_path = PROJECT_ROOT / "data" / "processed" / "test_set.csv"
    log.info(f"Loading test data from: {test_path}")
    df_test = pd.read_csv(test_path)

    X_test = df_test.drop(columns=["Class"])
    y_test = df_test["Class"]
    log.info(f"Test set shape: {X_test.shape} | Fraud cases: {y_test.sum()}")

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "avg_precision_score": round(average_precision_score(y_test, y_proba), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }

    log.info("── Production Model Evaluation ─────────────────")
    for k, v in metrics.items():
        log.info(f"  {k}: {v}")
    log.info("────────────────────────────────────────────────")

    print("\n── Full Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    output_path = PROJECT_ROOT / config["paths"]["metrics_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to: {output_path}")


if __name__ == "__main__":
    main()