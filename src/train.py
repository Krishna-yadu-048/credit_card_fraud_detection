import logging
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    with open(PROJECT_ROOT / config_path) as f:
        return yaml.safe_load(f)


def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def apply_smote(X_train, y_train, random_state: int):
    """
    SMOTE creates synthetic minority-class samples so the model
    sees a more balanced dataset during training.
    Only applied to training data — never to the test set.
    """
    log.info(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    log.info(f"Class distribution after SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def build_model(config: dict) -> XGBClassifier:
    model_cfg = config["model"]
    return XGBClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        learning_rate=model_cfg["learning_rate"],
        subsample=model_cfg["subsample"],
        colsample_bytree=model_cfg["colsample_bytree"],
        scale_pos_weight=model_cfg["scale_pos_weight"],
        eval_metric=model_cfg["eval_metric"],
        random_state=model_cfg["random_state"],
        use_label_encoder=False,
        verbosity=0,
    )


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "avg_precision": round(average_precision_score(y_true, y_proba), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
    }


def main():
    config = load_config()

    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]
    test_path      = PROJECT_ROOT / "data" / "processed" / "test_set.csv"

    log.info(f"Loading processed data from: {processed_path}")
    X, y = load_data(processed_path)
    log.info(f"Dataset shape: X={X.shape}, fraud rate={y.mean():.4%}")

    # train/test split — stratify keeps fraud ratio equal in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["model"]["random_state"],
        stratify=y,
    )
    log.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")

    # apply SMOTE to training data only
    X_train_res, y_train_res = apply_smote(
        X_train, y_train,
        random_state=config["data"]["smote_random_state"],
    )

    # MLflow run
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        log.info(f"MLflow run ID: {run.info.run_id}")

        mlflow.log_params(config["model"])
        mlflow.log_param("smote_applied", True)
        mlflow.log_param("test_size", config["data"]["test_size"])

        model = build_model(config)
        log.info("Training XGBoost model...")
        model.fit(X_train_res, y_train_res)

        # evaluate on the original (non-SMOTE) test set
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        log.info("── Evaluation Results ──────────────────────────")
        for k, v in metrics.items():
            log.info(f"  {k}: {v}")
        log.info("────────────────────────────────────────────────")

        mlflow.log_metrics(metrics)

        report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])
        mlflow.log_text(report, "classification_report.txt")

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=config["mlflow"]["model_name"],
        )

        # save test split so evaluate.py can load it
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_data = X_test.copy()
        test_data["Class"] = y_test.values
        test_data.to_csv(test_path, index=False)
        mlflow.log_artifact(str(test_path))

        log.info(f"Model registered as '{config['mlflow']['model_name']}' in MLflow registry.")

    log.info("Training complete")


if __name__ == "__main__":
    main()