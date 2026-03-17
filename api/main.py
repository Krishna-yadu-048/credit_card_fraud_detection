import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import mlflow.xgboost
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictionResponse, TransactionFeatures

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

with open(PROJECT_ROOT / "configs" / "model_config.yaml") as f:
    config = yaml.safe_load(f)

# ── model store ───────────────────────────────────────────────────────────────
model = None


# ── lifespan handler (replaces deprecated @app.on_event) ─────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs on startup
    global model
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)

    model_name = config["mlflow"]["model_name"]
    alias      = config["mlflow"]["registered_model_alias"]
    model_uri  = f"models:/{model_name}@{alias}"

    log.info(f"Loading model from MLflow registry: {model_uri}")
    try:
        model = mlflow.xgboost.load_model(model_uri)
        log.info("Model loaded successfully")
    except Exception as e:
        log.error(f"Failed to load model: {e}")

    yield
    # runs on shutdown (nothing to clean up here)


# ── app setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Predicts whether a credit card transaction is fraudulent. "
        "Model: XGBoost trained on the ULB creditcard dataset with SMOTE + class weights."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the API is running and the model is loaded."""
    return HealthResponse(status="ok", model_loaded=model is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionFeatures):
    """
    Predict whether a transaction is fraudulent.
    Returns is_fraud, fraud_probability, and a human-readable label.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check MLflow connection.")

    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    raw = transaction.model_dump()

    # apply the same log transform used in validate.py
    raw["Amount"] = np.log1p(raw["Amount"])

    features = np.array([[raw[col] for col in feature_names]])

    fraud_prob = float(model.predict_proba(features)[0][1])
    is_fraud   = fraud_prob >= 0.5

    return PredictionResponse(
        is_fraud=is_fraud,
        fraud_probability=round(fraud_prob, 4),
        label="Fraud" if is_fraud else "Legit",
    )