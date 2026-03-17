from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.97, 0.03]])
    mock_model.predict.return_value       = np.array([0])

    with patch("mlflow.xgboost.load_model", return_value=mock_model):
        with patch("mlflow.set_tracking_uri"):
            from api.main import app
            with TestClient(app) as c:
                yield c


VALID_TRANSACTION = {
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40, "V20": 0.25,
    "V21": -0.02, "V22": 0.28, "V23": -0.11, "V24": 0.07, "V25": 0.13,
    "V26": -0.19, "V27": 0.13, "V28": -0.02,
    "Amount": 149.62,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_model_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "ok"


class TestPredictEndpoint:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=VALID_TRANSACTION)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=VALID_TRANSACTION).json()
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "label" in data

    def test_predict_probability_range(self, client):
        prob = client.post("/predict", json=VALID_TRANSACTION).json()["fraud_probability"]
        assert 0.0 <= prob <= 1.0

    def test_predict_label_matches_is_fraud(self, client):
        data = client.post("/predict", json=VALID_TRANSACTION).json()
        if data["is_fraud"]:
            assert data["label"] == "Fraud"
        else:
            assert data["label"] == "Legit"

    def test_predict_missing_field_returns_422(self, client):
        bad_payload = {k: v for k, v in VALID_TRANSACTION.items() if k != "V1"}
        assert client.post("/predict", json=bad_payload).status_code == 422

    def test_predict_negative_amount_returns_422(self, client):
        bad_payload = {**VALID_TRANSACTION, "Amount": -50.0}
        assert client.post("/predict", json=bad_payload).status_code == 422
