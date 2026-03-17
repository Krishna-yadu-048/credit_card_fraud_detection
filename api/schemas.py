from pydantic import BaseModel, Field


class TransactionFeatures(BaseModel):
    """
    Input schema for a single credit card transaction.
    V1-V28 are PCA-transformed features. Amount is the raw transaction value in USD.
    """
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0.0, description="Transaction amount in USD (must be >= 0)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.3598071, "V2": -0.0727812, "V3": 2.5363467,
                "V4": 1.3781552, "V5": -0.3383208, "V6": 0.4623878,
                "V7": 0.2395986, "V8": 0.0986979, "V9": 0.3637870,
                "V10": 0.0907941, "V11": -0.5515995, "V12": -0.6178009,
                "V13": -0.9913898, "V14": -0.3111694, "V15": 1.4681770,
                "V16": -0.4704005, "V17": 0.2079708, "V18": 0.0257905,
                "V19": 0.4039936, "V20": 0.2514121, "V21": -0.0183067,
                "V22": 0.2778375, "V23": -0.1104740, "V24": 0.0669281,
                "V25": 0.1285394, "V26": -0.1891093, "V27": 0.1335584,
                "V28": -0.0210531, "Amount": 149.62,
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response returned by the /predict endpoint."""
    is_fraud: bool = Field(..., description="True if the transaction is predicted as fraudulent")
    fraud_probability: float = Field(..., description="Model confidence that this is fraud (0-1)")
    label: str = Field(..., description="Human-readable label: 'Fraud' or 'Legit'")


class HealthResponse(BaseModel):
    """Response returned by the /health endpoint."""
    status: str
    model_loaded: bool
