import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from sklearn.datasets import make_classification
from xgboost import XGBClassifier

from src.validate import build_schema


def make_dummy_transaction_df(n=100, fraud_rate=0.1) -> pd.DataFrame:
    """Creates a small synthetic DataFrame that mirrors the creditcard dataset schema."""
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    rows = []
    for label in [0] * n_legit + [1] * n_fraud:
        row = {f"V{i}": np.random.randn() for i in range(1, 29)}
        row["Time"]   = np.random.uniform(0, 172792)
        row["Amount"] = np.random.uniform(0, 500)
        row["Class"]  = label
        rows.append(row)

    return pd.DataFrame(rows)


class TestPanderaSchema:
    def test_valid_data_passes(self):
        df     = make_dummy_transaction_df()
        schema = build_schema()
        assert schema.validate(df) is not None

    def test_missing_column_fails(self):
        df     = make_dummy_transaction_df().drop(columns=["V1"])
        schema = build_schema()
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_invalid_class_label_fails(self):
        df           = make_dummy_transaction_df()
        df.loc[0, "Class"] = 5
        schema       = build_schema()
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_negative_amount_fails(self):
        df           = make_dummy_transaction_df()
        df.loc[0, "Amount"] = -10.0
        schema       = build_schema()
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)

    def test_null_values_fail(self):
        df           = make_dummy_transaction_df()
        df.loc[0, "V5"] = None
        schema       = build_schema()
        with pytest.raises(pa.errors.SchemaError):
            schema.validate(df)


class TestModelOutput:
    @pytest.fixture
    def trained_model(self):
        X, y = make_classification(
            n_samples=500,
            n_features=29,
            n_informative=10,
            weights=[0.95, 0.05],
            random_state=42,
        )
        model = XGBClassifier(n_estimators=10, random_state=42, verbosity=0, use_label_encoder=False)
        model.fit(X, y)
        return model

    def test_predict_returns_binary(self, trained_model):
        X, _ = make_classification(n_samples=20, n_features=29, random_state=0)
        preds = trained_model.predict(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, trained_model):
        X, _ = make_classification(n_samples=20, n_features=29, random_state=0)
        proba = trained_model.predict_proba(X)
        assert proba.shape == (20, 2)

    def test_predict_proba_sums_to_one(self, trained_model):
        X, _ = make_classification(n_samples=50, n_features=29, random_state=0)
        proba = trained_model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_fraud_proba_in_range(self, trained_model):
        X, _ = make_classification(n_samples=50, n_features=29, random_state=0)
        fraud_proba = trained_model.predict_proba(X)[:, 1]
        assert (fraud_proba >= 0).all() and (fraud_proba <= 1).all()