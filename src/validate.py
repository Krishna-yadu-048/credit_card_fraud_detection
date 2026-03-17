import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml
from pandera.pandas import Column, DataFrameSchema, Check

# ── logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

# ── resolve project root so paths always work regardless of how script is called
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str = "configs/model_config.yaml") -> dict:
    full_path = PROJECT_ROOT / config_path
    with open(full_path) as f:
        return yaml.safe_load(f)


# ── Pandera schema ─────────────────────────────────────────────────────────────
def build_schema() -> DataFrameSchema:
    v_columns = {
        f"V{i}": Column(float, nullable=False)
        for i in range(1, 29)
    }
    schema = DataFrameSchema(
        columns={
            **v_columns,
            "Time": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "Amount": Column(float, Check.greater_than_or_equal_to(0), nullable=False),
            "Class": Column(
                int,
                Check.isin([0, 1]),
                nullable=False,
            ),
        },
        coerce=True,
    )
    return schema


# ── cleaning ──────────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    log.info("Applying data cleaning steps...")

    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Dropped {before - len(df)} duplicate rows.")

    if config["data"]["amount_log_transform"]:
        df["Amount"] = np.log1p(df["Amount"])
        log.info("Log-transformed Amount column.")

    cols_to_drop = [c for c in config["data"]["drop_columns"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info(f"Dropped columns: {cols_to_drop}")

    return df


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    config = load_config()

    raw_path       = PROJECT_ROOT / config["paths"]["raw_data"]
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]
    reference_path = PROJECT_ROOT / config["paths"]["reference_data"]

    log.info(f"Project root:      {PROJECT_ROOT}")
    log.info(f"Loading raw data from: {raw_path}")

    if not raw_path.exists():
        log.error(f"Raw data file not found: {raw_path}")
        log.error("Download creditcard.csv from Kaggle and place it in data/raw/")
        sys.exit(1)

    df = pd.read_csv(raw_path)
    log.info(f"Raw data shape: {df.shape}")

    # validate
    log.info("Running Pandera schema validation...")
    schema = build_schema()
    try:
        schema.validate(df, lazy=True)
        log.info("Schema validation passed")
    except pa.errors.SchemaErrors as e:
        log.error("Schema validation FAILED. See errors below:")
        log.error(e.failure_cases)
        sys.exit(1)

    # clean
    df_clean = clean_data(df, config)

    # save processed data
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(processed_path, index=False)
    log.info(f"Processed data saved to: {processed_path} | shape: {df_clean.shape}")

    # save reference snapshot for Evidently drift monitoring
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.sample(frac=0.1, random_state=42).to_csv(reference_path, index=False)
    log.info(f"Reference snapshot saved to: {reference_path}")

    log.info("Validation complete")


if __name__ == "__main__":
    main()