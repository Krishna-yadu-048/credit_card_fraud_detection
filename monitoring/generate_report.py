import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from evidently import Report
from evidently.presets import DataDriftPreset

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

    reference_path = PROJECT_ROOT / config["paths"]["reference_data"]
    processed_path = PROJECT_ROOT / config["paths"]["processed_data"]

    log.info(f"Loading reference data from: {reference_path}")
    reference_df = pd.read_csv(reference_path)

    log.info(f"Simulating live data from: {processed_path}")
    full_df    = pd.read_csv(processed_path)
    current_df = full_df.sample(n=min(5000, len(full_df)), random_state=99)

    log.info(f"Reference size: {len(reference_df)} | Current size: {len(current_df)}")

    # only pass feature columns — drop the target
    feature_cols       = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    reference_features = reference_df[feature_cols]
    current_features   = current_df[feature_cols]

    report = Report(metrics=[DataDriftPreset()])

    # in Evidently v0.7+, run() returns a Snapshot object which has save_html()
    snapshot = report.run(reference_data=reference_features, current_data=current_features)

    output_dir = PROJECT_ROOT / "monitoring" / "dashboards"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"drift_report_{timestamp}.html"

    snapshot.save_html(str(output_path))

    log.info(f"Drift report saved to: {output_path}")
    log.info("Open the HTML file in a browser to inspect feature drift.")


if __name__ == "__main__":
    main()