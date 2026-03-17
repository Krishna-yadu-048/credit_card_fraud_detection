from dagster import Definitions

from orchestration.assets import (
    raw_data,
    validated_data,
    trained_model,
    evaluated_model,
    drift_report,
)
from orchestration.schedules import weekly_retraining, daily_drift_check

defs = Definitions(
    assets=[raw_data, validated_data, trained_model, evaluated_model, drift_report],
    schedules=[weekly_retraining, daily_drift_check],
)