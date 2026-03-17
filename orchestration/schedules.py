from dagster import ScheduleDefinition, define_asset_job, AssetSelection

full_pipeline_job = define_asset_job(
    name="full_fraud_pipeline",
    selection=AssetSelection.all(),
)

# runs every Monday at 02:00 UTC
weekly_retraining = ScheduleDefinition(
    job=full_pipeline_job,
    cron_schedule="0 2 * * 1",
    name="weekly_fraud_retraining",
    description="Retrains the fraud model every Monday at 02:00 UTC.",
)

drift_check_job = define_asset_job(
    name="daily_drift_check",
    selection=AssetSelection.keys("drift_report"),
)

# runs every day at 06:00 UTC
daily_drift_check = ScheduleDefinition(
    job=drift_check_job,
    cron_schedule="0 6 * * *",
    name="daily_drift_check",
    description="Generates a daily Evidently drift report to catch distribution shift early.",
)