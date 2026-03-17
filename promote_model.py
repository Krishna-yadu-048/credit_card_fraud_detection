import mlflow
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
with open(PROJECT_ROOT / "configs" / "model_config.yaml") as f:
    config = yaml.safe_load(f)

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = mlflow.MlflowClient()

model_name = config["mlflow"]["model_name"]
alias      = config["mlflow"]["registered_model_alias"]

# get the latest version
versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    print(f"No versions found for model '{model_name}'. Run .\\run.ps1 train first.")
    exit(1)

latest_version = max(versions, key=lambda v: int(v.version))

print(f"Model:   {model_name}")
print(f"Version: {latest_version.version}")
print(f"Setting alias '{alias}' on version {latest_version.version}...")

client.set_registered_model_alias(
    name=model_name,
    alias=alias,
    version=latest_version.version,
)

print(f"\nDone! Model '{model_name}' version {latest_version.version} is now aliased as '{alias}'.")
print("You can now run .\\run.ps1 serve")