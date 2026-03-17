param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

switch ($Command) {

    "help" {
        Write-Host ""
        Write-Host "  Credit Card Fraud Detection - Available Commands" -ForegroundColor Cyan
        Write-Host "  ------------------------------------------------"
        Write-Host "  .\run.ps1 validate   Run Pandera data validation on raw data"
        Write-Host "  .\run.ps1 train      Train XGBoost model and log run to MLflow"
        Write-Host "  .\run.ps1 evaluate   Evaluate Production model from MLflow registry"
        Write-Host "  .\run.ps1 serve      Start FastAPI server at http://localhost:8000"
        Write-Host "  .\run.ps1 mlflow     Launch MLflow UI at http://localhost:5000"
        Write-Host "  .\run.ps1 dagster    Launch Dagster UI at http://localhost:3000"
        Write-Host "  .\run.ps1 monitor    Generate Evidently AI drift report"
        Write-Host "  .\run.ps1 dvc        Run the full DVC pipeline"
        Write-Host "  .\run.ps1 test       Run all pytest unit tests"
        Write-Host "  .\run.ps1 lint       Run ruff linter across the codebase"
        Write-Host "  .\run.ps1 clean      Remove cache and temp files"
        Write-Host ""
    }

    "validate" {
        Write-Host "Running Pandera data validation..." -ForegroundColor Cyan
        uv run python -m src.validate
    }

    "train" {
        Write-Host "Training XGBoost model..." -ForegroundColor Cyan
        uv run python -m src.train
    }

    "evaluate" {
        Write-Host "Evaluating Production model from MLflow registry..." -ForegroundColor Cyan
        uv run python -m src.evaluate
    }

    "serve" {
        Write-Host "Starting FastAPI server at http://localhost:8000" -ForegroundColor Cyan
        Write-Host "Swagger docs available at http://localhost:8000/docs" -ForegroundColor Cyan
        uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    }

    "mlflow" {
        Write-Host "Launching MLflow UI at http://localhost:5000" -ForegroundColor Cyan
        uv run mlflow ui --host 0.0.0.0 --port 5000
    }

    "dagster" {
        Write-Host "Launching Dagster UI at http://localhost:3000" -ForegroundColor Cyan
        uv run dagster dev -f orchestration/__init__.py
    }

    "monitor" {
        Write-Host "Generating Evidently AI drift report..." -ForegroundColor Cyan
        uv run python -m monitoring.generate_report
        Write-Host "Report saved to monitoring\dashboards\. Open the HTML file in a browser." -ForegroundColor Green
    }

    "dvc" {
        Write-Host "Running full DVC pipeline..." -ForegroundColor Cyan
        uv run dvc repro
    }

    "test" {
        Write-Host "Running pytest..." -ForegroundColor Cyan
        uv run pytest tests/ -v --tb=short
    }

    "lint" {
        Write-Host "Running ruff linter..." -ForegroundColor Cyan
        uv run ruff check src/ api/ tests/ orchestration/ monitoring/
    }

    "clean" {
        Write-Host "Cleaning up cache files..." -ForegroundColor Cyan
        Get-ChildItem -Recurse -Directory -Filter "__pycache__"  | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Directory -Filter ".pytest_cache" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Directory -Filter "*.egg-info"   | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter "*.pyc"                   | Remove-Item -Force -ErrorAction SilentlyContinue
        Write-Host "Clean complete." -ForegroundColor Green
    }

    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host "Run .\run.ps1 help to see available commands." -ForegroundColor Yellow
    }
}
