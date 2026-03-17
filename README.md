# 💳 Credit Card Fraud Detection — MLOps Pipeline

An end-to-end MLOps project that detects fraudulent credit card transactions using XGBoost.
The pipeline covers data validation, model training, experiment tracking, REST API serving,
drift monitoring, and automated retraining orchestration.

Built as a portfolio project to demonstrate MLOps best practices including experiment
tracking, model registry, CI/CD, data versioning, and production monitoring.

> **Windows users:** This project uses `.\setup.ps1` and `.\run.ps1` instead of a Makefile.
> All commands use `uv run` so you never need to manually activate the virtual environment.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Pipeline Flow](#-pipeline-flow)
- [Quickstart](#-quickstart)
- [Running the Full Pipeline](#-running-the-full-pipeline)
- [API Usage](#-api-usage)
- [Running Tests](#-running-tests)
- [Orchestration with Dagster](#-orchestration-with-dagster)
- [Drift Monitoring](#-drift-monitoring)
- [CI/CD](#-cicd)

---

## 🔍 Project Overview

Credit card fraud detection is a classic imbalanced classification problem — only **0.17%**
of transactions in the dataset are fraudulent. This makes it a great showcase for:

- Handling severe class imbalance with **SMOTE + class weights**
- Tracking experiments and promoting models through **MLflow Model Registry**
- Serving predictions with a production-ready **FastAPI** container
- Catching data drift early with **Evidently AI**
- Automating the retraining loop with **Dagster**

**Model:** XGBoost classifier
**Key metrics:** ROC-AUC, Average Precision Score, F1, Precision, Recall

---

## 🛠 Tech Stack

| Tool | Layer | Why |
|---|---|---|
| **XGBoost** | ML Model | Industry standard for tabular fraud detection |
| **SMOTE** (imbalanced-learn) | Imbalance Handling | Oversamples the minority (fraud) class |
| **Pandera** | Data Validation | Schema-level contract — catches bad data before training |
| **MLflow** | Experiment Tracking & Registry | Logs every run; promotes models Staging → Production |
| **DVC** | Data Versioning | Tracks which data version produced which model |
| **FastAPI** | Model Serving | `/predict` and `/health` endpoints with auto Swagger docs |
| **Docker** | Environment Isolation | Reproducible container for the API |
| **Dagster** | Orchestration | DAG-based pipeline with scheduled retraining |
| **Evidently AI** | Monitoring | HTML drift reports comparing training vs live data |
| **GitHub Actions** | CI/CD | Lint + test on push; Docker build + push on merge to main |
| **uv** | Package Manager | Fast Python package and virtual environment management |

---

## 📁 Project Structure

```
credit_card_fraud_detection/
│
├── .github/workflows/
│   ├── ci_pipeline.yml         # Lint + test on every push
│   └── cd_pipeline.yml         # Build + push Docker image on merge to main
│
├── configs/
│   └── model_config.yaml       # All hyperparameters, paths, and settings
│
├── data/                       # Tracked by DVC — ignored by Git
│   ├── raw/                    # creditcard.csv goes here
│   └── processed/              # Cleaned data, test split, metrics.json
│
├── orchestration/
│   ├── __init__.py             # Dagster Definitions (assets + schedules)
│   ├── assets.py               # Dagster software-defined assets
│   └── schedules.py            # Weekly retraining + daily drift check
│
├── src/
│   ├── validate.py             # Pandera validation + data cleaning
│   ├── train.py                # SMOTE + XGBoost training + MLflow logging
│   └── evaluate.py             # Loads Production model, evaluates on test set
│
├── api/
│   ├── main.py                 # FastAPI app — /predict and /health
│   ├── schemas.py              # Pydantic request/response schemas
│   └── Dockerfile              # Multi-stage Docker build
│
├── monitoring/
│   ├── generate_report.py      # Evidently AI drift report generator
│   └── dashboards/             # Generated HTML reports (gitignored)
│
├── tests/
│   ├── test_api.py             # FastAPI endpoint tests (MLflow mocked)
│   └── test_model.py           # Pandera schema + XGBoost output tests
│
├── notebooks/
│   └── README.md               # EDA notebooks go here
│
├── dvc.yaml                    # DVC pipeline: validate → train → evaluate
├── pyproject.toml              # All Python dependencies
├── .\setup.ps1                   # Windows setup script — run once
├── .\run.ps1                     # Windows command runner (replaces Makefile)
├── .env.example                # Environment variable template
└── .gitignore
```

---

## 📦 Dataset

**Credit Card Fraud Detection** by the ULB Machine Learning Group
🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

| Property | Value |
|---|---|
| Transactions | 284,807 |
| Fraud cases | 492 (0.17%) |
| Features | V1–V28 (PCA), Time, Amount |
| Target | `Class` (0 = Legit, 1 = Fraud) |

Download `creditcard.csv` from Kaggle and place it at `data\raw\creditcard.csv`.

> **Note:** The `data\` folder is tracked by DVC and ignored by Git.
> The raw file will never be committed to your repository.

---

## 🔄 Pipeline Flow

```
Raw Data (creditcard.csv)
     │
     ▼
[DVC] — versions data in remote storage
     │
     ▼
[Pandera] — validates schema, types, value ranges
     │
     ▼  data\processed\creditcard_processed.csv
[SMOTE + XGBoost] — trains model, logs to MLflow
     │
     ▼  MLflow Model Registry (Staging → Production)
[FastAPI + Docker] — serves /predict from Production model
     │
     ▼  live prediction logs
[Evidently AI] — detects drift between training and live data
     │
     ▼  drift detected
[Dagster] — triggers retraining on schedule or drift signal
     └── loops back to top ↑
```

---

## ⚡ Quickstart

### 1. Install uv (if you don't have it)

Open PowerShell and run:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then close and reopen your terminal so `uv` is on your PATH.

### 2. Clone the repo

```bat
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 3. Run setup

```bat
.\setup.ps1
```

This creates a `.venv` and installs all dependencies.

### 4. Add the dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
and place it at:

```
data\raw\creditcard.csv
```

### 5. Start MLflow in a separate terminal

```bat
.\run.ps1 mlflow
```

MLflow UI will be at http://localhost:5000. Keep this terminal open.

---

## 🚀 Running the Full Pipeline

```bat
REM Step 1: Validate and clean the raw data
.\run.ps1 validate

REM Step 2: Train the model and log to MLflow
.\run.ps1 train

REM Step 3: Promote the model to Production
REM Go to http://localhost:5000 -> Models -> fraud-detector -> latest version
REM Click "Add alias" and type "Production", then save

REM Step 4: Evaluate the Production model
.\run.ps1 evaluate
```

Or run all stages at once with DVC:

```bat
.\run.ps1 dvc
```

---

## 🌐 API Usage

### Start the server

```bat
.\run.ps1 serve
```

API available at http://localhost:8000
Swagger docs at http://localhost:8000/docs

### Health check

```bat
curl http://localhost:8000/health
```

```json
{"status": "ok", "model_loaded": true}
```

### Predict a transaction

```bat
curl -X POST http://localhost:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"V1\": -1.36, \"V2\": -0.07, \"V3\": 2.54, \"V4\": 1.38, \"V5\": -0.34, \"V6\": 0.46, \"V7\": 0.24, \"V8\": 0.10, \"V9\": 0.36, \"V10\": 0.09, \"V11\": -0.55, \"V12\": -0.62, \"V13\": -0.99, \"V14\": -0.31, \"V15\": 1.47, \"V16\": -0.47, \"V17\": 0.21, \"V18\": 0.03, \"V19\": 0.40, \"V20\": 0.25, \"V21\": -0.02, \"V22\": 0.28, \"V23\": -0.11, \"V24\": 0.07, \"V25\": 0.13, \"V26\": -0.19, \"V27\": 0.13, \"V28\": -0.02, \"Amount\": 149.62}"
```

```json
{
  "is_fraud": false,
  "fraud_probability": 0.0312,
  "label": "Legit"
}
```

> **Tip:** It is easier to use the Swagger UI at http://localhost:8000/docs to test the API
> interactively rather than crafting curl commands on Windows.

---

## 🧪 Running Tests

```bat
.\run.ps1 test
```

- `tests\test_model.py` — Pandera schema checks and XGBoost output tests
- `tests\test_api.py` — FastAPI endpoint tests with MLflow fully mocked

---

## 🎛 Orchestration with Dagster

```bat
.\run.ps1 dagster
```

Open http://localhost:3000 to see the asset graph, run history, and schedules.

Schedules included:
- **Weekly retraining** — every Monday at 02:00 UTC
- **Daily drift check** — every day at 06:00 UTC

---

## 📊 Drift Monitoring

```bat
.\run.ps1 monitor
```

Generates an Evidently AI HTML report saved to `monitoring\dashboards\`.
Open the file in your browser to inspect which features have drifted.

---

## 🔁 CI/CD

| Event | Action |
|---|---|
| Push to any branch | Ruff lint + pytest via GitHub Actions |
| Merge to `main` | Docker image built and pushed to Docker Hub |

**Required GitHub Secrets:**
- `DOCKER_USERNAME` — your Docker Hub username
- `DOCKER_PASSWORD` — your Docker Hub access token

---

## 📄 License

MIT License — free to use, fork, and adapt for your own portfolio or projects.
