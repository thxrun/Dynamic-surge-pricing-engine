# Cloud-Agnostic End-to-End Dynamic Surge Pricing MLOps Engine

This repository contains a production-grade, cloud-agnostic MLOps pipeline for a **Dynamic Surge Pricing Engine**. The system predicts optimal real-time price multipliers based on regional rider demand, traffic density, weather conditions, and driver availability.

Rather than relying on proprietary cloud suites (like AWS SageMaker), this architecture is engineered purely with open-source, decoupled tools (**DVC, MLflow, FastAPI, Docker, GitHub Actions, and Evidently AI**). This ensures the entire stack can be deployed seamlessly across any infrastructure—whether on a local machine, on-premises servers, or cloud environments like AWS, GCP, or Azure.

---

## 🏗️ System Architecture & Data Flow

```
                  +-------------------------------------------------+
                  |              PHASE 1: DATA LINEAGE              |
                  |  Local CSVs ---> [DVC] ---> Cloud Object Store  |
                  +------------------------+------------------------+
                                           |
                                           v
                  +-------------------------------------------------+
                  |                PHASE 3: CI GATE                |
                  |    Git Push ---> [GitHub Actions] ---> Pytest    |
                  +------------------------+------------------------+
                                           | (Passes Performance Gate)
                                           v
                  +-------------------------------------------------+
                  |           PHASE 2: EXPERIMENT TRACKING          |
                  |  Compute Box ---> XGBoost ---> [MLflow Tracking]|
                  +------------------------+------------------------+
                                           | (Promoted to Prod Tag)
                                           v
                  +-------------------------------------------------+
                  |             PHASE 4: SERVING & DRIFT           |
                  |  [Docker / FastAPI] <--- Pulls Registry Model   |
                  |                         |                       |
                  |                         v                       |
                  |         Req/Resp Logs ---> [Evidently AI]       |
                  +-------------------------------------------------+

```

---

## 📁 Repository Structure

```text
├── .dvc/                        # DVC pipeline configurations & internal tracking
├── .github/workflows/           
│   └── ci-cd.yaml               # GitHub Actions CI/CD automation pipeline
├── data/                        
│   ├── raw_features.csv.dvc     # DVC pointer tracking raw file version
│   └── processed/               # Data artifacts folder (ignored by Git)
├── deployment/                  
│   ├── Dockerfile               # High-efficiency Python inference container
│   └── docker-compose.yaml      # Multi-container orchestration (App + Logging)
├── metrics/                     
│   └── data_drift_report.html   # Periodically generated Evidently AI profiles
├── src/                         
│   ├── __init__.py
│   ├── api.py                   # FastAPI application layer with MLflow integration
│   ├── config.py                # Central project configurations and paths
│   ├── preprocess.py            # Feature transformation and cleaning pipelines
│   └── train.py                 # Core model training execution & MLflow log loop
├── tests/                       
│   └── test_api.py              # Endpoint validation & request contract suites
├── requirements.txt             # Python absolute dependencies
└── dvc.yaml                     # Reproducible data stage definition file

```

---

## 🛠️ Technology Tool Stack

* **Data Lineage & Versioning:** Data Version Control (DVC) paired with local storage or MinIO / Google Cloud Storage.
* **Experiment Management & Registry:** MLflow tracking server.
* **Automation Loop:** GitHub Actions CI/CD runner.
* **Microservice Layer:** FastAPI + Uvicorn.
* **Containerization:** Docker & Docker Compose.
* **Production Telemetry:** Evidently AI.

---

## 🚀 Step-by-Step Implementation Lifecycle

### Phase 1: Local Data Architecture & Lineage

**Objective:** Decouple data tracking from Git source control to maintain a clean codebase while strictly enforcing data reproducibility.

```bash
# 1. Initialize git and DVC architectures
git init
dvc init

# 2. Configure a remote storage cache (Local directory mimicry or MinIO bucket)
dvc remote add -d local_remote /tmp/dvc_storage

# 3. Securely register large scale raw pricing training files
dvc add data/raw_features.csv

# 4. Commit pointer files to version control instead of bulk binaries
git add data/raw_features.csv.dvc .gitignore
git commit -m "chore: track base pricing data version via dvc"

```

The underlying pipeline processing steps are mapped cleanly inside the `dvc.yaml` tracking graph to allow automatic step caching:

```yaml
stages:
  preprocess:
    cmd: python src/preprocess.py
    deps:
      - data/raw_features.csv
      - src/preprocess.py
    outs:
      - data/processed/features.parquet

```

---

### Phase 2: Sandbox Experimentation & MLflow Tracking

**Objective:** Move away from local unversioned development models and catalog all parameters, artifacts, and training curves into an explicit visualization engine.

```bash
# Spin up your isolated MLflow server instance locally or on a central node
mlflow server --host 127.0.0.1 --port 5000

```

The `src/train.py` module handles model parameter binding and tracks the training runs programmatically inside MLflow:

```python
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("surge-pricing-engine")

with mlflow.start_run():
    params = {"max_depth": 6, "learning_rate": 0.1, "objective": "reg:squarederror"}
    mlflow.log_params(params)
    
    # Train regressor against processed parquet data
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    mlflow.log_metric("mae", mae)
    
    # Log and push directly to the central MLflow Model Registry
    mlflow.xgboost.log_model(
        model, 
        artifact_path="model", 
        registered_model_name="SurgePricingEngine"
    )

```

---

### Phase 3: Quality Gates & Continuous Integration (CI)

**Objective:** Automate formatting checks, run functional test coverage, and prevent regression updates through automated metrics gates.

The `.github/workflows/ci-cd.yaml` script handles testing automation:

```yaml
name: MLOps Production Engine Gate

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  evaluate-pipeline:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout Codebase
      uses: actions/checkout@v3

    - name: Set up Python Runtime
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install System Dependencies
      run: |
        pip install -r requirements.txt

    - name: Run Schema Functional Tests
      run: |
        pytest tests/

    - name: Execute MLflow Metric Champion Validation Gate
      env:
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
      run: |
        # Custom validation execution script logic
        # Compares incoming pull request MAE against current production model MAE.
        # If PR model MAE is less than production baseline, it auto-promotes tag to 'Production'.
        python src/evaluate_and_promote.py

```

---

### Phase 4: Serving & Drift Observation (CD)

**Objective:** Containerize your application to ensure consistent behavior across development and production environments, and catch model performance degradation before it impacts your business.

The `src/api.py` microservice decouples the API logic from model files by dynamically pulling down whichever model version holds the active `Production` tag inside the registry when booting:

```python
from fastapi import FastAPI
import mlflow.pyfunc
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Dynamic Surge Pricing Microservice")

# Global placeholder for hot-swapping model bytes smoothly
model = None

class PredictionPayload(BaseModel):
    rider_demand_score: float
    driver_availability_ratio: float
    precipitation_mm: float
    traffic_density_index: float

@app.on_event("startup")
def load_production_model():
    global model
    # Target the central model repository instead of local static files
    model_uri = "models:/SurgePricingEngine/Production"
    model = mlflow.pyfunc.load_model(model_uri)

@app.post("/predict")
def predict_surge_multiplier(payload: PredictionPayload):
    input_df = pd.DataFrame([payload.dict()])
    prediction = model.predict(input_df)
    return {"surge_multiplier": float(prediction[0])}

```

The app configuration is isolated within a lightweight production `deployment/Dockerfile`:

```dockerfile
FROM python:3.10-slim
WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]

```

#### Monitoring Real-World Data Drift

Over time, shifts in underlying market behaviors (e.g., changes in post-pandemic commuter traffic or extreme weather anomalies) will degrade the accuracy of your static pricing model.

The production application routes transaction payloads into a localized text log stream. A distinct diagnostic cron tab parses these log streams daily using **Evidently AI** to compute data drift profiles relative to the baseline training distributions:

```python
from conclusions import report
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Extract a reference training snapshot and contrast against production traffic data frames
drift_report = Report(metrics=[DataDriftPreset()])
drift_report.run(reference_data=reference_df, current_data=production_logs_df)

# Store visual dashboard diagnostics to root metrics folder
drift_report.save_html("metrics/data_drift_report.html")

```

---

## ⚡ Quickstart Execution Guide

To run this entire environment locally end-to-end, execute the following steps sequence:

1. **Clone and Install:**
```bash
git clone https://github.com/yourusername/dynamic-surge-pricing-mlops.git
cd dynamic-surge-pricing-mlops
pip install -r requirements.txt

```


2. **Pull Tracked Data Components:**
```bash
dvc pull

```


3. **Spin up Local Trackers and Infrastructure Engines:**

```bash
   docker compose -f deployment/docker-compose.yaml up --build

```

4. **Trigger local training run execution cycles:**

```bash
   python src/train.py

```

5. **Verify active running microservice parameters:**

```bash
   curl -X 'POST' \
     'http://localhost:8000/predict' \
     -H 'accept: application/json' \
     -H 'Content-Type: application/json' \
     -d '{
     "rider_demand_score": 0.85,
     "driver_availability_ratio": 0.31,
     "precipitation_mm": 12.4,
     "traffic_density_index": 0.91
   }'

```

```

```