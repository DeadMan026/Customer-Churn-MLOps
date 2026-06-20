# Telco Customer Churn - End-to-End ML Project

## Overview

Predicting customer churn for a telecom provider using the IBM Telco Customer Churn dataset. This project implements a full ML pipeline from raw data ingestion to validation, feature engineering, model training, experiment tracking, and initial serving preparation.

---

## Current Status

- [x] Project structure set up
- [x] Virtual environment and pinned dependencies configured
- [x] Exploratory Data Analysis (EDA)
- [x] Data validation with Great Expectations
- [x] Automated preprocessing pipeline
- [x] Feature engineering pipeline
- [x] Model training with XGBoost
- [x] MLflow experiment and artifact tracking
- [x] Initial serving inference module added
- [ ] FastAPI app wiring
- [ ] Gradio interface for local interaction
- [ ] Containerization (Docker)
- [ ] CI/CD pipeline
- [ ] Local API packaging finalization

---

## Modular Pipeline

### 1. Data Validation (`src/utils`)
- **`validate_data.py`**: Uses Great Expectations to enforce schema integrity, numeric ranges, and business logic checks.

### 2. Data Processing (`src/data`)
- **`load_data.py`**: Loads raw CSV data with basic file validation and type correction.
- **`preprocess.py`**: Cleans columns, drops ID fields, converts numeric values, maps the target, and handles missing numeric values.

### 3. Feature Engineering (`src/features`)
- **`build_features.py`**: Applies deterministic binary encoding and one-hot encoding to create model-ready features.

### 4. Model Training (`src/models`)
- **`train.py`**: Trains an XGBoost classifier and logs parameters, metrics, and model artifacts to MLflow.
- **`tune.py`**: Uses Optuna for hyperparameter tuning.
- **`evaluate.py`**: Prints model evaluation outputs such as classification report and confusion matrix.

### 5. Model Serving (`src/serving`)
- **`inference.py`**: Loads the trained MLflow model, reapplies serving-time feature transformations, aligns feature columns with training, and returns churn predictions.

### 6. App Layer (`src/app`)
- **`main.py`** and **`app.py`**: Initial scaffolding for the serving interface layer.
- Planned next step: add FastAPI endpoint wiring and a Gradio interface for local interactive use.

---

## Pipeline Testing

The project includes standalone scripts in the `scripts/` directory to verify that each stage of the pipeline behaves correctly.

### Phase 1: Data and Features
- **`scripts/test_pipeline_01_data_features.py`**: Validates the flow from raw data loading through preprocessing to feature engineering.

```powershell
python scripts/test_pipeline_01_data_features.py
```

### Phase 2: Modeling
- **`scripts/test_pipeline_02_modeling.py`**: Tests the modeling phase, including hyperparameter tuning with Optuna and XGBoost classification.

```powershell
python scripts/test_pipeline_02_modeling.py
```

### Full Pipeline Orchestration
- **`scripts/run_pipeline.py`**: Runs the complete end-to-end pipeline from loading to evaluation with MLflow tracking.

```powershell
python scripts/run_pipeline.py --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

## Project Structure

```text
app/                    # Reserved top-level app area
data/
  raw/                  # Raw data storage (ignored by git)
  processed/            # Intermediate processed outputs
notebooks/
  EDA.ipynb             # Exploratory Data Analysis
src/
  app/                  # FastAPI / Gradio app scaffolding
  data/                 # Data loading and preprocessing scripts
  features/             # Feature engineering logic
  models/               # Training, tuning, and evaluation scripts
  serving/              # Inference-time model serving logic
  utils/                # Validation and utility functions
scripts/                # Pipeline execution and test scripts
README.md               # Project documentation
requirements.txt        # Pinned dependencies
```

---

## Setup

1. Clone the repository.
2. Create and activate a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset.
4. Place the CSV file in `data/raw/` as `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

---

## Scope Note

This project is intended to stay simple and local-first. Paid cloud deployment steps are intentionally out of scope.

The next interface milestone is to expose the model through a local API layer and add a Gradio UI for interactive predictions.
