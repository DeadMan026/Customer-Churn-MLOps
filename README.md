# Telco Customer Churn – End-to-End ML Project

## Overview

Predicting customer churn for a telecom provider using the IBM Telco Customer Churn dataset. The goal is to build a full ML pipeline — from raw data to a deployed, accessible inference service.

---

## Current Status

- [x] Project structure set up
- [x] Virtual environment + dependencies configured
- [x] Exploratory Data Analysis (EDA)
  - Dataset overview and `.describe()` analysis
  - Identified binary vs multi-category variables
  - Label encoding for binary columns
  - One-Hot Encoding for multi-category columns
  - Collapsed redundant dummy columns (`No internet service`, `No phone service`)
  - Correlation analysis with churn target
  - VIF (Variance Inflation Factor) analysis to detect multicollinearity
- [ ] Feature engineering
- [ ] Model training & evaluation
- [ ] MLflow experiment tracking
- [ ] API development
- [ ] Containerization
- [ ] CI/CD pipeline
- [ ] Cloud deployment

---

## Planned Pipeline

### Modeling
- Train an XGBoost classifier on the engineered features
- Track all experiments, metrics, and artifacts with **MLflow**
- Evaluate using AUC-ROC, precision, recall, F1

### Inference Service
- **FastAPI** app exposing a `POST /predict` endpoint
- Health check at `GET /`
- **Gradio** UI mounted at `/ui` for quick manual testing

### Containerization
- Dockerized app with `uvicorn` as the entrypoint
- Image built via **GitHub Actions** and pushed to Docker Hub

### CI/CD
- GitHub Actions workflow on push to `main`:
  - Build Docker image
  - Push to Docker Hub
  - Optionally trigger ECS redeployment

### Cloud Deployment (AWS)
- **ECS Fargate** to run the container (serverless)
- **Application Load Balancer (ALB)** on port 80 forwarding to container on port 8000
- **CloudWatch Logs** for observability
- Security groups scoped: ALB allows inbound 80 from internet; task allows inbound 8000 from ALB SG only

> Note: The plan above reflects current intent. Implementation details may evolve as the project progresses.

---

## Project Structure

```
├── data/
│   ├── raw/            # Original dataset (not committed)
│   ├── processed/      # Cleaned & transformed data
│   └── external/       # Any third-party data
├── notebooks/          # EDA and exploration notebooks
├── src/
│   ├── data/           # Data loading & preprocessing scripts
│   ├── features/       # Feature engineering
│   ├── models/         # Training & evaluation
│   └── utils/          # Shared utilities
├── app/                # FastAPI + Gradio app
├── configs/            # Config files (hyperparameters, paths)
├── scripts/            # One-off utility scripts
├── tests/              # Unit and integration tests
├── .github/workflows/  # CI/CD pipelines
├── docker/             # Dockerfile and related configs
├── great_expectations/ # Data validation
├── mlruns/             # MLflow tracking (not committed)
└── artifacts/          # Saved models (not committed)
```

---

## Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) — 7,043 customers, 21 features including demographics, services subscribed, account info, and a `Churn` target column.

---

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
uv pip install -r requirements.txt
```
