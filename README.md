# Telco Customer Churn – End-to-End ML Project

## Overview

Predicting customer churn for a telecom provider using the IBM Telco Customer Churn dataset. The goal is to build a full ML pipeline — from raw data to a deployed, accessible inference service.

---

## Current Status

- [x] Project structure set up
- [x] Virtual environment + dependencies configured
- [x] Exploratory Data Analysis (EDA)
- [x] Data Processing & Feature Engineering
  - Basic cleaning: ID removal, data type fixing (`TotalCharges`), target mapping
  - Feature extraction: Binary encoding for 2-category features
  - One-hot encoding for multi-category features with multicollinearity handling
- [ ] Model training & evaluation
- [ ] MLflow experiment tracking
- [ ] API development
- [ ] Containerization
- [ ] CI/CD pipeline
- [ ] Cloud deployment

---

## Modular Pipeline

### 1. Data Processing (`src/data`)
- **`load_data.py`**: Robust CSV loading utility.
- **`preprocess.py`**: Automated cleaning pipeline including whitespace stripping, ID column removal, and numerical conversion for inconsistent fields.

### 2. Feature Engineering (`src/features`)
- **`build_features.py`**: Automated feature transformation pipeline.
  - Deterministic binary mapping for demographic and service categories.
  - Multi-category one-hot encoding with `drop_first=True`.
  - Type-safe conversions for model compatibility (XGBoost).

### 3. Planned: Modeling
- Train an XGBoost classifier on the engineered features
- Track all experiments, metrics, and artifacts with **MLflow**
- Evaluate using AUC-ROC, precision, recall, F1

---

## Project Structure

```
├── app/                # Future FastAPI implementation
├── data/               # Local data storage (ignored by git)
├── notebooks/
│   └── EDA.ipynb       # Exploratory Data Analysis
├── src/
│   ├── data/           # Data loading and preprocessing scripts
│   └── features/       # Feature engineering and transformation logic
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── requirements.txt    # Project dependencies
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
