# Telco Customer Churn – End-to-End ML Project

## Overview

Predicting customer churn for a telecom provider using the IBM Telco Customer Churn dataset. This project implements a full ML pipeline — from raw data to experiment tracking and validation.

---

## Current Status

- [x] Project structure set up
- [x] Virtual environment + dependencies configured (Pinned versions)
- [x] Exploratory Data Analysis (EDA) with portable path handling (`pathlib`)
- [x] Data Validation Suite (Great Expectations)
- [x] Automated Preprocessing Pipeline
- [x] Feature Engineering (Binary & One-Hot Encoding)
- [x] Model training (XGBoost)
- [ ] MLflow experiment & artifact tracking
- [ ] API development (FastAPI)
- [ ] Containerization (Docker)
- [ ] CI/CD pipeline
- [ ] Cloud deployment

---

## Modular Pipeline

### 1. Data Validation (`src/utils`)
- **`validate_data.py`**: Uses **Great Expectations** to enforce schema integrity, numeric ranges, and business logic (e.g., `TotalCharges >= MonthlyCharges`).

### 2. Data Processing (`src/data`)
- **`load_data.py`**: Robust CSV loading utility with error handling.
- **`preprocess.py`**: Automated cleaning pipeline including whitespace stripping, ID removal, and numerical conversion for inconsistent fields.

### 3. Feature Engineering (`src/features`)
- **`build_features.py`**: Automated transformation pipeline.
  - Deterministic binary mapping for demographic and service categories.
  - Multi-category one-hot encoding with `drop_first=True`.
  - Type-safe conversions for XGBoost compatibility.

### 4. Model Training (`src/models`)
- **`train.py`**: XGBoost classifier implementation with integrated **MLflow** tracking.
  - Logs hyper-parameters, accuracy, and recall metrics.
  - Automatically logs model artifacts and input datasets for full lineage.

---

## Pipeline Testing

The project includes standalone scripts in the `scripts/` directory to verify the integrity of the pipeline stages. These scripts are useful for debugging and ensuring that changes to the core logic don't break the end-to-end flow.

### Phase 1: Data & Features
- **`scripts/test_pipeline_01_data_features.py`**: Validates the flow from raw data loading through preprocessing to feature engineering.
  ```powershell
  python scripts/test_pipeline_01_data_features.py
  ```

### Phase 2: Modeling
- **`scripts/test_pipeline_02_modeling.py`**: Tests the modeling phase, including hyperparameter tuning with **Optuna** and XGBoost classification.
  ```powershell
  python scripts/test_pipeline_02_modeling.py
  ```

---

## Project Structure

```
├── app/                # Future FastAPI implementation
├── data/
│   ├── raw/            # Raw data storage (ignored by git)
│   └── processed/      # Intermediate processed data
├── notebooks/
│   └── EDA.ipynb       # Exploratory Data Analysis (using pathlib)
├── src/
│   ├── data/           # Data loading and preprocessing scripts
│   ├── features/       # Feature engineering logic
│   ├── models/         # Training scripts
│   └── utils/          # Validation and utility functions
├── .gitignore          # Git ignore rules
├── README.md           # Project documentation
└── requirements.txt    # Pinned dependencies
```

---

## Setup

1. **Clone the repository**
2. **Environment Setup:**
   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Data Placement:**
   - Download the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset.
   - Place the CSV file in `data/raw/` and name it `WA_Fn-UseC_-Telco-Customer-Churn.csv`.

4. **Running EDA:**
   The notebook in `notebooks/EDA.ipynb` uses `pathlib` for root-centric path resolution, making it portable across environments.
