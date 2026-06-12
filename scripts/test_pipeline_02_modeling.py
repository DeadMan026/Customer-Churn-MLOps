import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import optuna

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Path to processed data
PROCESSED_DATA_PATH = os.path.join("data", "processed", "telco_churn_processed.csv")

def main():
    print("=== Phase 2: Modeling with XGBoost ===")

    if not os.path.exists(PROCESSED_DATA_PATH):
        print(f"❌ Error: Processed data not found at {PROCESSED_DATA_PATH}")
        print("Please run the Phase 1 test or data preparation scripts first.")
        return

    df = pd.read_csv(PROCESSED_DATA_PATH)

    # target must be numeric 0/1
    if df["Churn"].dtype == "object":
        print("Mapping target column 'Churn' to numeric values...")
        df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1})

    assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
    assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1"

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    THRESHOLD = 0.4

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "eval_metric": "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= THRESHOLD).astype(int)
        from sklearn.metrics import recall_score
        return float(recall_score(y_test, y_pred, pos_label=1))

    print("\nStarting hyperparameter optimization with Optuna...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    
    print("\n✅ Phase 2 Completed!")
    print(f"Best Params: {study.best_params}")
    print(f"Best Recall Score: {study.best_value:.4f}")

if __name__ == "__main__":
    main()