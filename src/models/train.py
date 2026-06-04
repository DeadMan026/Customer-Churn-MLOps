import mlflow
import pandas as pd
import mlflow.xgboost as mlflow_xgboost
from mlflow.data.pandas_dataset import from_pandas
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def train_model(df: pd.DataFrame, target_col: str):
    """
    Trains an XGBoost model and logs with MLflow.

    Args:
        df (pd.DataFrame): Feature dataset.
        target_col (str): Name of the target column.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    # Initialize the model with your best parameters
    model = XGBClassifier(
        n_estimators=376,
        learning_rate=0.11552620892008209,
        max_depth=6,
        subsample=0.996637854862527,
        colsample_bytree=0.9862899696923655,
        min_child_weight=10,
        gamma=4.895479894313713,
        reg_alpha=3.7611319076202037,
        reg_lambda=0.1326009596944906,
        random_state=42,  # Added for reproducibility
        eval_metric='logloss' 
    )

    with mlflow.start_run():
        # train model
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        rec = float(recall_score(y_test, preds))

        # Log parameters, metrics and model
        mlflow.log_param("n_estimators",376)
        mlflow.log_metric("accuracy",acc)
        mlflow.log_metric("recall",rec)
        mlflow_xgboost.log_model(model,"model")


        # 🔑 Log dataset so it shows in MLflow UI
        train_ds = from_pandas1(df, name="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")


