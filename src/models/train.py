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
        n_estimators=381,
        learning_rate=0.06146955484874729,
        max_depth=4,
        subsample=0.9999156440233287,
        colsample_bytree=0.9509340024134957,
        min_child_weight=4,
        gamma=4.061122396595465,
        reg_alpha=4.9843016642256135,
        reg_lambda=2.7784428587760575,
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
        train_ds = from_pandas(df, name="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")


