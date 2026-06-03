import pandas as pd

def preprocess_data(df : pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Basic cleaning for Telco churn.
    - trim column names
    - drop obvious ID cols
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
    df.columns = df.columns.str.strip()   # Remove leading/trailing whitespace

    # drop ids if present  (future-proofing with multiple checks)
    for col in ['customer_id', 'customerID', 'CustomerID']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # TotalCharges often have blackes in this dataset
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen should be 0/1 int if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0)

    # Mapping target Churn to 0/1
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].str.strip().map({"No" : 0, "Yes" : 1})

    
    # simple NA strategy:
    # - numeric: fill with 0
    # - others: leave for encoders to handle (get_dummies ignores NaN safely)
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df