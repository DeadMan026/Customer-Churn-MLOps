import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Fix common type inference issue for TotalCharges in Telco dataset
    if "TotalCharges" in df.columns:
        # Convert to numeric, turning empty strings/spaces into NaN
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        
    return df
