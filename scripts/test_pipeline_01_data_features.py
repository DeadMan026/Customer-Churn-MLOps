import os
import sys
import pandas as pd

# Add the project root to sys.path to allow imports from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# Use relative path for data - matching the README setup instructions
DATA_FILE = os.path.join("data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
TARGET_COL = "Churn"

def main():
    print("=== Testing Phase 1: Load → Preprocess → Build Features ===")

    # 1. Load Data
    print(f"\n[1] Loading data from {DATA_FILE}...")
    try:
        df = load_data(DATA_FILE)
        print(f"Data loaded. Shape: {df.shape}")
        print(df.head(3))
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the dataset is placed in 'data/raw/' as per README instructions.")
        return

    # 2. Preprocess
    print("\n[2] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing. Shape: {df_clean.shape}")
    print(df_clean.head(3))

    # 3. Build Features
    print("\n[3] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering. Shape: {df_features.shape}")
    print(df_features.head(3))

    print("\n✅ Phase 1 pipeline completed successfully!")

if __name__ == "__main__":
    main()