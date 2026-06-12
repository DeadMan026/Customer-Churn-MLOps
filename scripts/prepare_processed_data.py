import os, sys
import pandas as pd

# make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/Telco-Customer-Churn.csv"
OUT = "data/processed/telco_churn_processed.csv"