"""
preprocess_sivas.py — Preprocessing script for Sivas Transit Dataset
=====================================================================
Modular script to join stop arrivals, trip metadata, and stop coordinates
for the Sivas dataset. Fits perfectly in RAM.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure ai-engine root (two levels up) is importable
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET, CATEGORICAL_FEATURES, X_TRAIN_PARQUET, X_TEST_PARQUET

# Logger Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_sivas")

def main():
    logger.info("Starting Sivas data preprocessing...")

    # 1. Load Data
    try:
        arrivals = pd.read_csv(RAW_DATA_DIR / "stop_arrivals.csv")
        stops = pd.read_csv(RAW_DATA_DIR / "Bus_Stops.csv", usecols=["stop_id", "distance_from_prev_km"])
    except FileNotFoundError as e:
        logger.error(f"Missing raw CSV files: {e}")
        return

    # 2. Convert Timestamps
    arrivals["arrival_time"] = pd.to_datetime(arrivals["actual_arrival"], errors="coerce")
    arrivals.dropna(subset=["arrival_time"], inplace=True)
    
    # 3. Merges
    logger.info("Merging datasets...")
    # Join with stops to get distance
    df = arrivals.merge(stops, on="stop_id", how="left")

    # 4. Feature Selection & Cleanup
    features = [
        "cumulative_delay_min", "distance_from_prev_km", 
        "traffic_level", "weather_condition", "line_id", "stop_id"
    ]
    
    # Check if target exists
    if TARGET not in df.columns:
        logger.error(f"Target column '{TARGET}' not found in merged dataframe.")
        return

    # Fill missing distances just in case
    df["distance_from_prev_km"] = df["distance_from_prev_km"].fillna(0.0)

    # Select only the features we need
    df = df[features + [TARGET] + ["arrival_time"]].copy()
    df.dropna(inplace=True)

    # Convert Categorical types as defined in config
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Ensure continuous features are correctly typed
    continuous = ["cumulative_delay_min", "distance_from_prev_km"]
    for col in continuous:
        df[col] = df[col].astype("float32")
    df[TARGET] = df[TARGET].astype("float32")

    # 5. Temporal Split
    # Train: March 1-23, 2025 | Test: March 24-30, 2025
    train_end = pd.Timestamp("2025-03-24")
    
    train_mask = df["arrival_time"] < train_end
    test_mask = (df["arrival_time"] >= train_end) & (df["arrival_time"] < pd.Timestamp("2025-03-31"))
    
    train_df = df[train_mask].drop(columns=["arrival_time"])
    test_df = df[test_mask].drop(columns=["arrival_time"])

    # 6. Save as Parquet
    logger.info(f"Saving splits to {PROCESSED_DATA_DIR}...")
    train_df.to_parquet(X_TRAIN_PARQUET, index=False)
    test_df.to_parquet(X_TEST_PARQUET, index=False)

    logger.info(f"Sivas Preprocessing Complete! Train: {len(train_df)} rows, Test: {len(test_df)} rows.")

if __name__ == "__main__":
    main()
