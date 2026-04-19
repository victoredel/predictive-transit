"""
train_model.py — XGBoost Training from pre-processed Parquet
===================================================================
Modular training script that adapts to different datasets based on config.py.
Handles both directory-based partitions (Boston) and single-file Parquets (Sivas).

Usage:
  python train_model.py
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Ensure ai-engine root (two levels up) is importable
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import dynamic configuration
from config import X_TRAIN_PARQUET as PARQUET_IN, MODEL_PATH, TARGET, CATEGORICAL_FEATURES, ACTIVE_DATASET

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_model")

XGB_PARAMS = {
    'max_depth': 12,
    'learning_rate': 0.1440,
    'subsample': 0.9724,
    'colsample_bytree': 0.9814,
    'min_child_weight': 15,
    'n_estimators': 800,
    'early_stopping_rounds': 40,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'enable_categorical': True,
    'random_state': 42,
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'device': 'cpu',
}

def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def main() -> None:
    t_start = time.time()
    logger.info("╔═══════════════════════════════════════════════════════════════════╗")
    logger.info(f"║  MBTA XGBoost — Training: {ACTIVE_DATASET.upper():<31} ║")
    logger.info("╚═══════════════════════════════════════════════════════════════════╝")
    logger.info("  🧠 Initial RAM: %.0f MB", ram_mb())

    # 1. Load data
    if not PARQUET_IN.exists():
        logger.error("❌ Input %s not found. Run preprocessing first.", PARQUET_IN)
        sys.exit(1)

    logger.info("  Loading data...")
    
    if PARQUET_IN.is_dir():
        # Directory-based loading (Boston)
        parquet_files = list(PARQUET_IN.glob("*.parquet"))
        if not parquet_files:
            logger.error("❌ No .parquet files found inside %s", PARQUET_IN)
            sys.exit(1)

        dfs = []
        for f in parquet_files:
            logger.info("    Reading and sampling %s...", f.name)
            chunk = pd.read_parquet(f)
            # Use lower fraction or no sampling for Sivas-size if needed, 
            # but Boston logic was 30%
            if ACTIVE_DATASET == "boston":
                chunk = chunk.sample(frac=0.30, random_state=42)
            dfs.append(chunk)
            del chunk
            gc.collect()
        df = pd.concat(dfs, ignore_index=True)
        del dfs
    else:
        # Single file loading (Sivas)
        df = pd.read_parquet(PARQUET_IN)

    # Ensure memory-efficient types
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
            
    # Ensure continuous features are strictly float32
    continuous_cols = df.select_dtypes(include=['float64', 'float16', 'int64', 'int32']).columns
    for col in continuous_cols:
        if col != TARGET:
            df[col] = df[col].astype(np.float32)

    logger.info("  ✅ Resulting Dataset: %d rows | RAM: %.0f MB", len(df), ram_mb())

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    del df; gc.collect()

    # 2. Split
    logger.info("  Partitioning data (95/5)...")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.05, random_state=42)
    logger.info("  ✅ Train: %d | Val: %d", len(X_tr), len(X_val))

    # 3. Training
    logger.info("  🚀 Training XGBoost...")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    
    t0 = time.time()
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=50,
    )
    logger.info("  ✅ Trained in %.1fs", time.time() - t0)

    # 4. Fast Evaluation
    preds = model.predict(X_val)
    mae = float(np.mean(np.abs(preds - y_val)))
    logger.info("  📊 VAL MAE = %.2f (%.1f min if seconds)", mae, mae/60)

    # 5. Saving
    model.save_model(MODEL_PATH)
    logger.info("  ✅ Model saved to %s (%.1f MB)", MODEL_PATH.name, MODEL_PATH.stat().st_size / 1e6)
    
    logger.info("🎉 Completed in %.1f min", (time.time() - t_start) / 60)

if __name__ == "__main__":
    main()
