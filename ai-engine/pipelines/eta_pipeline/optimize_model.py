"""
optimize_model.py — Hyperparameter Search with Optuna
==========================================================
Uses an ultra-reduced sample (5%) of the partitioned training dataset
to quickly find the best error metric (RMSE) using Bayesian techniques.

Usage:
  python optimize_model.py
"""

import gc
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("optimize_model")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import X_TRAIN_PARQUET, TARGET, CATEGORICAL_FEATURES

PARQUET_IN = X_TRAIN_PARQUET

# Global variables for data (to avoid copying them within the Optuna loop)
X_th = None
y_th = None

# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------
def load_and_prepare_data() -> tuple:
    global X_th, y_th

    logger.info("=" * 70)
    logger.info("STEP 1: CHUNKED LOADING AND EXTREME DOWNSAMPLING (5%)")
    logger.info("=" * 70)

    if not PARQUET_IN.exists():
        logger.error("❌ An error occurred. Partitioned directory not found: %s", PARQUET_IN)
        sys.exit(1)

    parquet_files = list(PARQUET_IN.glob("*.parquet"))
    dfs = []
    
    for f in parquet_files:
        logger.info("    Reading and distilling %s to 5%%...", f.name)
        try:
            chunk = pd.read_parquet(f)
            chunk = chunk.sample(frac=0.05, random_state=42)
            dfs.append(chunk)
            del chunk
        except Exception as e:
            logger.warning("Could not read partition %s: %s", f.name, e)
        gc.collect()

    logger.info("  Concatenating %d reduced partitions...", len(dfs))
    df = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()

    # Data Types (Anti-OOM)
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    continuous_cols = df.select_dtypes(include=['float64', 'float16', 'int64', 'int32']).columns
    for col in continuous_cols:
        df[col] = df[col].astype(np.float32)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    del df; gc.collect()
    
    X_th, y_th = X, y    
    return X, y

# ---------------------------------------------------------------------------
# Optuna Objective Function
# ---------------------------------------------------------------------------
def objective(trial: optuna.Trial) -> float:
    # Get the previously loaded global dataset
    global X_th, y_th
    
    # Intelligent Hyperparameter Grid
    param = {
        "max_depth":             trial.suggest_int("max_depth", 5, 12),
        "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample":             trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight":      trial.suggest_int("min_child_weight", 1, 20),
        
        # Fixed for speed
        "n_estimators":          300,
        "early_stopping_rounds": 20,
        "objective":             "reg:squarederror",
        "tree_method":           "hist",
        "enable_categorical":    True,
        "random_state":          42,
        "eval_metric":           "rmse",
        "n_jobs":                -1,
        "device":                "cpu",
    }

    # Volatile Intelligent Split: 80% Train, 20% Valid
    X_tr, X_val, y_tr, y_val = train_test_split(X_th, y_th, test_size=0.20, random_state=42)
    
    # Optuna pruning callback to speed up futile trials
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, "validation_1-rmse")

    model = xgb.XGBRegressor(**param, callbacks=[pruning_callback])
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        verbose=False
    )

    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    # Cleanup for next round
    del X_tr, X_val, y_tr, y_val
    gc.collect()

    return float(rmse)

# ---------------------------------------------------------------------------
# Main Routine
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    logger.info("╔═══════════════════════════════════════════════════════════════════╗")
    logger.info("║       OPTUNA — Tuning XGBoost Hyperparameters (5%% Data)          ║")
    logger.info("╚═══════════════════════════════════════════════════════════════════╝")

    load_and_prepare_data()

    logger.info("=" * 70)
    logger.info("STEP 2: OPTUNA BAYESIAN EXECUTION (30 Trials)")
    logger.info("=" * 70)
    
    study = optuna.create_study(direction="minimize", study_name="mbta_xgboost")
    
    try:
        study.optimize(objective, n_trials=30, gc_after_trial=True)
    except KeyboardInterrupt:
        logger.warning("\n⏹️ Search interrupted or cancelled. Showing best results so far...")
        
    elapsed = time.time() - t0

    print("\n" + "═"*70)
    print(f"🎉 SEARCH FINISHED (Time: {elapsed/60:.1f} min)")
    print("═"*70)
    
    if len(study.trials) > 0 and study.best_trial is not None:
        best_trial = study.best_trial
        print(f"🏅 Best RMSE reached in validation: {best_trial.value:.3f} seconds")
        print("\n⚙️  BEST PARAMETERS (Copy-paste ready for train_model.py):")
        print("XGB_PARAMS = {")
        for key, value in best_trial.params.items():
            if isinstance(value, float):
                print(f"    '{key}': {value:.4f},")
            else:
                print(f"    '{key}': {value},")
        
        # Show other fixed parameters
        print("    'n_estimators': 800,")
        print("    'early_stopping_rounds': 40,")
        print(r"    'objective': 'reg:squarederror',")
        print(r"    'tree_method': 'hist',")
        print("    'enable_categorical': True,")
        print("    'random_state': 42,")
        print(r"    'eval_metric': 'rmse',")
        print("    'n_jobs': -1,")
        print("}")
    else:
        print("⚠️ No trials were completed successfully.")

if __name__ == "__main__":
    main()
