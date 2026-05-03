"""
evaluate_model.py — Out-of-Time Evaluation for Multiple Datasets
===================================================================
Modular evaluation script that handles both Boston and Sivas.
Integrates dynamic config for paths, target, and features.

Usage:
  python evaluate_model.py
"""

import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure ai-engine root (two levels up) is importable
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import dynamic configuration
from config import X_TEST_PARQUET as PARQUET_IN, MODEL_PATH, TARGET, CATEGORICAL_FEATURES, ACTIVE_DATASET

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_model")

def log_memory_usage(context: str) -> None:
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 ** 2)
    logger.info("🧠 RAM [%s]: %.1f MB", context, rss_mb)

def load_model() -> xgb.XGBRegressor:
    logger.info("=" * 70)
    logger.info(f"STEP 1: LOADING MODEL for {ACTIVE_DATASET.upper()}")
    logger.info("=" * 70)

    if not MODEL_PATH.exists():
        logger.critical(f"❌ MODEL NOT FOUND: '{MODEL_PATH}'. Run train_model.py first.")
        sys.exit(1)

    logger.info("  Loading binary model (.ubj) to preserve RAM...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    logger.info("  ✅ Model loaded: %s (%.2f MB)", MODEL_PATH.name, size_mb)
    return model

def evaluate(model, X, y) -> dict:
    logger.info("=" * 70)
    logger.info(f"INFERENCE AND METRICS: {ACTIVE_DATASET.upper()}")
    logger.info("=" * 70)

    logger.info("  🔮 Inference over %d samples ...", len(X))
    t0 = time.time()
    y_pred    = model.predict(X)
    t_infer   = time.time() - t0
    throughput = len(X) / t_infer

    logger.info("  ✅ Inference: %.3fs | throughput=%.0f records/s", t_infer, throughput)

    y_arr        = y.values
    mask_nonzero = y_arr != 0
    y_nz         = y_arr[mask_nonzero]
    yp_nz        = y_pred[mask_nonzero]

    mae  = mean_absolute_error(y_arr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
    r2   = r2_score(y_arr, y_pred)
    mape = np.mean(np.abs((y_nz - yp_nz) / y_nz)) * 100 if len(y_nz) > 0 else 0.0

    metrics = {
        "mae":   float(mae),
        "rmse":  float(rmse),
        "r2":    float(r2),
        "mape":  float(mape),
    }

    # Print Report Table
    div = "─" * 72
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║       OUT-OF-TIME REPORT — {ACTIVE_DATASET.upper():<41} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Samples    : {f'{len(y):,}':<58} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"  {'MAE  — Mean Absolute Error':<45} {mae:>10.2f}")
    print(f"  {'RMSE — Root Mean Squared Error':<45} {rmse:>10.2f}")
    print(f"  {'R²   — Coefficient of Determination':<45} {r2:>10.4f}")
    print(f"  {'MAPE — Mean Absolute Percentage Error':<45} {mape:>10.2f} %")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    # Stratified Analysis (Boston only)
    if "num_paradas_salto" in X.columns:
        print("╔══════════════════════════════════════════════════════════════════════════╗")
        print("║  STRATIFIED ANALYSIS BY STOP HORIZON (BOSTON)                            ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print(f"║ {'Stops':<9} | {'Samples':<12} | {'MAE':<10} | {'RMSE':<10} | {'MAPE %':<10} | {'R²':<10} ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        
        num_paradas_arr = X["num_paradas_salto"].values
        for k in range(1, 11):
            mask_k = num_paradas_arr == k
            if not mask_k.any(): continue
            y_k, yp_k = y_arr[mask_k], y_pred[mask_k]
            mae_k  = mean_absolute_error(y_k, yp_k)
            rmse_k = np.sqrt(mean_squared_error(y_k, yp_k))
            try: r2_k = r2_score(y_k, yp_k)
            except: r2_k = 0.0
            print(f"║ {k:<9} | {f'{mask_k.sum():,}':<12} | {mae_k:<10.2f} | {rmse_k:<10.2f} | {'N/A':<10} | {r2_k:<10.4f} ║")
        print("╚══════════════════════════════════════════════════════════════════════════╝")
        print()

    return metrics, y_pred

def report_feature_importance(model):
    logger.info("=" * 70)
    logger.info(f"FEATURE IMPORTANCE for {ACTIVE_DATASET.upper()}")
    logger.info("=" * 70)
    importance_dict = model.get_booster().get_score(importance_type="gain")
    if not importance_dict:
        logger.warning("  ⚠️  No importance data available.")
        return
    fi_df = pd.DataFrame.from_dict(importance_dict, orient="index", columns=["gain"]).sort_values("gain", ascending=False).reset_index()
    print("\n  Top Feature Importance (Gain):")
    for idx, row in fi_df.head(10).iterrows():
        print(f"    {row['index']:<25}: {row['gain']:.2f}")

def main() -> None:
    total_start = time.time()
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info(f"║  MBTA XGBoost — Evaluation: {ACTIVE_DATASET.upper():<31} ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    log_memory_usage("start")

    try:
        model = load_model()
        
        # Load Partitioned or Single File
        if PARQUET_IN.is_dir():
            parquet_files = list(PARQUET_IN.glob("*.parquet"))
            dfs = [pd.read_parquet(f).sample(frac=0.30, random_state=42) if ACTIVE_DATASET=="boston" else pd.read_parquet(f) for f in parquet_files]
            df = pd.concat(dfs, ignore_index=True)
        else:
            df = pd.read_parquet(PARQUET_IN)

        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                df[col] = df[col].astype("category")
                
        continuous_cols = df.select_dtypes(include=['float64', 'float16', 'int64', 'int32']).columns
        for col in continuous_cols:
            if col != TARGET:
                df[col] = df[col].astype(np.float32)
        df[TARGET] = df[TARGET].astype(np.float32)

        X_test = df.drop(columns=[TARGET])
        y_test = df[TARGET]
        del df; gc.collect()

        evaluate(model, X_test, y_test)
        report_feature_importance(model)

        logger.info(f"🎉 Evaluation for {ACTIVE_DATASET} complete in {(time.time()-total_start)/60:.1f} min")

    except Exception as exc:
        logger.critical(f"❌ Error: {exc}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
