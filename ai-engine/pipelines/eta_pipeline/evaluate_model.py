"""
evaluate_model.py — Out-of-Time Evaluation for Multiple Datasets
===================================================================
Supports both partitioned directory datasets (Boston, Istanbul) and
single-file datasets (Sivas).

For directory datasets, inference is performed file-by-file (one per
k-horizon) to keep RAM bounded. Global metrics (MAE, RMSE, R²) are
accumulated mathematically without loading all predictions simultaneously.
A stratified breakdown by stop-horizon (k) is also reported.

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
    logger.info("STEP 1: LOADING MODEL for %s", ACTIVE_DATASET.upper())
    logger.info("=" * 70)

    if not MODEL_PATH.exists():
        logger.critical("❌ MODEL NOT FOUND: '%s'. Run train_model.py first.", MODEL_PATH)
        sys.exit(1)

    logger.info("  Loading binary model (.ubj) to preserve RAM...")
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    logger.info("  ✅ Model loaded: %s (%.2f MB)", MODEL_PATH.name, size_mb)
    return model


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Apply categorical and float32 casts to a partition dataframe."""
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    continuous_cols = df.select_dtypes(include=["float64", "float16", "int64", "int32"]).columns
    for col in continuous_cols:
        if col != TARGET:
            df[col] = df[col].astype(np.float32)
    df[TARGET] = df[TARGET].astype(np.float32)
    return df


def report_feature_importance(model: xgb.XGBRegressor) -> None:
    logger.info("=" * 70)
    logger.info("FEATURE IMPORTANCE for %s", ACTIVE_DATASET.upper())
    logger.info("=" * 70)
    importance_dict = model.get_booster().get_score(importance_type="gain")
    if not importance_dict:
        logger.warning("  ⚠️  No importance data available.")
        return
    fi_df = (
        pd.DataFrame.from_dict(importance_dict, orient="index", columns=["gain"])
        .sort_values("gain", ascending=False)
        .reset_index()
    )
    print("\n  Top Feature Importance (Gain):")
    for _, row in fi_df.head(10).iterrows():
        print(f"    {row['index']:<25}: {row['gain']:.2f}")


# ---------------------------------------------------------------------------
# Chunked Evaluation (directories with part_k.parquet)
# ---------------------------------------------------------------------------

def evaluate_partitioned(model: xgb.XGBRegressor, parquet_dir: Path) -> None:
    """
    Iterate over sorted part_k.parquet files one at a time.
    Accumulate global metrics (MAE, RMSE, R²) mathematically using running
    sums — no full prediction arrays are kept in RAM simultaneously.
    Also accumulates per-k stratified statistics for the breakdown table.
    """
    logger.info("=" * 70)
    logger.info("STEP 2: CHUNKED INFERENCE over directory: %s", parquet_dir.name)
    logger.info("=" * 70)

    parquet_files = sorted(parquet_dir.glob("part_*.parquet"))
    if not parquet_files:
        logger.error("❌ No part_*.parquet files found in %s", parquet_dir)
        sys.exit(1)

    logger.info("  Found %d partition files.", len(parquet_files))

    # Global accumulators (computed in a single pass)
    total_n   = 0
    sum_ae    = 0.0   # for MAE
    sum_se    = 0.0   # for RMSE
    sum_y     = 0.0   # for R² (need global mean)
    sum_y2    = 0.0   # for R² SS_tot
    sum_res2  = 0.0   # for R² SS_res

    # Per-k-horizon accumulators: {k: {"n": int, "sum_ae": float, "sum_se": float}}
    k_stats: dict[int, dict] = {}

    t_total_infer = 0.0

    for part_path in parquet_files:
        # Extract k from filename (e.g. part_03.parquet → k=3)
        try:
            k = int(part_path.stem.split("_")[1])
        except (IndexError, ValueError):
            k = -1

        df = pd.read_parquet(part_path)
        df = cast_types(df)

        X = df.drop(columns=[TARGET])
        y = df[TARGET].values.astype(np.float64)
        del df
        gc.collect()

        t0 = time.time()
        y_pred = model.predict(X).astype(np.float64)
        t_total_infer += time.time() - t0
        del X
        gc.collect()

        n = len(y)
        ae = np.abs(y - y_pred)
        se = (y - y_pred) ** 2

        # Global accumulators
        total_n  += n
        sum_ae   += ae.sum()
        sum_se   += se.sum()
        sum_y    += y.sum()
        sum_y2   += (y ** 2).sum()
        sum_res2 += se.sum()

        # Per-k accumulators
        if k not in k_stats:
            k_stats[k] = {"n": 0, "sum_ae": 0.0, "sum_se": 0.0}
        k_stats[k]["n"]      += n
        k_stats[k]["sum_ae"] += ae.sum()
        k_stats[k]["sum_se"] += se.sum()

        logger.info(
            "  k=%-2d | %8d rows | MAE=%.2f | RMSE=%.2f | RAM: %.0f MB",
            k, n, ae.mean(), np.sqrt(se.mean()), psutil.Process(os.getpid()).memory_info().rss / 1024**2,
        )
        del y, y_pred, ae, se
        gc.collect()

    # -----------------------------------------------------------------------
    # Global metrics
    # -----------------------------------------------------------------------
    global_mae  = sum_ae / total_n
    global_rmse = np.sqrt(sum_se / total_n)
    global_mean = sum_y / total_n
    ss_tot      = sum_y2 - total_n * global_mean ** 2
    global_r2   = 1.0 - (sum_res2 / ss_tot) if ss_tot > 0 else 0.0
    throughput  = total_n / t_total_infer if t_total_infer > 0 else 0

    # -----------------------------------------------------------------------
    # Print global report
    # -----------------------------------------------------------------------
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print(f"║       OUT-OF-TIME REPORT — {ACTIVE_DATASET.upper():<41} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Samples    : {f'{total_n:,}':<58} ║")
    print(f"║  Throughput : {f'{throughput:,.0f} records/s':<58} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"  {'MAE  — Mean Absolute Error':<45} {global_mae:>10.2f}")
    print(f"  {'RMSE — Root Mean Squared Error':<45} {global_rmse:>10.2f}")
    print(f"  {'R²   — Coefficient of Determination':<45} {global_r2:>10.4f}")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    # -----------------------------------------------------------------------
    # Stratified breakdown by k-horizon
    # -----------------------------------------------------------------------
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║  STRATIFIED ANALYSIS BY STOP HORIZON (k)                                ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║ {'k':<6} | {'Samples':<14} | {'MAE':>10} | {'RMSE':>10} | {'% of Total':>10} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    for k_val in sorted(k_stats.keys()):
        st      = k_stats[k_val]
        mae_k   = st["sum_ae"] / st["n"]
        rmse_k  = np.sqrt(st["sum_se"] / st["n"])
        pct     = 100.0 * st["n"] / total_n
        print(f"║ {k_val:<6} | {f'{st[\"n\"]:,}':<14} | {mae_k:>10.2f} | {rmse_k:>10.2f} | {pct:>9.1f}% ║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()


# ---------------------------------------------------------------------------
# Single-file Evaluation (Sivas)
# ---------------------------------------------------------------------------

def evaluate_single_file(model: xgb.XGBRegressor, parquet_path: Path) -> None:
    """Standard monolithic evaluation for single-file datasets (Sivas)."""
    logger.info("=" * 70)
    logger.info("STEP 2: INFERENCE — single file: %s", parquet_path.name)
    logger.info("=" * 70)

    df = pd.read_parquet(parquet_path)
    df = cast_types(df)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    del df
    gc.collect()

    logger.info("  🔮 Inference over %d samples...", len(X))
    t0 = time.time()
    y_pred = model.predict(X)
    t_infer = time.time() - t0
    logger.info("  ✅ Inference: %.3fs | throughput=%.0f records/s", t_infer, len(X) / t_infer)

    y_arr        = y.values
    mask_nonzero = y_arr != 0
    y_nz         = y_arr[mask_nonzero]
    yp_nz        = y_pred[mask_nonzero]

    mae  = mean_absolute_error(y_arr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
    r2   = r2_score(y_arr, y_pred)
    mape = np.mean(np.abs((y_nz - yp_nz) / y_nz)) * 100 if len(y_nz) > 0 else 0.0

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

    # Stratified analysis if num_paradas_salto column is present
    if "num_paradas_salto" in X.columns:
        print("╔══════════════════════════════════════════════════════════════════════════╗")
        print("║  STRATIFIED ANALYSIS BY STOP HORIZON                                    ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        print(f"║ {'k':<6} | {'Samples':<14} | {'MAE':>10} | {'RMSE':>10} | {'R²':>10} ║")
        print("╠══════════════════════════════════════════════════════════════════════════╣")
        hops = X["num_paradas_salto"].values
        for k_val in range(1, 11):
            mask_k = hops == k_val
            if not mask_k.any():
                continue
            y_k, yp_k = y_arr[mask_k], y_pred[mask_k]
            mae_k  = mean_absolute_error(y_k, yp_k)
            rmse_k = np.sqrt(mean_squared_error(y_k, yp_k))
            try:
                r2_k = r2_score(y_k, yp_k)
            except Exception:
                r2_k = 0.0
            print(f"║ {k_val:<6} | {f'{mask_k.sum():,}':<14} | {mae_k:>10.2f} | {rmse_k:>10.2f} | {r2_k:>10.4f} ║")
        print("╚══════════════════════════════════════════════════════════════════════════╝")
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║  MBTA XGBoost — Evaluation: %-31s ║", ACTIVE_DATASET.upper())
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    log_memory_usage("start")

    try:
        model = load_model()
        log_memory_usage("after model load")

        if PARQUET_IN.is_dir():
            # Directory-based: Boston, Istanbul
            evaluate_partitioned(model, PARQUET_IN)
        else:
            # Single-file: Sivas
            evaluate_single_file(model, PARQUET_IN)

        report_feature_importance(model)
        logger.info(
            "🎉 Evaluation for %s complete in %.1f min",
            ACTIVE_DATASET, (time.time() - total_start) / 60,
        )

    except Exception as exc:
        logger.critical("❌ Error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
