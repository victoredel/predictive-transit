"""
evaluate_model.py — Out-of-Time Evaluation for Multiple Datasets
===================================================================
Supports both partitioned directory datasets (Boston, Istanbul) and
single-file datasets (Sivas).

For directory datasets, inference is performed file-by-file (one per
k-horizon) to keep RAM bounded. Global metrics (MAE, RMSE, R2) are
accumulated mathematically without loading all predictions simultaneously.
A stratified breakdown by stop-horizon (k) is also reported.

Chained Inference (Boston / Istanbul — k > MAX_LOOKAHEAD_STOPS):
-----------------------------------------------------------------
The XGBoost model was trained on O-D pairs up to MAX_LOOKAHEAD_STOPS (10).
For test partitions where k > 10, a direct prediction is out of the training
distribution. Instead we apply a greedy additive decomposition:

    k = 15  →  segments = [10, 5]
    t_hat(k=15) = predict(X | num_paradas_salto=10)
                + predict(X | num_paradas_salto=5)

This is mathematically equivalent to assuming travel time is additive across
sub-segments, which holds for transit systems where consecutive legs share
the same route context (same route_id, direction_id, origin speed lags).
The approach preserves the feature distribution seen during training and
avoids distributional shift on num_paradas_salto.

Usage:
  ACTIVE_DATASET=boston python evaluate_model.py
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

# Ensure ai-engine root is importable
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import (
    X_TEST_PARQUET as PARQUET_IN,
    MODEL_PATH,
    TARGET,
    CATEGORICAL_FEATURES,
    ACTIVE_DATASET,
    MAX_LOOKAHEAD_STOPS,
)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_model")

# Width of the report box
_BOX_W = 76


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)


def log_step(msg: str) -> None:
    logger.info("=" * 70)
    logger.info(msg)
    logger.info("  RAM: %.0f MB", ram_mb())
    logger.info("=" * 70)


def box_line(content: str, width: int = _BOX_W) -> str:
    """Return a box line padded to exactly `width` chars between the borders."""
    return f"| {content:<{width}} |"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model() -> xgb.XGBRegressor:
    log_step(f"STEP 1: LOADING MODEL for {ACTIVE_DATASET.upper()}")

    if not MODEL_PATH.exists():
        logger.critical("MODEL NOT FOUND: '%s'. Run train_model.py first.", MODEL_PATH)
        sys.exit(1)

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    logger.info("  Model loaded: %s (%.2f MB) | RAM: %.0f MB",
                MODEL_PATH.name, size_mb, ram_mb())
    return model


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply categorical casts (required by XGBoost) and downcast continuous
    columns to float32 to reduce RAM. TARGET is kept as float32 for
    accumulator precision (will be upcast to float64 during metric math).
    """
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    for col in df.select_dtypes(include=["float64", "float16", "int64", "int32"]).columns:
        df[col] = df[col].astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Chained Inference
# ---------------------------------------------------------------------------

def predict_chained(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    k: int,
) -> np.ndarray:
    """
    Predict travel time for an O-D pair spanning k stops when k exceeds
    the model's training horizon (MAX_LOOKAHEAD_STOPS).

    Mathematical decomposition
    --------------------------
    Given that travel times are additive across consecutive sub-legs, and
    assuming that the feature context (speed lags, route, historical average)
    is representative of the full journey:

        t_hat(k) = sum_i [ predict(X | num_paradas_salto = s_i) ]

    where [s_1, s_2, ..., s_m] is a partition of k into segments each of
    size <= MAX_LOOKAHEAD_STOPS.

    Example: k=25, MAX=10  ->  segments = [10, 10, 5]
    Caveat: the historical average feature (tiempo_promedio_historico) in X
    already reflects the full-k O-D pair; for segments it would ideally be
    rescaled proportionally. We apply a proportional scaling heuristic:
        tph_segment = tph_full * (seg_k / k)
    This keeps the feature in-distribution while preserving the ratio.

    Parameters
    ----------
    model   : loaded XGBRegressor (.ubj)
    X       : feature matrix for rows where num_paradas_salto == k
    k       : actual stop-horizon of these rows

    Returns
    -------
    np.ndarray, shape (len(X),), dtype float64
        Summed predicted travel times in seconds.
    """
    n_full   = k // MAX_LOOKAHEAD_STOPS
    remainder = k %  MAX_LOOKAHEAD_STOPS

    segment_sizes: list[int] = [MAX_LOOKAHEAD_STOPS] * n_full
    if remainder > 0:
        segment_sizes.append(remainder)

    total_pred = np.zeros(len(X), dtype=np.float64)

    for seg_k in segment_sizes:
        X_seg = X.copy()

        # Override num_paradas_salto to the sub-segment size
        if "num_paradas_salto" in X_seg.columns:
            X_seg["num_paradas_salto"] = np.float32(seg_k)

        # Proportionally scale tiempo_promedio_historico so that the feature
        # remains in the training distribution for a seg_k-stop horizon.
        if "tiempo_promedio_historico" in X_seg.columns:
            scale = seg_k / k
            X_seg["tiempo_promedio_historico"] = (
                X_seg["tiempo_promedio_historico"].astype(np.float32) * np.float32(scale)
            )

        seg_pred = model.predict(X_seg).astype(np.float64)
        total_pred += seg_pred
        del X_seg
        gc.collect()

    logger.info(
        "    [chained] k=%d -> segments %s | %d sub-predictions | RAM: %.0f MB",
        k, segment_sizes, len(segment_sizes), ram_mb(),
    )
    return total_pred


# ---------------------------------------------------------------------------
# Chunked Evaluation  (Boston / Istanbul — partitioned directories)
# ---------------------------------------------------------------------------

def evaluate_partitioned(model: xgb.XGBRegressor, parquet_dir: Path) -> None:
    """
    Iterate over sorted part_k.parquet files one at a time.

    RAM strategy
    ------------
    - Load one partition -> infer -> accumulate scalars -> delete arrays.
    - Never hold more than one partition in memory simultaneously.
    - All global and per-k metrics are computed from running sums, not arrays.

    Chaining
    --------
    Partitions with k <= MAX_LOOKAHEAD_STOPS: direct model.predict().
    Partitions with k > MAX_LOOKAHEAD_STOPS: predict_chained() decomposes k
    into sub-segments and sums the model's outputs.
    """
    log_step(f"STEP 2: CHUNKED INFERENCE over directory: {parquet_dir.name}")

    parquet_files = sorted(
        parquet_dir.glob("part_*.parquet"),
        key=lambda p: int(p.stem.split("_")[1]),   # numeric sort: part_2 < part_10
    )
    if not parquet_files:
        logger.error("No part_*.parquet files found in %s", parquet_dir)
        sys.exit(1)

    logger.info("  Found %d partition files.", len(parquet_files))

    # ------------------------------------------------------------------
    # Global running sums (no arrays stored)
    # ------------------------------------------------------------------
    total_n   = 0
    sum_ae    = 0.0   # sum of |y - y_hat|          -> MAE
    sum_se    = 0.0   # sum of (y - y_hat)^2        -> RMSE
    sum_y     = 0.0   # sum of y                    -> global mean for R2
    sum_y2    = 0.0   # sum of y^2                  -> SS_tot
    sum_res2  = 0.0   # sum of (y - y_hat)^2        -> SS_res (= sum_se)

    # Per-k accumulators: {k: {"n": int, "sum_ae": float, "sum_se": float,
    #                          "mode": "direct" | "chained"}}
    k_stats: dict[int, dict] = {}

    t_total_infer = 0.0
    t_total_load  = 0.0

    for part_path in parquet_files:
        # -----------------------------------------------------------------
        # Extract k from filename: part_3.parquet -> 3, part_10.parquet -> 10
        # -----------------------------------------------------------------
        try:
            k = int(part_path.stem.split("_")[1])
        except (IndexError, ValueError):
            logger.warning("  Cannot parse k from filename %s — skipping.", part_path.name)
            continue

        # -----------------------------------------------------------------
        # Load partition
        # -----------------------------------------------------------------
        t_load_start = time.time()
        df = pd.read_parquet(part_path)
        df = cast_types(df)
        t_total_load += time.time() - t_load_start

        n_rows = len(df)
        X = df.drop(columns=[TARGET])
        y = df[TARGET].values.astype(np.float64)
        del df
        gc.collect()

        # -----------------------------------------------------------------
        # Inference: direct or chained
        # -----------------------------------------------------------------
        t0 = time.time()
        if k <= MAX_LOOKAHEAD_STOPS:
            y_pred = model.predict(X).astype(np.float64)
            infer_mode = "direct"
        else:
            logger.info(
                "  k=%-2d | k > %d: applying chained inference.",
                k, MAX_LOOKAHEAD_STOPS,
            )
            y_pred = predict_chained(model, X, k)
            infer_mode = "chained"
        t_total_infer += time.time() - t0

        del X
        gc.collect()

        # -----------------------------------------------------------------
        # Compute partition metrics and accumulate
        # -----------------------------------------------------------------
        ae = np.abs(y - y_pred)
        se = (y - y_pred) ** 2

        total_n  += n_rows
        sum_ae   += float(ae.sum())
        sum_se   += float(se.sum())
        sum_y    += float(y.sum())
        sum_y2   += float((y ** 2).sum())
        sum_res2 += float(se.sum())

        if k not in k_stats:
            k_stats[k] = {"n": 0, "sum_ae": 0.0, "sum_se": 0.0, "mode": infer_mode}
        k_stats[k]["n"]      += n_rows
        k_stats[k]["sum_ae"] += float(ae.sum())
        k_stats[k]["sum_se"] += float(se.sum())

        logger.info(
            "  k=%-2d | %9s | %8d rows | MAE=%8.2f | RMSE=%8.2f | RAM: %.0f MB",
            k, infer_mode, n_rows, ae.mean(), np.sqrt(se.mean()), ram_mb(),
        )
        del y, y_pred, ae, se
        gc.collect()

    if total_n == 0:
        logger.error("No rows accumulated — check partition files.")
        return

    # ------------------------------------------------------------------
    # Final global metrics
    # ------------------------------------------------------------------
    global_mae  = sum_ae / total_n
    global_rmse = np.sqrt(sum_se / total_n)
    global_mean = sum_y / total_n
    ss_tot      = sum_y2 - total_n * global_mean ** 2
    global_r2   = 1.0 - (sum_res2 / ss_tot) if ss_tot > 0 else 0.0
    throughput  = total_n / t_total_infer if t_total_infer > 0 else 0.0

    # ------------------------------------------------------------------
    # Global report
    # ------------------------------------------------------------------
    W = 72
    print()
    print("=" * W)
    print(f"  OUT-OF-TIME REPORT  |  {ACTIVE_DATASET.upper()}")
    print("=" * W)
    print(f"  {'Evaluation samples':<35} {total_n:>15,}")
    print(f"  {'Throughput':<35} {throughput:>12,.0f} rec/s")
    print(f"  {'Load time':<35} {t_total_load:>14.2f} s")
    print(f"  {'Inference time':<35} {t_total_infer:>14.2f} s")
    print("-" * W)
    print(f"  {'MAE  (Mean Absolute Error)':<35} {global_mae:>15.2f} s")
    print(f"  {'RMSE (Root Mean Squared Error)':<35} {global_rmse:>15.2f} s")
    print(f"  {'R2   (Coeff. of Determination)':<35} {global_r2:>15.4f}")
    print("=" * W)
    print()

    # ------------------------------------------------------------------
    # Stratified breakdown by k-horizon
    # ------------------------------------------------------------------
    hdr_k       = "k"
    hdr_mode    = "Inference"
    hdr_samples = "Samples"
    hdr_mae     = "MAE (s)"
    hdr_rmse    = "RMSE (s)"
    hdr_pct     = "% Total"

    print("=" * W)
    print("  STRATIFIED ANALYSIS BY STOP HORIZON (k)")
    print("=" * W)
    print(
        f"  {hdr_k:<4} | {hdr_mode:<9} | {hdr_samples:>10} | "
        f"{hdr_mae:>10} | {hdr_rmse:>10} | {hdr_pct:>8}"
    )
    print("-" * W)

    for k_val in sorted(k_stats.keys()):
        st     = k_stats[k_val]
        mae_k  = st["sum_ae"] / st["n"]
        rmse_k = np.sqrt(st["sum_se"] / st["n"])
        pct    = 100.0 * st["n"] / total_n
        mode   = st["mode"]
        n_fmt  = f"{st['n']:,}"
        pct_fmt = f"{pct:.1f}%"
        print(
            f"  {k_val:<4} | {mode:<9} | {n_fmt:>10} | "
            f"{mae_k:>10.2f} | {rmse_k:>10.2f} | {pct_fmt:>8}"
        )

    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# Single-file Evaluation  (Sivas)
# ---------------------------------------------------------------------------

def evaluate_single_file(model: xgb.XGBRegressor, parquet_path: Path) -> None:
    """Monolithic evaluation for single-file datasets (Sivas)."""
    log_step(f"STEP 2: INFERENCE — single file: {parquet_path.name}")

    df = pd.read_parquet(parquet_path)
    df = cast_types(df)

    X     = df.drop(columns=[TARGET])
    y_arr = df[TARGET].values.astype(np.float64)
    del df
    gc.collect()

    logger.info("  Inference over %d samples | RAM: %.0f MB", len(X), ram_mb())
    t0     = time.time()
    y_pred = model.predict(X).astype(np.float64)
    t_inf  = time.time() - t0
    logger.info(
        "  Done: %.3fs | %.0f records/s | RAM: %.0f MB",
        t_inf, len(X) / t_inf, ram_mb(),
    )

    mask_nz = y_arr != 0
    y_nz    = y_arr[mask_nz]
    yp_nz   = y_pred[mask_nz]

    mae  = mean_absolute_error(y_arr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
    r2   = r2_score(y_arr, y_pred)
    mape = np.mean(np.abs((y_nz - yp_nz) / y_nz)) * 100 if len(y_nz) > 0 else 0.0

    W = 72
    n_fmt = f"{len(y_arr):,}"
    print()
    print("=" * W)
    print(f"  OUT-OF-TIME REPORT  |  {ACTIVE_DATASET.upper()}")
    print("=" * W)
    print(f"  {'Evaluation samples':<35} {n_fmt:>15}")
    print("-" * W)
    print(f"  {'MAE  (Mean Absolute Error)':<35} {mae:>15.2f}")
    print(f"  {'RMSE (Root Mean Squared Error)':<35} {rmse:>15.2f}")
    print(f"  {'R2   (Coeff. of Determination)':<35} {r2:>15.4f}")
    print(f"  {'MAPE (Mean Absolute Pct Error)':<35} {mape:>14.2f} %")
    print("=" * W)
    print()

    # Stratified analysis if column is present
    if "num_paradas_salto" in X.columns:
        hops = X["num_paradas_salto"].values
        print("=" * W)
        print("  STRATIFIED ANALYSIS BY STOP HORIZON (k)")
        print("=" * W)
        print(f"  {'k':<4} | {'Samples':>10} | {'MAE':>10} | {'RMSE':>10} | {'R2':>10}")
        print("-" * W)
        unique_hops = sorted(np.unique(hops).astype(int))
        for k_val in unique_hops:
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
            n_k_fmt = f"{mask_k.sum():,}"
            print(
                f"  {k_val:<4} | {n_k_fmt:>10} | {mae_k:>10.2f} | "
                f"{rmse_k:>10.2f} | {r2_k:>10.4f}"
            )
        print("=" * W)
        print()


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def report_feature_importance(model: xgb.XGBRegressor) -> None:
    log_step(f"FEATURE IMPORTANCE — {ACTIVE_DATASET.upper()}")
    importance_dict = model.get_booster().get_score(importance_type="gain")
    if not importance_dict:
        logger.warning("  No importance data available.")
        return
    fi_df = (
        pd.DataFrame.from_dict(importance_dict, orient="index", columns=["gain"])
        .sort_values("gain", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    print("\n  Top 15 Features by Gain:")
    print(f"  {'Feature':<30} {'Gain':>12}")
    print("  " + "-" * 44)
    for _, row in fi_df.head(15).iterrows():
        print(f"  {row['feature']:<30} {row['gain']:>12.2f}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()
    log_step(
        f"Evaluate Model — {ACTIVE_DATASET.upper()} "
        f"| MAX_LOOKAHEAD={MAX_LOOKAHEAD_STOPS}"
    )

    try:
        model = load_model()

        if PARQUET_IN.is_dir():
            evaluate_partitioned(model, PARQUET_IN)
        else:
            evaluate_single_file(model, PARQUET_IN)

        report_feature_importance(model)

        elapsed = (time.time() - t_start) / 60.0
        logger.info(
            "Evaluation for %s complete in %.1f min | RAM: %.0f MB",
            ACTIVE_DATASET, elapsed, ram_mb(),
        )

    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
