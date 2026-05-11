"""
preprocess_toronto.py — Vectorized ETL and Feature Engineering for TTC (Toronto)
=================================================================================
Industrial-grade pipeline mirroring the Boston structure. Consumes raw GTFS-RT
telemetry from the SQLite database produced by the ingestion layer, applies
Downsampling and Spatial Snapping, then uses a vectorized Shift-Join to produce
Origin-Destination pairs capped at a 10-stop lookahead horizon.

Usage:
  python preprocess_toronto.py --split train
  python preprocess_toronto.py --split test
  python preprocess_toronto.py --split both
"""

import argparse
import gc
import logging
import os
import pickle
import sqlite3
import sys
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the ai-engine root (two levels up from this file) is on sys.path
# so that `config.py` can be imported regardless of cwd.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, X_TRAIN_PARQUET, X_TEST_PARQUET

import numpy as np
import pandas as pd
import psutil
import holidays

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess_toronto")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR        = Path(__file__).parent
DB_FILE         = RAW_DATA_DIR / "ttc_vehicle_positions.db"
STATE_FILE      = PROCESSED_DATA_DIR / "toronto_historical_state.pkl"

TORONTO_LAT     = 43.6532
TORONTO_LON     = -79.3832
TIMEZONE        = "America/Toronto"

# Chronological split ratio: 80 % train / 20 % test
TRAIN_RATIO     = 0.80

# Maximum forward-horizon and outlier guard (mirrors Boston)
MAX_LOOKAHEAD_STOPS = 10
MAX_TRAVEL_TIME_S   = 7_200   # 2 hours

# Downsampling cadence: 1-minute bins per vehicle
RESAMPLE_FREQ   = "1min"

# Source SQLite table written by ingest_ttc_gtfs_rt.py
SOURCE_TABLE    = "raw_vehicle_positions"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def get_ram_mb() -> float:
    """Returns current process RSS memory usage in megabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def log_step(message: str) -> None:
    """Prints a decorated section header to the logger."""
    logger.info("=" * 70)
    logger.info(message)
    logger.info("=" * 70)


def vectorized_haversine(
    lat1: pd.Series, lon1: pd.Series,
    lat2: pd.Series, lon2: pd.Series,
) -> pd.Series:
    """Returns great-circle distance in metres using a fully vectorized Haversine formula."""
    earth_radius_m = 6_371_000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi    = np.radians(lat2 - lat1)
    d_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(d_phi / 2.0) ** 2
        + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    )
    central_angle = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (earth_radius_m * central_angle).astype("float32")


# ---------------------------------------------------------------------------
# Step 1 — Load raw telemetry from SQLite and perform chronological split
# ---------------------------------------------------------------------------
def load_raw_telemetry(split_type: str) -> pd.DataFrame:
    """
    Reads the raw_vehicle_positions table from SQLite, converts the Unix
    timestamp to a timezone-aware datetime, and returns only the rows
    belonging to the requested split (train = first 80 %, test = last 20 %).
    """
    log_step(f"STEP 1: RAW DATA LOADING — SPLIT={split_type.upper()}")

    if not DB_FILE.exists():
        raise FileNotFoundError(
            f"SQLite database not found: {DB_FILE}. "
            "Run ingest_ttc_gtfs_rt.py (in ingestion/) first."
        )

    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query(
        f"SELECT * FROM {SOURCE_TABLE} ORDER BY timestamp ASC", conn
    )
    conn.close()

    logger.info("  Total rows from DB: %d | RAM: %.0f MB", len(df), get_ram_mb())

    # Convert Unix epoch (seconds) to tz-aware datetime
    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], unit="s", utc=True)
        .dt.tz_convert(TIMEZONE)
    )

    # Chronological split — avoids data leakage
    split_cutoff = int(len(df) * TRAIN_RATIO)
    if split_type == "train":
        df = df.iloc[:split_cutoff].copy()
    else:  # test
        df = df.iloc[split_cutoff:].copy()

    logger.info("  Rows after chronological split (%s): %d", split_type, len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2 — Downsampling + Spatial Snapping
# ---------------------------------------------------------------------------
def downsample_and_snap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsampling: Resamples GPS pings to a 1-minute fixed cadence per vehicle
    using the first observation within each window. This removes sub-minute
    noise bursts common in GTFS-RT feeds.

    Spatial Snapping: Fills missing stop_id values with a sentinel constant.
    When a proper stop-shape lookup table is available under RAW_DATA_DIR,
    replace this section with a nearest-neighbour join to snap coordinates
    to the closest static GTFS stop.
    """
    log_step("STEP 2: DOWNSAMPLING & SPATIAL SNAPPING")

    # -- Downsampling --------------------------------------------------------
    df = df.set_index("timestamp")
    df = (
        df.groupby("vehicle_id", sort=False)
        .resample(RESAMPLE_FREQ)
        .first()
        .reset_index(level=0, drop=True)   # drop repeated vehicle_id level
        .reset_index()                      # promote timestamp back to column
    )
    df = df.dropna(subset=["current_lat", "current_lon"]).reset_index(drop=True)
    logger.info("  Rows after downsampling to %s cadence: %d", RESAMPLE_FREQ, len(df))

    # -- Spatial Snapping ----------------------------------------------------
    # Replace missing stop IDs with a sentinel.
    # TODO: Replace with a nearest-neighbour snap once the TTC stops.txt is
    # available under RAW_DATA_DIR / "stops.txt".
    df["current_stop_id"] = df["current_stop_id"].fillna("SNAPPED_UNKNOWN")

    # direction_id is not present in raw GTFS-RT vehicle positions.
    # Default to "0" (inbound) as a conservative placeholder; update once
    # the trip_id → direction mapping is joined from static GTFS.
    if "direction_id" not in df.columns:
        df["direction_id"] = "0"

    df = df.sort_values(["vehicle_id", "timestamp"]).reset_index(drop=True)
    logger.info("  RAM after snapping: %.0f MB", get_ram_mb())
    return df


# ---------------------------------------------------------------------------
# Step 3 — Velocity feature engineering (vectorized lags)
# ---------------------------------------------------------------------------
def engineer_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes per-vehicle, chronologically-ordered velocity and three lagged
    velocity features using vectorized Pandas shift operations.
    Also emits a sequential time_point_order used in the Shift-Join loop.
    """
    log_step("STEP 3: VELOCITY FEATURE ENGINEERING")

    df.sort_values(["vehicle_id", "timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    vehicle_group = df.groupby("vehicle_id", sort=False)

    df["prev_lat"]  = vehicle_group["current_lat"].shift(1)
    df["prev_lon"]  = vehicle_group["current_lon"].shift(1)
    df["prev_time"] = vehicle_group["timestamp"].shift(1)

    distance_m = vectorized_haversine(
        df["current_lat"], df["current_lon"],
        df["prev_lat"],   df["prev_lon"],
    )
    elapsed_s = (df["timestamp"] - df["prev_time"]).dt.total_seconds()

    df["vel_ms"] = (distance_m / elapsed_s).replace([np.inf, -np.inf], np.nan).astype("float32")

    df["vel_lag_1"] = vehicle_group["vel_ms"].shift(1).fillna(0.0).astype("float32")
    df["vel_lag_2"] = vehicle_group["vel_ms"].shift(2).fillna(0.0).astype("float32")
    df["vel_lag_3"] = vehicle_group["vel_ms"].shift(3).fillna(0.0).astype("float32")

    # Sequential integer order within each vehicle trip (used in Shift-Join mask)
    df["time_point_order"] = vehicle_group.cumcount()

    df.drop(columns=["prev_lat", "prev_lon", "prev_time", "vel_ms"], inplace=True)
    gc.collect()

    logger.info("  Velocity features computed | RAM: %.0f MB", get_ram_mb())
    return df


# ---------------------------------------------------------------------------
# Step 4 — Shift-Join loop: produce O-D pairs and compute all features
# ---------------------------------------------------------------------------
def run_shift_join(split_type: str, df: pd.DataFrame) -> None:
    """
    Iterates a sliding shift-join from k=1 up to MAX_LOOKAHEAD_STOPS (train)
    or until no valid pairs remain (test). For each hop distance k:
      1. Aligns each origin row with its destination row k steps ahead.
      2. Enforces same-vehicle and forward-time constraints.
      3. Computes travel time, projected distance, temporal features, and
         the expanding-window historical average time (no data leakage).
      4. Writes the final feature set as a Snappy-compressed Parquet partition.
    """
    log_step(f"STEP 4: SHIFT-JOIN O-D GENERATION (split={split_type.upper()})")

    out_dir = X_TRAIN_PARQUET if split_type == "train" else X_TEST_PARQUET
    is_test = split_type == "test"

    # Load persisted historical state from train phase (test split only)
    historical_state: dict = {}
    if is_test and STATE_FILE.exists():
        with open(STATE_FILE, "rb") as f:
            historical_state = pickle.load(f)
        logger.info("  Loaded historical state: %d triplets", len(historical_state))
    elif is_test:
        logger.warning("  No historical_state found. Test stats will start from zero.")

    # Clean / recreate the output partition directory
    import shutil
    if out_dir.exists():
        shutil.rmtree(out_dir) if out_dir.is_dir() else out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build holiday set for Ontario, Canada
    ontario_holidays: set = set()
    for year in [2024, 2025]:
        ontario_holidays.update(holidays.country_holidays("CA", subdiv="ON", years=year).keys())

    # Column subsets reused in every loop iteration
    origin_cols = [
        "vehicle_id", "time_point_order", "current_stop_id", "timestamp",
        "current_lat", "current_lon", "route_id", "direction_id",
        "vel_lag_1", "vel_lag_2", "vel_lag_3",
    ]
    destination_cols = [
        "vehicle_id", "time_point_order", "current_stop_id", "timestamp",
        "current_lat", "current_lon",
    ]

    df_dest    = df[destination_cols]
    total_rows = 0
    group_keys = ["route_id", "stop_id_origen", "stop_id_destino"]

    k = 0
    while True:
        k += 1

        # Train: bounded by lookahead horizon. Test: runs until no pairs left.
        if not is_test and k > MAX_LOOKAHEAD_STOPS:
            break

        shifted_dest = df_dest.shift(-k)

        same_vehicle   = (df["vehicle_id"] == shifted_dest["vehicle_id"]).fillna(False)
        forward_in_time = (shifted_dest["time_point_order"] > df["time_point_order"]).fillna(False)

        if not is_test:
            within_horizon = (
                (shifted_dest["time_point_order"] - df["time_point_order"]) <= MAX_LOOKAHEAD_STOPS
            ).fillna(False)
            valid_pairs = same_vehicle & forward_in_time & within_horizon
        else:
            valid_pairs = same_vehicle & forward_in_time

        if not valid_pairs.any():
            if is_test:
                logger.info("  k=%-2d | No valid O-D pairs remaining. Stopping.", k)
            del shifted_dest
            gc.collect()
            break

        # Extract and rename origin / destination columns
        origin_chunk = (
            df.loc[valid_pairs, origin_cols]
            .rename(columns={
                "time_point_order": "tpo_o",
                "current_stop_id":  "stop_id_origen",
                "timestamp":        "actual_o",
                "current_lat":      "lat_o",
                "current_lon":      "lon_o",
            })
            .reset_index(drop=True)
        )

        dest_chunk = (
            shifted_dest.loc[valid_pairs]
            .drop(columns=["vehicle_id"])
            .rename(columns={
                "time_point_order": "tpo_d",
                "current_stop_id":  "stop_id_destino",
                "timestamp":        "actual_d",
                "current_lat":      "lat_d",
                "current_lon":      "lon_d",
            })
            .reset_index(drop=True)
        )

        chunk = pd.concat([origin_chunk, dest_chunk], axis=1)

        # --- Core feature computation ---
        chunk["tiempo_viaje_segundos"] = (
            (chunk["actual_d"] - chunk["actual_o"])
            .dt.total_seconds()
            .astype("float32")
        )
        chunk["distancia_proyectada"] = vectorized_haversine(
            chunk["lat_o"], chunk["lon_o"],
            chunk["lat_d"], chunk["lon_d"],
        )

        # Discard outliers and rows missing spatial data
        valid_target = (
            (chunk["tiempo_viaje_segundos"] > 0)
            & (chunk["tiempo_viaje_segundos"] <= MAX_TRAVEL_TIME_S)
        )
        chunk = chunk[valid_target].dropna(subset=["distancia_proyectada"]).copy()

        chunk["num_paradas_salto"] = (chunk["tpo_d"] - chunk["tpo_o"]).astype("float32")

        # --- Expanding-window historical average (no data leakage) ---
        chunk.sort_values("actual_o", inplace=True)

        key_tuples = list(
            zip(chunk["route_id"], chunk["stop_id_origen"], chunk["stop_id_destino"])
        )
        prev_sums   = np.array([historical_state.get(kt, (0.0, 0))[0] for kt in key_tuples])
        prev_counts = np.array([historical_state.get(kt, (0.0, 0))[1] for kt in key_tuples])

        cumulative_sum   = (
            chunk.groupby(group_keys, observed=True)["tiempo_viaje_segundos"].cumsum()
            - chunk["tiempo_viaje_segundos"]
        )
        cumulative_count = chunk.groupby(group_keys, observed=True).cumcount()

        total_sum   = prev_sums   + cumulative_sum.values
        total_count = prev_counts + cumulative_count.values

        default_estimate = chunk["distancia_proyectada"] / 5.0   # fallback: dist / 5 m·s⁻¹
        safe_count       = np.where(total_count == 0, 1, total_count)
        average_time     = total_sum / safe_count
        chunk["tiempo_promedio_historico"] = np.where(
            total_count == 0, default_estimate, average_time
        ).astype("float32")

        # Update historical state with this chunk's aggregates
        chunk_aggregates = (
            chunk.groupby(group_keys, observed=True)["tiempo_viaje_segundos"]
            .agg(sum_val="sum", count_val="count")
            .reset_index()
        )
        for row in chunk_aggregates.itertuples(index=False):
            key_idx = (row.route_id, row.stop_id_origen, row.stop_id_destino)
            old_sum, old_count = historical_state.get(key_idx, (0.0, 0))
            historical_state[key_idx] = (old_sum + row.sum_val, old_count + row.count_val)

        # --- Temporal and calendar features ---
        chunk["hora_del_dia"] = chunk["actual_o"].dt.hour.astype("Int8")
        chunk["dia_semana"]   = chunk["actual_o"].dt.dayofweek.astype("Int8")
        chunk["mes"]          = chunk["actual_o"].dt.month.astype("Int8")
        chunk["is_holiday"]   = (
            chunk["actual_o"].dt.date.isin(ontario_holidays).astype("float32")
        )

        # Weather columns are intentionally null here; they can be back-filled
        # by joining against an Open-Meteo historical archive for Toronto once
        # the optional weather fetch module is wired in.
        chunk["temperature_2m"] = np.nan
        chunk["precipitation"]  = 0.0
        chunk["snowfall"]       = 0.0

        # Final column projection — exactly matches the Boston schema and
        # the 'toronto' entry in DATASET_CONFIGS
        final_columns = [
            "tiempo_viaje_segundos",
            "hora_del_dia", "dia_semana", "mes",
            "temperature_2m", "precipitation", "snowfall",
            "is_holiday",
            "route_id", "direction_id", "stop_id_origen", "stop_id_destino",
            "distancia_proyectada",
            "vel_lag_1", "vel_lag_2", "vel_lag_3",
            "num_paradas_salto", "tiempo_promedio_historico",
        ]
        chunk = chunk[final_columns]
        total_rows += len(chunk)

        partition_path = out_dir / f"part_{k}.parquet"
        chunk.to_parquet(partition_path, index=False, compression="snappy")
        logger.info(
            "  k=%-2d | Written %8d rows → %s | RAM: %.0f MB",
            k, len(chunk), partition_path.name, get_ram_mb(),
        )

        del chunk, origin_chunk, dest_chunk, chunk_aggregates, shifted_dest
        gc.collect()

    del df_dest, df
    gc.collect()

    # Persist historical state so the test split can resume from it
    if split_type == "train":
        with open(STATE_FILE, "wb") as f:
            pickle.dump(historical_state, f)
        logger.info(
            "  Saved historical state to disk — %d triplets.", len(historical_state)
        )

    logger.info(
        "  Shift-join complete: %d total rows written to %s", total_rows, out_dir.name
    )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(split_type: str) -> None:
    """Full ETL pipeline for a single split (train or test)."""
    pipeline_start = time.time()
    logger.info(">>> Starting Toronto ETA pipeline — split=%s", split_type)

    df = load_raw_telemetry(split_type)
    df = downsample_and_snap(df)
    df = engineer_velocity_features(df)

    run_shift_join(split_type, df)

    elapsed_minutes = (time.time() - pipeline_start) / 60
    logger.info("Pipeline finished in %.1f min | RAM: %.0f MB", elapsed_minutes, get_ram_mb())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vectorized ETL pipeline for TTC (Toronto) vehicle position data."
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "both"],
        default="both",
        help="Dataset split to process (default: both).",
    )
    args = parser.parse_args()

    if args.split in ("train", "both"):
        run_pipeline("train")
    if args.split in ("test", "both"):
        run_pipeline("test")


if __name__ == "__main__":
    main()
