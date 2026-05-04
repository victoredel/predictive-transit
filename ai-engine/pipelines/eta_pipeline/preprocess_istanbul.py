"""
preprocess_istanbul.py — Leakage-Free Preprocessing for Istanbul GTFS-IETT
===========================================================================
Implements a strict chronological train/test split at the trip_id level to
prevent data leakage. Applies a Chunking & Flush strategy (one Parquet per
k-horizon) to keep RAM usage bounded on 30 GB Kaggle instances.

Expanding Window historical average (tiempo_promedio_historico) is computed
using only data with arrival_seconds strictly before the current row,
mirroring the anti-leakage pattern from the Boston pipeline.

Usage:
  python preprocess_istanbul.py
"""

import gc
import logging
import os
import pickle
import shutil
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import openmeteo_requests
import requests_cache

# Ensure ai-engine root is importable
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TARGET,
    CATEGORICAL_FEATURES,
    X_TRAIN_PARQUET,
    X_TEST_PARQUET,
    MAX_LOOKAHEAD_STOPS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess_istanbul")

# Persists historical averages from train split so the test split can reuse them
STATE_FILE = PROCESSED_DATA_DIR / "istanbul_historical_state.pkl"

# Max plausible travel time between stops in Istanbul (2 hours)
MAX_TRAVEL_TIME_S = 7_200

# Final feature columns written to every part_k.parquet
FINAL_COLS = [
    TARGET,
    "hora_del_dia",
    "temperature_2m",
    "precipitation",
    "stop_id",
    "route_id",
    "direction_id",
    "dest_stop_id",
    "stop_lat",
    "stop_lon",
    "dest_lat",
    "dest_lon",
    "arrival_seconds",
    "distancia_proyectada",
    "velocidad_tramo_m_s",
    "vel_lag_1",
    "vel_lag_2",
    "vel_lag_3",
    "num_paradas_salto",
    "tiempo_promedio_historico",
]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2


def log_step(msg: str) -> None:
    logger.info("=" * 70)
    logger.info(msg)
    logger.info("  RAM: %.0f MB", ram_mb())
    logger.info("=" * 70)


def calc_haversine_vectorized(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorized Haversine distance in meters."""
    R = 6_371_000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c).astype("float32")


def parse_time_string(time_str):
    """Convert HH:MM:SS string (including >24h) to pd.Timedelta."""
    if pd.isna(time_str) or time_str == "":
        return pd.NaT
    try:
        h, m, s = map(int, str(time_str).strip().split(":"))
        return pd.Timedelta(hours=h, minutes=m, seconds=s)
    except Exception:
        return pd.NaT


def clean_coord(val) -> float:
    """Normalize coordinate strings like '410082' → 41.0082."""
    if pd.isna(val):
        return np.nan
    s = str(val)
    digits = "".join(filter(str.isdigit, s))
    if len(digits) < 2:
        return np.nan
    try:
        return float(digits[:2] + "." + digits[2:])
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def fetch_istanbul_weather() -> pd.DataFrame:
    """Fetch historical hourly weather for Istanbul via Open-Meteo archive."""
    ISTANBUL_LAT = 41.0082
    ISTANBUL_LON = 28.9784
    cache_session = requests_cache.CachedSession(".weather_cache_istanbul", expire_after=-1)
    om = openmeteo_requests.Client(session=cache_session)
    target_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "latitude": ISTANBUL_LAT,
        "longitude": ISTANBUL_LON,
        "start_date": target_date,
        "end_date": target_date,
        "hourly": ["temperature_2m", "precipitation"],
    }
    try:
        logger.info("  Fetching weather for Istanbul on %s...", target_date)
        resp = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )[0]
        hourly = resp.Hourly()
        temps  = hourly.Variables(0).ValuesAsNumpy()
        precip = hourly.Variables(1).ValuesAsNumpy()
        return pd.DataFrame({
            "hora_del_dia":  np.arange(len(temps), dtype="int8"),
            "temperature_2m": temps.astype("float32"),
            "precipitation":  precip.astype("float32"),
        })
    except Exception as exc:
        logger.error("  Failed to fetch weather: %s. Using zeros.", exc)
        return pd.DataFrame()


def load_and_prepare_base() -> pd.DataFrame:
    """
    Load stop_times.txt, interpolate missing arrival times, merge stop
    coordinates and trips metadata (route_id, direction_id), and clean
    coordinate columns.
    Returns a single base DataFrame sorted by (trip_id, stop_sequence).
    """
    stop_times_path = RAW_DATA_DIR / "gtfs_iett" / "stop_times" / "stop_times.txt"
    stops_path      = RAW_DATA_DIR / "gtfs_iett" / "stops.csv"
    trips_path      = RAW_DATA_DIR / "gtfs_iett" / "trips.csv"

    log_step("STEP 1: Loading full stop_times.txt")
    logger.warning("  Source: %s", stop_times_path)

    stop_times_dtypes = {
        "trip_id":       "category",
        "stop_id":       "int32",
        "stop_sequence": "int32",
    }
    df = pd.read_csv(
        stop_times_path,
        sep=",",
        usecols=["trip_id", "stop_id", "stop_sequence", "arrival_time"],
        dtype=stop_times_dtypes,
    )
    logger.info("  Loaded %d rows | RAM: %.0f MB", len(df), ram_mb())

    log_step("STEP 2: Parsing and Interpolating Arrival Times")
    df["arrival_time_td"] = df["arrival_time"].apply(parse_time_string)
    df.sort_values(["trip_id", "stop_sequence"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["arrival_seconds"] = df["arrival_time_td"].dt.total_seconds()
    df["arrival_seconds"] = (
        df.groupby("trip_id", observed=True)["arrival_seconds"]
        .transform(lambda x: x.interpolate(method="linear"))
    )
    df.dropna(subset=["arrival_seconds"], inplace=True)
    df.drop(columns=["arrival_time", "arrival_time_td"], inplace=True)
    logger.info("  After interpolation: %d rows | RAM: %.0f MB", len(df), ram_mb())

    log_step("STEP 3: Merging Stop Coordinates and Cleaning")
    stops = pd.read_csv(stops_path, sep=";", usecols=["stop_id", "stop_lat", "stop_lon"])
    stops["stop_id"] = stops["stop_id"].astype("int32")
    df = df.merge(stops, on="stop_id", how="left")
    del stops; gc.collect()

    for col in ["stop_lat", "stop_lon"]:
        df[col] = df[col].apply(clean_coord).astype("float32")
    df.dropna(subset=["stop_lat", "stop_lon"], inplace=True)
    df["arrival_seconds"] = df["arrival_seconds"].astype("float32")

    log_step("STEP 3b: Merging trips.csv (route_id, direction_id)")
    trips = pd.read_csv(
        trips_path,
        sep=";",
        usecols=["trip_id", "route_id", "direction_id"],
        dtype={"trip_id": "category", "route_id": "category", "direction_id": "category"},
    )
    df = df.merge(trips, on="trip_id", how="left")
    del trips; gc.collect()
    logger.info(
        "  After merge + clean: %d rows | route_id nulls: %d | RAM: %.0f MB",
        len(df), df["route_id"].isna().sum(), ram_mb(),
    )
    return df


# ---------------------------------------------------------------------------
# Speed Lags Engineering (mirrors Boston's engineer_base_and_speeds)
# ---------------------------------------------------------------------------

def engineer_base_and_speeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-stop speed from the previous stop (Haversine distance / elapsed
    time) and derive vel_lag_1, vel_lag_2, vel_lag_3 by shifting within each
    trip_id group.  Mirrors the Boston pipeline's engineer_base_and_speeds().

    Expects df to be sorted by (trip_id, stop_sequence) — already guaranteed
    by load_and_prepare_base().
    """
    log_step("STEP 3c: Speed Lag Engineering (vel_lag_1 / vel_lag_2 / vel_lag_3)")

    grp = df.groupby("trip_id", sort=False, observed=True)

    # Previous stop coordinates and arrival time (one position back within trip)
    df["prev_lat"]     = grp["stop_lat"].shift(1)
    df["prev_lon"]     = grp["stop_lon"].shift(1)
    df["prev_actual"]  = grp["arrival_seconds"].shift(1)

    # Haversine distance and elapsed time to compute instantaneous speed
    dist_m = calc_haversine_vectorized(
        df["stop_lat"], df["stop_lon"],
        df["prev_lat"], df["prev_lon"],
    )
    time_s = (df["arrival_seconds"] - df["prev_actual"]).astype("float32")

    df["vel_ms"] = (
        (dist_m / time_s)
        .replace([np.inf, -np.inf], np.nan)
        .astype("float32")
    )

    # Lag features: shift vel_ms within each trip
    df["vel_lag_1"] = grp["vel_ms"].shift(1).fillna(0.0).astype("float32")
    df["vel_lag_2"] = grp["vel_ms"].shift(2).fillna(0.0).astype("float32")
    df["vel_lag_3"] = grp["vel_ms"].shift(3).fillna(0.0).astype("float32")

    df.drop(columns=["prev_lat", "prev_lon", "prev_actual", "vel_ms"], inplace=True)
    gc.collect()

    logger.info(
        "  vel_lag_1 non-zero: %d | vel_lag_2: %d | vel_lag_3: %d | RAM: %.0f MB",
        (df["vel_lag_1"] != 0).sum(),
        (df["vel_lag_2"] != 0).sum(),
        (df["vel_lag_3"] != 0).sum(),
        ram_mb(),
    )
    return df


# ---------------------------------------------------------------------------
# Chronological Trip Split
# ---------------------------------------------------------------------------

def split_trips_chronologically(df: pd.DataFrame, train_frac: float = 0.80):
    """
    Divide trip_ids into train/test by the chronological order of their
    earliest stop. The oldest train_frac of trips form the train set;
    the newest (1 - train_frac) form the test set.

    This guarantees no trip crosses the boundary, eliminating all temporal
    data leakage between splits.
    """
    log_step("STEP 4: Chronological Train/Test Split (by trip_id)")

    trip_start = (
        df.groupby("trip_id", observed=True)["arrival_seconds"]
        .min()
        .sort_values()
    )
    n_trips   = len(trip_start)
    split_idx = int(n_trips * train_frac)
    train_trips = set(trip_start.iloc[:split_idx].index)
    test_trips  = set(trip_start.iloc[split_idx:].index)

    df_train = df[df["trip_id"].isin(train_trips)].reset_index(drop=True)
    df_test  = df[df["trip_id"].isin(test_trips)].reset_index(drop=True)

    logger.info(
        "  Trips → Train: %d | Test: %d",
        len(train_trips), len(test_trips),
    )
    logger.info(
        "  Rows → Train: %d | Test: %d",
        len(df_train), len(df_test),
    )
    return df_train, df_test


# ---------------------------------------------------------------------------
# Core: Chunking & Flush with Expanding Window
# ---------------------------------------------------------------------------

def run_split(
    df_split: pd.DataFrame,
    split_name: str,
    out_dir: Path,
    weather_df: pd.DataFrame,
    historical_state: dict,
) -> dict:
    """
    Generate O-D pairs for one split (train or test).

    For train: iterates k from 1 to MAX_LOOKAHEAD_STOPS (bounded horizon).
    For test:  iterates k starting at 1 with a while-True loop that only
               breaks when no more valid O-D pairs exist in the data,
               allowing k > MAX_LOOKAHEAD_STOPS for long routes (needed by
               the chained-inference evaluator).

    For each k-horizon:
      1. Shift destination rows by k positions.
      2. Apply trip-boundary and sequence-order masks.
      3. Compute spatial features and the expanding-window historical average
         (using only data with earlier arrival_seconds than the current row).
      4. Save the resulting chunk as part_{k:02d}.parquet.
      5. Delete chunk and call gc.collect() before the next iteration.

    Returns the updated historical_state dictionary.
    """
    is_test = split_name.lower() == "test"
    k_limit_label = "unbounded" if is_test else str(MAX_LOOKAHEAD_STOPS)
    log_step(
        f"STEP 5: O-D Expansion + Flush — {split_name.upper()} "
        f"(k=1..{k_limit_label})"
    )

    # Clean and create output directory
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Columns needed to look up destination rows after shift
    dest_cols = ["trip_id", "stop_id", "stop_sequence", "arrival_seconds", "stop_lat", "stop_lon"]  # origin-side extras handled separately
    df_dest = df_split[dest_cols].copy()

    # Key for expanding window groupby
    pair_key = ["stop_id", "dest_stop_id"]
    total_rows = 0

    # -----------------------------------------------------------------------
    # Dynamic loop:
    #   • train → bounded for-loop up to MAX_LOOKAHEAD_STOPS
    #   • test  → unbounded while-True; breaks only when no O-D pairs remain
    # -----------------------------------------------------------------------
    k = 0
    while True:
        k += 1

        # For train: stop after the model's training horizon
        if not is_test and k > MAX_LOOKAHEAD_STOPS:
            break

        df_shifted = df_dest.shift(-k)

        mask_trip = (df_split["trip_id"] == df_shifted["trip_id"]).fillna(False)
        mask_seq  = (df_shifted["stop_sequence"] > df_split["stop_sequence"]).fillna(False)
        mask      = mask_trip & mask_seq

        if not mask.any():
            if is_test:
                # No more physically-reachable O-D pairs — end of all routes
                logger.info(
                    "  k=%-2d | No valid O-D pairs — all routes exhausted. Stopping.",
                    k,
                )
                del df_shifted
                gc.collect()
                break
            else:
                logger.info("  k=%-2d | No valid O-D pairs — skipping.", k)
                del df_shifted
                gc.collect()
                continue

        # --- Build chunk with origin + destination columns ---
        chunk = df_split.loc[
            mask,
            [c for c in [
                "trip_id", "stop_id", "stop_sequence", "arrival_seconds",
                "stop_lat", "stop_lon",
                "route_id", "direction_id",
                "vel_lag_1", "vel_lag_2", "vel_lag_3",
            ] if c in df_split.columns]
        ].copy()

        chunk["dest_stop_id"]         = df_shifted.loc[mask, "stop_id"].values
        chunk["dest_lat"]             = df_shifted.loc[mask, "stop_lat"].values.astype("float32")
        chunk["dest_lon"]             = df_shifted.loc[mask, "stop_lon"].values.astype("float32")
        chunk["dest_arrival_seconds"] = df_shifted.loc[mask, "arrival_seconds"].values.astype("float32")
        chunk[TARGET]                 = (chunk["dest_arrival_seconds"] - chunk["arrival_seconds"]).astype("float32")

        del df_shifted
        gc.collect()

        # Filter implausible travel times
        chunk = chunk[
            (chunk[TARGET] > 0) & (chunk[TARGET] <= MAX_TRAVEL_TIME_S)
        ].copy()

        if len(chunk) == 0:
            logger.info("  k=%-2d | Zero rows after target filter — skipping.", k)
            del chunk
            gc.collect()
            continue

        chunk["dest_stop_id"] = chunk["dest_stop_id"].astype("int32")

        # Spatial features
        chunk["distancia_proyectada"] = calc_haversine_vectorized(
            chunk["stop_lat"], chunk["stop_lon"],
            chunk["dest_lat"], chunk["dest_lon"],
        )
        chunk["velocidad_tramo_m_s"] = (
            chunk["distancia_proyectada"] / chunk[TARGET]
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
        chunk["num_paradas_salto"] = np.int8(k)

        # ==============================================================
        # EXPANDING WINDOW — tiempo_promedio_historico (Anti-Leakage)
        # Sort by arrival_seconds ensures we only look at the past.
        # cumsum - current_value gives the sum of all strictly prior rows
        # within this chunk for the same (stop_id, dest_stop_id) pair.
        # ==============================================================
        chunk.sort_values("arrival_seconds", inplace=True)
        chunk.reset_index(drop=True, inplace=True)

        keys          = list(zip(chunk["stop_id"].astype(int), chunk["dest_stop_id"].astype(int)))
        suma_previa   = np.array([historical_state.get(ky, (0.0, 0))[0] for ky in keys], dtype="float64")
        conteo_previo = np.array([historical_state.get(ky, (0.0, 0))[1] for ky in keys], dtype="int64")

        chunk["_t64"]  = chunk[TARGET].astype("float64")
        suma_interna   = chunk.groupby(pair_key, observed=True)["_t64"].cumsum() - chunk["_t64"]
        conteo_interna = chunk.groupby(pair_key, observed=True).cumcount()

        total_sum   = suma_previa   + suma_interna.values
        total_count = conteo_previo + conteo_interna.values

        default_val = (chunk["distancia_proyectada"] / 5.0).values  # 5 m/s fallback speed
        denom = np.where(total_count == 0, 1, total_count)
        tph   = total_sum / denom
        chunk["tiempo_promedio_historico"] = np.where(
            total_count == 0, default_val, tph
        ).astype("float32")
        chunk.drop(columns=["_t64"], inplace=True)

        # Update global state with this chunk's aggregates
        aggs = (
            chunk.groupby(pair_key, observed=True)[TARGET]
            .agg(s="sum", c="count")
            .reset_index()
        )
        for row in aggs.itertuples(index=False):
            ky = (int(row.stop_id), int(row.dest_stop_id))
            old_s, old_c = historical_state.get(ky, (0.0, 0))
            historical_state[ky] = (old_s + row.s, old_c + row.c)
        del aggs
        # ==============================================================

        # Temporal features
        chunk["hora_del_dia"] = ((chunk["arrival_seconds"] // 3600) % 24).astype("int8")

        # Weather merge by hour-of-day
        if not weather_df.empty:
            chunk = chunk.merge(weather_df, on="hora_del_dia", how="left")
        else:
            chunk["temperature_2m"] = np.float32(0.0)
            chunk["precipitation"]  = np.float32(0.0)

        chunk["temperature_2m"] = chunk["temperature_2m"].fillna(0.0).astype("float32")
        chunk["precipitation"]  = chunk["precipitation"].fillna(0.0).astype("float32")

        # Categorical encoding
        for col in CATEGORICAL_FEATURES:
            if col in chunk.columns:
                chunk[col] = chunk[col].astype("category")

        # Select and enforce final column set
        out_cols = [c for c in FINAL_COLS if c in chunk.columns]
        chunk = chunk[out_cols]

        # --- FLUSH: save partition and free RAM ---
        part_path = out_dir / f"part_{k:02d}.parquet"
        chunk.to_parquet(part_path, index=False, compression="snappy")
        total_rows += len(chunk)
        logger.info(
            "  k=%-2d | %8d rows → %s | RAM: %.0f MB",
            k, len(chunk), part_path.name, ram_mb(),
        )
        del chunk
        gc.collect()

    del df_dest
    gc.collect()

    logger.info(
        "  ✅ %s done: %d total rows saved to %s/",
        split_name.upper(), total_rows, out_dir.name,
    )
    return historical_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    script_start = time.time()
    log_step("Istanbul Preprocessing — Leakage-Free + Chunking & Flush Architecture")

    # 1. Load and prepare the full base dataframe
    df = load_and_prepare_base()

    # 1b. Compute speed lag features (vel_lag_1/2/3) — mirrors Boston pipeline
    df = engineer_base_and_speeds(df)

    # 2. Fetch weather once (reused by both train and test splits)
    weather_df = fetch_istanbul_weather()

    # 3. Split trips chronologically — no trip crosses the boundary
    df_train, df_test = split_trips_chronologically(df, train_frac=0.80)
    del df
    gc.collect()

    # 4. TRAIN — build expanding-window O-D pairs and flush per k
    historical_state: dict = {}
    historical_state = run_split(df_train, "train", X_TRAIN_PARQUET, weather_df, historical_state)
    del df_train
    gc.collect()

    # Persist state so the test split inherits train's learned averages
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "wb") as fh:
        pickle.dump(historical_state, fh)
    logger.info("  historical_state persisted: %d unique O-D pairs.", len(historical_state))

    # 5. TEST — continues accumulating from the train historical state
    historical_state = run_split(df_test, "test", X_TEST_PARQUET, weather_df, historical_state)
    del df_test
    gc.collect()

    elapsed = (time.time() - script_start) / 60.0
    log_step(f"Istanbul Preprocessing Complete in {elapsed:.1f} minutes")


if __name__ == "__main__":
    main()
