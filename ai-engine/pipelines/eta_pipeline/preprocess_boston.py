"""
preprocess_data.py — Vectorized ETL and Feature Engineering for MBTA
=====================================================================
Industrial-grade pipeline. Uses vectorized joins (Pandas) to create
Origin-Destination pairs limited to a 10-stop horizon, optimizing
cache and reducing memory dependencies O(N^2) in loops.

Usage:
  python preprocess_data.py --split train
  python preprocess_data.py --split test
"""

import argparse
import gc
import glob
import logging
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

# Ensure ai-engine root (two levels up) is importable
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Import configuration
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, TARGET, CATEGORICAL_FEATURES, X_TRAIN_PARQUET, X_TEST_PARQUET

import numpy as np
import pandas as pd
import psutil
import holidays

import requests_cache
import openmeteo_requests
from retry_requests import retry

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logger Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess_data")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).parent
DATASETS_DIR = RAW_DATA_DIR
DIR_2024     = DATASETS_DIR / "MBTA_Bus_Arrival_Departure_Times_2024"
DIR_2025     = DATASETS_DIR / "MBTA_Bus_Arrival_Departure_Times_2025"
STOPS_CSV    = DATASETS_DIR / "Bus_Stops.csv"
STATE_FILE   = PROCESSED_DATA_DIR / "historical_state.pkl"

OUT_TRAIN    = X_TRAIN_PARQUET
OUT_TEST     = X_TEST_PARQUET

BOSTON_LAT   = 42.3601
BOSTON_LON   = -71.0589

WEATHER_START = "2024-01-01"
WEATHER_END   = "2025-12-31"

MAX_LOOKAHEAD_STOPS = 10     # Vital RAM limit
MAX_TRAVEL_TIME_S   = 7_200  # Max 2h between stops

MBTA_COLS = [
    "service_date", "route_id", "direction_id",
    "half_trip_id", "stop_id", "time_point_order",
    "scheduled", "actual",
]

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def ram_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def log_step(msg: str) -> None:
    logger.info("=" * 70)
    logger.info(msg)
    logger.info("=" * 70)

def calc_haversine_vectorized(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """Vectorized Haversine implementation for Pandas."""
    R = 6_371_000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c).astype("float32")


# ---------------------------------------------------------------------------
# Weather Fetch
# ---------------------------------------------------------------------------
def fetch_weather() -> pd.DataFrame:
    log_step("CONTEXT: WEATHER FETCH (Open-Meteo)")
    
    cache_session = requests_cache.CachedSession(
        cache_name=str(BASE_DIR / ".weather_cache_full"),
        backend="sqlite", expire_after=-1,
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude": BOSTON_LAT, "longitude": BOSTON_LON,
        "start_date": WEATHER_START, "end_date": WEATHER_END,
        "hourly": ["temperature_2m", "precipitation", "snowfall"],
        "timezone": "America/New_York",
    }
    
    logger.info("  Downloading weather from Open-Meteo...")
    try:
        response = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)[0]
        hourly = response.Hourly()

        dt_index = pd.date_range(
            start=pd.Timestamp(hourly.Time(), unit="s", tz="America/New_York"),
            end=pd.Timestamp(hourly.TimeEnd(), unit="s", tz="America/New_York"),
            freq=pd.tseries.frequencies.to_offset(f"{hourly.Interval()}s"),
            inclusive="left",
        )
        weather_df = pd.DataFrame({
            "datetime": dt_index,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy().astype("float32"),
            "precipitation": hourly.Variables(1).ValuesAsNumpy().astype("float32"),
            "snowfall": hourly.Variables(2).ValuesAsNumpy().astype("float32"),
        })
    except Exception as exc:
        logger.error("  ❌ Open-Meteo failed: %s", exc)
        return pd.DataFrame()
        
    weather_df["hour_key"] = weather_df["datetime"].dt.tz_convert("UTC").dt.floor("h")
    weather_df.drop(columns=["datetime"], inplace=True)
    return weather_df


# ---------------------------------------------------------------------------
# Load and Parse MBTA Data
# ---------------------------------------------------------------------------
def load_mbta_data(split: str) -> pd.DataFrame:
    log_step(f"STEP 1: MBTA DATA LOADING ({split.upper()})")
    
    if split == "train":
        files = sorted(glob.glob(str(DIR_2024 / "*2024*.csv")))
    else:
        files = sorted(glob.glob(str(DIR_2025 / "*2025*.csv")))

    if not files:
        raise FileNotFoundError(f"No files for split: {split}")

    logger.info("  Files found: %d", len(files))
    
    dtype_map = {
        "route_id": "category", "direction_id": "category",
        "half_trip_id": "string[pyarrow]", "stop_id": "category",
        "time_point_order": "Int32",
    }

    dfs = []
    # Use strict usecols
    usecols = ["service_date", "route_id", "direction_id", "half_trip_id", "stop_id", "time_point_order", "actual"]
    for f in files:
        dfs.append(pd.read_csv(f, usecols=usecols, dtype=dtype_map))
    df = pd.concat(dfs, ignore_index=True)
    del dfs; gc.collect()

    logger.info("  Parsing timestamps (epoch 1900)...")
    df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.tz_localize("America/New_York")
    raw = pd.to_datetime(df["actual"], errors="coerce")
    df["actual"] = df["service_date"] + (raw - pd.Timestamp("1900-01-01", tz="UTC"))
    df.dropna(subset=["actual"], inplace=True)
    df.drop(columns=["service_date"], inplace=True)

    logger.info("  %d rows loaded | RAM: %.0f MB", len(df), ram_mb())
    return df


def load_stops() -> pd.DataFrame:
    log_step("STEP 1B: COORDINATES LOADING (bus_stops.csv)")
    df = pd.read_csv(STOPS_CSV, usecols=["stop_id", "stop_lat", "stop_lon"])
    df["stop_id"] = df["stop_id"].astype("string[pyarrow]")
    df["stop_lat"] = df["stop_lat"].astype("float32")
    df["stop_lon"] = df["stop_lon"].astype("float32")
    df = df.drop_duplicates(subset=["stop_id"])
    return df


# ---------------------------------------------------------------------------
# Base Engineering and Speeds
# ---------------------------------------------------------------------------
def engineer_base_and_speeds(df: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    log_step("STEP 2: BASE ENGINEERING AND SPEEDS")
    
    # Merge with coordinates
    df["stop_id"] = df["stop_id"].astype("string[pyarrow]")
    df = df.merge(stops, on="stop_id", how="left")
    
    logger.info("  Chronologically sorting the trip...")
    df.sort_values(["half_trip_id", "time_point_order"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("  Calculating Vectorized Lags (v = d/t)...")
    grp = df.groupby("half_trip_id", sort=False)
    
    # 1 position back (immediate previous)
    df["prev_lat"] = grp["stop_lat"].shift(1)
    df["prev_lon"] = grp["stop_lon"].shift(1)
    df["prev_actual"] = grp["actual"].shift(1)
    
    # Distance to previous point
    dist_m = calc_haversine_vectorized(df["stop_lat"], df["stop_lon"], df["prev_lat"], df["prev_lon"])
    time_s = (df["actual"] - df["prev_actual"]).dt.total_seconds()
    
    # Add to DataFrame BEFORE shifting to avoid KeyErrors
    df["vel_ms"] = (dist_m / time_s).replace([np.inf, -np.inf], np.nan).astype("float32")
    
    df["vel_lag_1"] = df.groupby("half_trip_id", sort=False)["vel_ms"].shift(1).fillna(0.0).astype("float32")
    df["vel_lag_2"] = df.groupby("half_trip_id", sort=False)["vel_ms"].shift(2).fillna(0.0).astype("float32")
    df["vel_lag_3"] = df.groupby("half_trip_id", sort=False)["vel_ms"].shift(3).fillna(0.0).astype("float32")
    
    df.drop(columns=["prev_lat", "prev_lon", "prev_actual", "vel_ms"], inplace=True)
    gc.collect()
    
    return df


# ---------------------------------------------------------------------------
# Streaming Chunks & Feature Pipeline
# ---------------------------------------------------------------------------
def run_split(split: str, weather_df: pd.DataFrame, stops_df: pd.DataFrame) -> None:
    t0 = time.time()
    out_path = X_TRAIN_PARQUET if split == "train" else X_TEST_PARQUET
    
    # --- Historical Persistent State ---
    historical_state = {}
    if split == "test":
        if STATE_FILE.exists():
            with open(STATE_FILE, "rb") as f:
                historical_state = pickle.load(f)
            logger.info("  Loaded historical_state from disk with %d triplets.", len(historical_state))
        else:
            logger.warning("  Train historical_state not found. Starting test from scratch.")
            
    # Handle out_path as a partitions Directory
    import shutil
    if out_path.exists():
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    out_path.mkdir(exist_ok=True, parents=True)
    
    df         = load_mbta_data(split)
    
    df         = engineer_base_and_speeds(df, stops_df)
    
    # Pre-calculate holiday dates
    holiday_dates = set()
    for y in [2024, 2025]:
        holiday_dates.update(holidays.country_holidays("US", subdiv="MA", years=y).keys())

    log_step(f"STEP 3-5: STREAMING SHIFT-JOIN TO PARQUET (Max Horizon: {MAX_LOOKAHEAD_STOPS})")
    
    cols_o = [
        "half_trip_id", "time_point_order", "stop_id", "actual",
        "stop_lat", "stop_lon", "route_id", "direction_id",
        "vel_lag_1", "vel_lag_2", "vel_lag_3"
    ]
    
    cols_d = [
        "half_trip_id", "time_point_order", "stop_id", "actual",
        "stop_lat", "stop_lon"
    ]
    
    df_dest = df[cols_d]
    total_filas = 0
    triplete = ['route_id', 'stop_id_origen', 'stop_id_destino']

    for k in range(1, MAX_LOOKAHEAD_STOPS + 1):
        df_d_shifted = df_dest.shift(-k)

        mask_trip = (df["half_trip_id"] == df_d_shifted["half_trip_id"]).fillna(False)
        mask_dir = (df_d_shifted["time_point_order"] > df["time_point_order"]).fillna(False)
        mask_order = ((df_d_shifted["time_point_order"] - df["time_point_order"]) <= MAX_LOOKAHEAD_STOPS).fillna(False)
        
        mask = mask_trip & mask_dir & mask_order

        if mask.any():
            # Extract and rename only in validated partitions to save RAM
            chunk_o = df.loc[mask, cols_o].rename(columns={
                "time_point_order": "tpo_o", "stop_id": "stop_id_origen",
                "actual": "actual_o", "stop_lat": "lat_o", "stop_lon": "lon_o"
            }).reset_index(drop=True)
            
            chunk_d = df_d_shifted.loc[mask].drop(columns=["half_trip_id"]).rename(columns={
                "time_point_order": "tpo_d", "stop_id": "stop_id_destino",
                "actual": "actual_d", "stop_lat": "lat_d", "stop_lon": "lon_d"
            }).reset_index(drop=True)
            
            chunk = pd.concat([chunk_o, chunk_d], axis=1)
            
            # --- Fast Feature Engineering (Inside the chunk) ---
            chunk["tiempo_viaje_segundos"] = (chunk["actual_d"] - chunk["actual_o"]).dt.total_seconds().astype("float32")
            chunk["distancia_proyectada"]  = calc_haversine_vectorized(chunk["lat_o"], chunk["lon_o"], chunk["lat_d"], chunk["lon_d"])
            
            # Clean outliers with reduced RAM footprint
            mask_target = (chunk["tiempo_viaje_segundos"] > 0) & (chunk["tiempo_viaje_segundos"] <= MAX_TRAVEL_TIME_S)
            chunk = chunk[mask_target].copy()
            chunk.dropna(subset=["distancia_proyectada"], inplace=True)
            
            chunk["num_paradas_salto"] = (chunk["tpo_d"] - chunk["tpo_o"]).astype("float32")
            
            # ==== HISTORICAL AVERAGE TIME LOGIC (EXPANDING WINDOW) ====
            # Sort strictly by time to avoid looking into the future
            chunk.sort_values("actual_o", inplace=True)
            
            # 1. Retrieve the previous state from the global dictionary in a vectorized way
            keys = list(zip(chunk['route_id'], chunk['stop_id_origen'], chunk['stop_id_destino']))
            suma_previa = np.array([historical_state.get(ky, (0.0, 0))[0] for ky in keys])
            conteo_previo = np.array([historical_state.get(ky, (0.0, 0))[1] for ky in keys])
            
            # 2. Cumulative internal sums of the current chunk (ignoring its own row: -tiempo)
            suma_interna = chunk.groupby(triplete, observed=True)["tiempo_viaje_segundos"].cumsum() - chunk["tiempo_viaje_segundos"]
            conteo_interna = chunk.groupby(triplete, observed=True).cumcount()
            
            # 3. Consolidate means and assign
            total_sum = suma_previa + suma_interna.values
            total_count = conteo_previo + conteo_interna.values
            
            default_val = chunk["distancia_proyectada"] / 5.0
            denom = np.where(total_count == 0, 1, total_count)
            tph = total_sum / denom
            chunk["tiempo_promedio_historico"] = np.where(total_count == 0, default_val, tph).astype("float32")
            
            # 4. Update global state for future chunks or splits
            chunk_aggs = chunk.groupby(triplete, observed=True)["tiempo_viaje_segundos"].agg(sum_val='sum', count_val='count').reset_index()
            for row in chunk_aggs.itertuples(index=False):
                idx = (row.route_id, row.stop_id_origen, row.stop_id_destino)
                old_sum, old_count = historical_state.get(idx, (0.0, 0))
                historical_state[idx] = (old_sum + row.sum_val, old_count + row.count_val)
                
            # ================================================================
            
            chunk["hora_del_dia"] = chunk["actual_o"].dt.hour.astype("Int8")
            chunk["dia_semana"]   = chunk["actual_o"].dt.dayofweek.astype("Int8")
            chunk["mes"]          = chunk["actual_o"].dt.month.astype("Int8")
            chunk["hour_key"]     = chunk["actual_o"].dt.tz_convert("UTC").dt.floor("h")
            chunk["is_holiday"]   = chunk["actual_o"].dt.date.isin(holiday_dates).astype("float32")
            
            if weather_df is not None and not weather_df.empty:
                chunk = chunk.merge(weather_df, on="hour_key", how="left")
            else:
                chunk["temperature_2m"] = np.nan
                chunk["precipitation"] = 0.0
                chunk["snowfall"] = 0.0
                
            final_cols = [
                "tiempo_viaje_segundos",
                "hora_del_dia", "dia_semana", "mes",
                "temperature_2m", "precipitation", "snowfall",
                "is_holiday",
                "route_id", "direction_id", "stop_id_origen", "stop_id_destino",
                "distancia_proyectada",
                "vel_lag_1", "vel_lag_2", "vel_lag_3",
                "num_paradas_salto", "tiempo_promedio_historico"
            ]
            chunk = chunk[final_cols]
            total_filas += len(chunk)
            
            # Export as partition using compressed engine
            part_name = out_path / f"part_{k}.parquet"
            chunk.to_parquet(part_name, index=False, compression="snappy")
            logger.info("  Offset k=%-2d | Saved %8d rows to %s", k, len(chunk), part_name.name)

        del df_d_shifted
        gc.collect()

    del df_dest, df
    gc.collect()

    if split == "train":
        with open(STATE_FILE, "wb") as f:
            pickle.dump(historical_state, f)
        logger.info("  Saved historical_state to disk with %d triplets.", len(historical_state))
        
    elapsed = time.time() - t0
    logger.info("🎉 %s generated (Partitioned Dataset): %d rows in %.1f min", out_path.name, total_filas, elapsed / 60)

def main():
    parser = argparse.ArgumentParser(description="Advanced Vectorized ETL")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    # Load core dependencies only once
    weather_df = fetch_weather()
    if weather_df is not None and not weather_df.empty:
        weather_df.sort_values("hour_key", inplace=True)
        weather_df["precipitation"] = weather_df["precipitation"].fillna(0.0)
        weather_df["snowfall"] = weather_df["snowfall"].fillna(0.0)
    
    stops_df = load_stops()

    if args.split in ("train", "both"):
        run_split("train", weather_df, stops_df)
    if args.split in ("test", "both"):
        run_split("test", weather_df, stops_df)

if __name__ == "__main__":
    main()
