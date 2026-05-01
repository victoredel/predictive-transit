"""
preprocess_istanbul.py — Production Preprocessing script for Istanbul
=====================================================================
Modular script to read stop times, interpolate, merge with stops,
and prepare train/test splits in Parquet format.
"""

import sys
import os
import time
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import psutil
import requests_cache
import openmeteo_requests
from datetime import datetime, timedelta

# Ensure ai-engine root is in sys.path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    TARGET, 
    CATEGORICAL_FEATURES, 
    X_TRAIN_PARQUET, 
    X_TEST_PARQUET
)

# Logger Config
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_istanbul")

def ram_mb():
    """Return the current memory usage of this process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def log_step(msg):
    """Print a visual separator, a log message, and the current RAM usage."""
    logger.info("=" * 60)
    logger.info(f"=== {msg} ===")
    logger.info(f"=== Current RAM: {ram_mb():.2f} MB ===")
    logger.info("=" * 60)

def calc_haversine_vectorized(lat1, lon1, lat2, lon2):
    """Vectorized Haversine implementation for distance in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return (R * c).astype("float32")

def fetch_istanbul_weather():
    """Fetch historical weather for Istanbul (Open-Meteo)."""
    # Coordinates for Istanbul
    ISTANBUL_LAT = 41.0082
    ISTANBUL_LON = 28.9784
    
    cache_session = requests_cache.CachedSession('.weather_cache_istanbul', expire_after=-1)
    om = openmeteo_requests.Client(session=cache_session)

    # Simulate a date (7 days ago) to get historical data
    target_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    params = {
        "latitude": ISTANBUL_LAT,
        "longitude": ISTANBUL_LON,
        "start_date": target_date,
        "end_date": target_date,
        "hourly": ["temperature_2m", "precipitation"]
    }
    
    try:
        logger.info(f"Fetching weather for Istanbul on {target_date}...")
        responses = om.weather_api("https://archive-api.open-meteo.com/v1/archive", params=params)
        response = responses[0]
        hourly = response.Hourly()
        
        # Extract hourly data
        temp = hourly.Variables(0).ValuesAsNumpy()
        precip = hourly.Variables(1).ValuesAsNumpy()
        
        weather_df = pd.DataFrame({
            "hora_del_dia": np.arange(len(temp)),
            "temperature_2m": temp,
            "precipitation": precip
        })
        return weather_df
    except Exception as e:
        logger.error(f"Failed to fetch weather: {e}")
        return pd.DataFrame()

def parse_time_string(time_str):
    if pd.isna(time_str) or time_str == '':
        return pd.NaT
    try:
        hours, minutes, seconds = map(int, str(time_str).strip().split(':'))
        return pd.Timedelta(hours=hours, minutes=minutes, seconds=seconds)
    except Exception:
        return pd.NaT

def main():
    script_start_time = time.time()
    log_step("Starting Istanbul Data Preprocessing")

    # Define paths based on RAW_DATA_DIR
    stop_times_path = RAW_DATA_DIR / "gtfs_iett" / "stop_times" / "stop_times.txt"
    stops_path = RAW_DATA_DIR / "gtfs_iett" / "stops.csv"

    # 1. Load Data
    try:
        logger.warning(f"Loading stop_times file from {stop_times_path}...")
        logger.info("Capped read at 1M rows to allow for 10-stop expansion without OOM.")
        load_start_time = time.time()
        
        stop_times_dtypes = {
            'trip_id': 'category',
            'stop_id': 'int32',
            'stop_sequence': 'int32'
        }
        df = pd.read_csv(
            stop_times_path, 
            sep=',', 
            usecols=['trip_id', 'stop_id', 'stop_sequence', 'arrival_time'],
            dtype=stop_times_dtypes,
            nrows=500000
        )
        
        load_duration = time.time() - load_start_time
        logger.info(f"Successfully loaded stop_times.csv in {load_duration:.2f} seconds.")
        logger.info(f"Current RAM after loading stop_times: {ram_mb():.2f} MB.")
        
        logger.info(f"Loading stops from {stops_path}...")
        stops = pd.read_csv(stops_path, sep=';', usecols=['stop_id', 'stop_lat', 'stop_lon'])
        stops['stop_id'] = stops['stop_id'].astype('int32')
        
    except FileNotFoundError as e:
        logger.error(f"Missing raw CSV files: {e}")
        return

    # 2. Interpolation & Target Calculation Preparation
    log_step("Parsing and Interpolating Times")
    df['arrival_time_td'] = df['arrival_time'].apply(parse_time_string)
    df = df.sort_values(by=['trip_id', 'stop_sequence']).reset_index(drop=True)
    df['arrival_seconds'] = df['arrival_time_td'].dt.total_seconds()
    
    interp_start_time = time.time()
    logger.info("Applying linear interpolation to missing arrival times...")
    df['arrival_seconds'] = df.groupby('trip_id', observed=False)['arrival_seconds'].transform(lambda x: x.interpolate(method='linear'))
    interp_duration = time.time() - interp_start_time
    logger.info(f"Interpolation completed in {interp_duration:.2f} seconds.")
    
    # Drop rows that couldn't be interpolated
    df = df.dropna(subset=['arrival_seconds'])

    # 3. Merge with stops
    log_step("Merging with Stops Data")
    logger.info(f"Stop times shape: {df.shape}")
    df = df.merge(stops, on="stop_id", how="left")
    logger.info(f"Merged shape: {df.shape}")

    logger.info("Cleaning coordinate columns...")
    def clean_coord(val):
        if pd.isna(val): return np.nan
        s = str(val)
        digits = "".join(filter(str.isdigit, s))
        if len(digits) < 2: return np.nan
        try:
            return float(digits[:2] + "." + digits[2:])
        except:
            return np.nan

    for col in ['stop_lat', 'stop_lon']:
        if col in df.columns:
            df[col] = df[col].apply(clean_coord)
    
    df = df.dropna(subset=['stop_lat', 'stop_lon'])

    # 4. Lookahead Expansion (Origin-Destination Pairs)
    log_step("Lookahead Expansion: Generating O-D Pairs (k=1..10)")
    MAX_LOOKAHEAD_STOPS = 10
    chunks = []
    
    for k in range(1, MAX_LOOKAHEAD_STOPS + 1):
        df_shifted = df.shift(-k)
        
        # Mask: Same trip and sequence difference is k
        mask_trip = (df['trip_id'] == df_shifted['trip_id'])
        mask_seq = (df_shifted['stop_sequence'] > df['stop_sequence'])
        
        mask = mask_trip & mask_seq
        
        if mask.any():
            chunk = df[mask].copy()
            # Destinations
            chunk['dest_stop_id'] = df_shifted.loc[mask, 'stop_id']
            chunk['dest_lat'] = df_shifted.loc[mask, 'stop_lat']
            chunk['dest_lon'] = df_shifted.loc[mask, 'stop_lon']
            chunk['dest_arrival_seconds'] = df_shifted.loc[mask, 'arrival_seconds']
            
            # Target: travel time
            chunk[TARGET] = chunk['dest_arrival_seconds'] - chunk['arrival_seconds']
            
            # Filter valid targets
            chunk = chunk[chunk[TARGET] > 0].copy()
            
            # Spatial Features for this k-hop
            chunk['distancia_proyectada'] = calc_haversine_vectorized(
                chunk['stop_lat'], chunk['stop_lon'], 
                chunk['dest_lat'], chunk['dest_lon']
            )
            
            # Velocity
            chunk['velocidad_tramo_m_s'] = chunk['distancia_proyectada'] / chunk[TARGET]
            chunk['velocidad_tramo_m_s'] = chunk['velocidad_tramo_m_s'].replace([np.inf, -np.inf], np.nan).fillna(0)
            
            chunk['num_paradas_salto'] = k
            
            # Memory optimization: force float32 early
            for col in ['stop_lat', 'stop_lon', 'arrival_seconds', 'dest_lat', 'dest_lon', 'dest_arrival_seconds', TARGET, 'distancia_proyectada', 'velocidad_tramo_m_s']:
                if col in chunk.columns:
                    chunk[col] = chunk[col].astype('float32')
            
            chunks.append(chunk)
            del df_shifted
            import gc
            gc.collect()
            
    if not chunks:
        logger.error("No O-D pairs generated!")
        return
        
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    logger.info(f"Expanded O-D pairs shape: {df.shape}")

    # 5. Feature Engineering: Temporal & Weather
    log_step("Feature Engineering: Temporal and Weather")
    
    # A. Temporal Features
    df['hora_del_dia'] = ((df['arrival_seconds'] // 3600) % 24).astype('int8')
    
    # B. Weather Integration
    weather_df = fetch_istanbul_weather()
    if not weather_df.empty:
        df = df.merge(weather_df, on="hora_del_dia", how="left")
    else:
        df['temperature_2m'] = 0.0
        df['precipitation'] = 0.0
        
    # Final Cleanup
    for col in ["distancia_proyectada", "temperature_2m", "precipitation", "velocidad_tramo_m_s"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 6. Formatting
    new_features = ["hora_del_dia", "distancia_proyectada", "velocidad_tramo_m_s", "temperature_2m", "precipitation", "num_paradas_salto"]
    features = ['trip_id', 'stop_id', 'stop_sequence', 'stop_lat', 'stop_lon', 'arrival_seconds'] + new_features
    df = df[features + [TARGET]].copy()

    continuous = ['stop_lat', 'stop_lon', 'arrival_seconds', TARGET] + new_features
    for col in continuous:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("float32")
    
    df = df.dropna(subset=[TARGET]) 
    logger.info(f"Final shape before split: {df.shape}")

    # Categorical Types
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 7. Train/Test Split
    log_step("Performing Train/Test Split")
    df = df.sort_values(by=['arrival_seconds'])
    split_idx = int(len(df) * 0.8)
    
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # 8. Save as Parquet
    log_step("Saving Parquet Files")
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df.to_parquet(X_TRAIN_PARQUET, index=False)
    test_df.to_parquet(X_TEST_PARQUET, index=False)
    logger.info(f"Istanbul Preprocessing Complete! Train: {len(train_df)} rows, Test: {len(test_df)} rows.")
    
    total_time_minutes = (time.time() - script_start_time) / 60.0
    log_step(f"Total Execution Time: {total_time_minutes:.2f} minutes")

if __name__ == "__main__":
    main()
