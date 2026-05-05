import os
import gc
import logging
import time
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import holidays
import requests

# Ensure ai-engine root is in sys.path if run directly
import sys
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

os.environ["ACTIVE_DATASET"] = "izmir"
from config import (
    RAW_DATA_DIR, X_TRAIN_PARQUET, X_TEST_PARQUET, 
    MAX_LOOKAHEAD_STOPS, MAX_TRAVEL_TIME_S, MODELS_DIR, CATEGORICAL_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_izmir")

STATE_FILE = MODELS_DIR / "historical_state_izmir.pkl"
GTFS_DIR = RAW_DATA_DIR / "gtfs_static" / "bus-eshot-gtfs"

def calc_haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def fetch_weather(start_date: str, end_date: str):
    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude=38.42&longitude=27.14"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,snowfall"
        f"&timezone=UTC"
    )
    logger.info(f"Fetching Open-Meteo data for Izmir: {start_date} to {end_date}")
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["hourly"])
        df["hour_key"] = pd.to_datetime(df["time"]).dt.tz_localize("UTC")
        df.drop(columns=["time"], inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error fetching weather: {e}")
        return None

def parse_gtfs_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    """Parse GTFS string times like '25:30:00' correctly into datetime."""
    # Split into hours, minutes, seconds
    parts = time_series.str.split(':', expand=True).astype(float)
    td = pd.to_timedelta(parts[0], unit='h') + pd.to_timedelta(parts[1], unit='m') + pd.to_timedelta(parts[2], unit='s')
    return date_series + td

def load_and_unroll_gtfs() -> pd.DataFrame:
    logger.info("Loading GTFS Static Files from %s", GTFS_DIR)
    
    calendar_file = GTFS_DIR / "calendar.txt"
    trips_file = GTFS_DIR / "trips.txt"
    stop_times_file = GTFS_DIR / "stop_times.txt"
    stops_file = GTFS_DIR / "stops.txt"
    
    if not all([f.exists() for f in [calendar_file, trips_file, stop_times_file, stops_file]]):
        raise FileNotFoundError(f"Missing essential GTFS files in {GTFS_DIR}")
        
    cal = pd.read_csv(calendar_file, dtype={"service_id": str})
    # Parse dates
    cal["start_date"] = pd.to_datetime(cal["start_date"].astype(str), format="%Y%m%d")
    cal["end_date"] = pd.to_datetime(cal["end_date"].astype(str), format="%Y%m%d")
    
    # 7-day timeline simulation to protect RAM
    start_sim = cal["start_date"].min()
    end_sim = start_sim + pd.Timedelta(days=7)
    logger.info(f"Simulating 7 days of GTFS data: {start_sim.date()} to {end_sim.date()}")
    
    dates = pd.date_range(start=start_sim, end=end_sim, freq='D')
    date_df = pd.DataFrame({"sim_date": dates})
    date_df["day_of_week"] = date_df["sim_date"].dt.day_name().str.lower()
    
    logger.info("Matching active service_ids for generated dates...")
    active_services = []
    for _, row in date_df.iterrows():
        d = row["sim_date"]
        dow = row["day_of_week"]
        # Find active services
        active = cal[(cal["start_date"] <= d) & (cal["end_date"] >= d) & (cal[dow] == 1)]
        for s_id in active["service_id"]:
            active_services.append({"sim_date": d, "service_id": s_id})
            
    service_map = pd.DataFrame(active_services)
    if service_map.empty:
        raise ValueError("No active services found for the simulated date range!")
        
    logger.info("Loading trips and stop_times...")
    trips = pd.read_csv(trips_file, usecols=["route_id", "service_id", "trip_id", "direction_id"], dtype=str)
    stop_times = pd.read_csv(stop_times_file, usecols=["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"], dtype=str)
    stops = pd.read_csv(stops_file, usecols=["stop_id", "stop_lat", "stop_lon"], dtype=str)
    
    logger.info("Merging and flattening unrolled history...")
    # Map trips to dates
    unrolled_trips = service_map.merge(trips, on="service_id", how="inner")
    
    # Map stop_times to unrolled trips
    df = unrolled_trips.merge(stop_times, on="trip_id", how="inner")
    
    logger.info(f"Generated {len(df):,} unrolled stop_times rows. Parsing datetimes...")
    df["stop_sequence"] = df["stop_sequence"].astype(int)
    
    # Composite simulated trip id
    df["simulated_trip_id"] = df["trip_id"] + "_" + df["sim_date"].dt.strftime("%Y%m%d")
    
    # Parse GTFS >24h times
    df["cikis_zaman"] = parse_gtfs_time(df["sim_date"], df["departure_time"])
    df["arrival_time_dt"] = parse_gtfs_time(df["sim_date"], df["arrival_time"])
    
    # Merge stops for coordinates
    df = df.merge(stops, on="stop_id", how="left")
    df["stop_lat"] = df["stop_lat"].astype(float)
    df["stop_lon"] = df["stop_lon"].astype(float)
    
    # Drop unneeded cols to save RAM
    del unrolled_trips, trips, stop_times, stops, service_map, cal, date_df
    gc.collect()
    
    logger.info("Building Origin-Destination physical macro-legs...")
    df = df.sort_values(by=["simulated_trip_id", "stop_sequence"]).reset_index(drop=True)
    
    # Same trip check
    same_trip = df["simulated_trip_id"] == df["simulated_trip_id"].shift(-1)
    
    df["stop_id_destino"] = df["stop_id"].shift(-1)
    df["varis_zaman"] = df["arrival_time_dt"].shift(-1)
    df["lat_d"] = df["stop_lat"].shift(-1)
    df["lon_d"] = df["stop_lon"].shift(-1)
    
    # Keep only valid transitions
    df = df[same_trip].copy()
    
    df = df.rename(columns={"stop_id": "stop_id_origen", "stop_lat": "lat_o", "stop_lon": "lon_o"})
    
    df["tiempo_viaje_base"] = (df["varis_zaman"] - df["cikis_zaman"]).dt.total_seconds().astype(np.float32)
    # Filter 0 or negative times
    df = df[df["tiempo_viaje_base"] > 0]
    
    logger.info(f"Unrolled {len(df):,} physical trips between scheduled stops.")
    
    # Keep only needed columns
    keep_cols = [
        "simulated_trip_id", "route_id", "direction_id", 
        "stop_id_origen", "stop_id_destino", "cikis_zaman", "varis_zaman",
        "lat_o", "lon_o", "lat_d", "lon_d", "tiempo_viaje_base"
    ]
    df = df[keep_cols]
    gc.collect()
    
    return df

def engineer_base_features(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("PHASE 3: Engineering Base Features...")
    
    df["distancia_base"] = calc_haversine_vectorized(df["lat_o"], df["lon_o"], df["lat_d"], df["lon_d"]).astype(np.float32)
    df = df.dropna(subset=["distancia_base"])
    
    logger.info("Calculating scheduled velocity lags (vel_lag_1, 2, 3) per simulated trip...")
    df = df.sort_values(by=["simulated_trip_id", "cikis_zaman"])
    df["speed_ms"] = df["distancia_base"] / df["tiempo_viaje_base"]
    df["speed_ms"] = df["speed_ms"].replace([np.inf, -np.inf], np.nan)
    
    df["vel_lag_1"] = df.groupby("simulated_trip_id")["speed_ms"].shift(1).astype(np.float32).fillna(0.0)
    df["vel_lag_2"] = df.groupby("simulated_trip_id")["speed_ms"].shift(2).astype(np.float32).fillna(0.0)
    df["vel_lag_3"] = df.groupby("simulated_trip_id")["speed_ms"].shift(3).astype(np.float32).fillna(0.0)
    df = df.drop(columns=["speed_ms", "distancia_base", "tiempo_viaje_base"])
    
    logger.info("Engineering temporal and weather features...")
    df["hora_del_dia"] = df["cikis_zaman"].dt.hour.astype("Int8")
    df["dia_semana"] = df["cikis_zaman"].dt.dayofweek.astype("Int8")
    df["mes"] = df["cikis_zaman"].dt.month.astype("Int8")
    
    years = list(df["cikis_zaman"].dt.year.unique())
    tr_holidays = holidays.country_holidays("TR", years=years)
    df["is_holiday"] = df["cikis_zaman"].dt.date.isin(tr_holidays).astype(np.float32)
    
    df["hour_key"] = df["cikis_zaman"].dt.tz_localize("UTC").dt.floor("h")
    if weather_df is not None:
        df = df.merge(weather_df, on="hour_key", how="left")
    else:
        df["temperature_2m"] = np.nan
        df["precipitation"] = 0.0
        df["snowfall"] = 0.0
        
    df["temperature_2m"] = df["temperature_2m"].ffill().bfill().astype(np.float32)
    df["precipitation"] = df["precipitation"].fillna(0.0).astype(np.float32)
    df["snowfall"] = df["snowfall"].fillna(0.0).astype(np.float32)
    
    # In GTFS unrolling, the proxy trip ID is simply the simulated_trip_id
    # We assign numeric IDs for the shift loop
    df["trip_id"] = pd.factorize(df["simulated_trip_id"])[0]
    df = df.drop(columns=["simulated_trip_id", "hour_key"])
    
    return df

def run_split(split_name: str, df_split: pd.DataFrame):
    t0 = time.time()
    out_dir = X_TRAIN_PARQUET if split_name == "train" else X_TEST_PARQUET
    
    import shutil
    if out_dir.exists():
        if out_dir.is_dir(): shutil.rmtree(out_dir)
        else: out_dir.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    is_test = (split_name == "test")
    k_limit_label = "unbounded" if is_test else str(MAX_LOOKAHEAD_STOPS)
    logger.info("="*70)
    logger.info(f"STREAMING SHIFT-JOIN TO PARQUET (split={split_name.upper()}, k=1..{k_limit_label})")
    
    historical_state = {}
    if is_test:
        if STATE_FILE.exists():
            with open(STATE_FILE, "rb") as f:
                historical_state = pickle.load(f)
            logger.info(f"  Loaded historical_state with {len(historical_state)} triplets.")
        else:
            logger.warning("  Train historical_state not found.")

    cols_o = [
        "trip_id", "cikis_zaman", "stop_id_origen", "lat_o", "lon_o", "route_id", "direction_id",
        "vel_lag_1", "vel_lag_2", "vel_lag_3", "hora_del_dia", "dia_semana", "mes",
        "is_holiday", "temperature_2m", "precipitation", "snowfall"
    ]
    cols_d = [
        "trip_id", "varis_zaman", "stop_id_destino", "lat_d", "lon_d"
    ]
    
    df_origin = df_split[cols_o]
    df_dest = df_split[cols_d]
    triplete = ['route_id', 'direction_id', 'stop_id_origen', 'stop_id_destino']
    total_rows = 0
    
    k = 0
    while True:
        k += 1
        if not is_test and k > MAX_LOOKAHEAD_STOPS:
            break
            
        df_d_shifted = df_dest.shift(-k)
        mask_trip = (df_origin["trip_id"] == df_d_shifted["trip_id"]).fillna(False)
        
        if not mask_trip.any():
            if is_test:
                logger.info(f"  k={k:<2d} | No valid trips remaining. Stopping.")
                del df_d_shifted
                gc.collect()
                break
            else:
                del df_d_shifted
                gc.collect()
                continue
                
        chunk_o = df_origin.loc[mask_trip].reset_index(drop=True)
        chunk_d = df_d_shifted.loc[mask_trip].drop(columns=["trip_id"]).reset_index(drop=True)
        
        chunk = pd.concat([chunk_o, chunk_d], axis=1)
        
        chunk["tiempo_viaje_segundos"] = (chunk["varis_zaman"] - chunk["cikis_zaman"]).dt.total_seconds().astype(np.float32)
        chunk["distancia_proyectada"] = calc_haversine_vectorized(chunk["lat_o"], chunk["lon_o"], chunk["lat_d"], chunk["lon_d"]).astype(np.float32)
        
        mask_target = (chunk["tiempo_viaje_segundos"] > 0) & (chunk["tiempo_viaje_segundos"] <= MAX_TRAVEL_TIME_S)
        chunk = chunk[mask_target].copy()
        chunk.dropna(subset=["distancia_proyectada"], inplace=True)
        
        if chunk.empty:
            if is_test:
                logger.info(f"  k={k:<2d} | All remaining trips exceed {MAX_TRAVEL_TIME_S}s. Stopping.")
                del df_d_shifted, chunk_o, chunk_d, chunk
                gc.collect()
                break
            else:
                del df_d_shifted, chunk_o, chunk_d, chunk
                gc.collect()
                continue
        
        chunk["num_paradas_salto"] = np.float32(k)
        
        chunk.sort_values("cikis_zaman", inplace=True)
        
        keys = list(zip(chunk['route_id'], chunk['direction_id'], chunk['stop_id_origen'], chunk['stop_id_destino']))
        suma_previa = np.array([historical_state.get(ky, (0.0, 0))[0] for ky in keys])
        conteo_previo = np.array([historical_state.get(ky, (0.0, 0))[1] for ky in keys])
        
        suma_interna = chunk.groupby(triplete, observed=True)["tiempo_viaje_segundos"].cumsum() - chunk["tiempo_viaje_segundos"]
        conteo_interna = chunk.groupby(triplete, observed=True).cumcount()
        
        total_sum = suma_previa + suma_interna.values
        total_count = conteo_previo + conteo_interna.values
        
        default_val = chunk["distancia_proyectada"] / 5.0
        denom = np.where(total_count == 0, 1, total_count)
        tph = total_sum / denom
        chunk["tiempo_promedio_historico"] = np.where(total_count == 0, default_val, tph).astype(np.float32)
        
        chunk_aggs = chunk.groupby(triplete, observed=True)["tiempo_viaje_segundos"].agg(sum_val='sum', count_val='count').reset_index()
        for row in chunk_aggs.itertuples(index=False):
            idx = (row.route_id, row.direction_id, row.stop_id_origen, row.stop_id_destino)
            old_sum, old_count = historical_state.get(idx, (0.0, 0))
            historical_state[idx] = (old_sum + row.sum_val, old_count + row.count_val)
            
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
        for col in CATEGORICAL_FEATURES:
            chunk[col] = chunk[col].astype("category")
            
        total_rows += len(chunk)
        
        part_name = out_dir / f"part_{k}.parquet"
        chunk.to_parquet(part_name, index=False, compression="snappy")
        logger.info(f"  k={k:<2d} | Saved {len(chunk):8d} rows to {part_name.name}")
        
        del chunk, chunk_o, chunk_d, df_d_shifted, chunk_aggs
        gc.collect()
        
    del df_origin, df_dest, df_split
    gc.collect()

    if split_name == "train":
        with open(STATE_FILE, "wb") as f:
            pickle.dump(historical_state, f)
        logger.info(f"  Saved historical_state with {len(historical_state)} triplets.")
        
    logger.info(f"🎉 {split_name.upper()} generated: {total_rows:,} rows in {(time.time() - t0)/60:.1f} min")

def main():
    t0 = time.time()
    logger.info("="*70)
    logger.info("IZMIR PREPROCESSING PIPELINE (GTFS UNROLLING)")
    logger.info("="*70)
    
    df = load_and_unroll_gtfs()
    
    min_date = df["cikis_zaman"].min().strftime("%Y-%m-%d")
    max_date = df["cikis_zaman"].max().strftime("%Y-%m-%d")
    weather_df = fetch_weather(min_date, max_date)
    
    df = engineer_base_features(df, weather_df)
    
    logger.info("PHASE 4: Shift Loop & Expanding Window")
    logger.info("Splitting 80/20 chronologically into Train and Test partitions before shifting...")
    df = df.sort_values(by="cikis_zaman").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    del df
    gc.collect()
    
    logger.info("Sorting partitions by trip_id to align shift sequences...")
    df_train = df_train.sort_values(by=["trip_id", "cikis_zaman"]).reset_index(drop=True)
    df_test = df_test.sort_values(by=["trip_id", "cikis_zaman"]).reset_index(drop=True)
    
    run_split("train", df_train)
    run_split("test", df_test)
    
    logger.info(f"✨ Full Pipeline Complete in {(time.time() - t0)/60:.1f} minutes")

if __name__ == "__main__":
    main()
