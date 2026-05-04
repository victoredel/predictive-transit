import os
import gc
import glob
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

# Force ACTIVE_DATASET to konya so config.py loads the correct paths
os.environ["ACTIVE_DATASET"] = "konya"
from config import (
    RAW_DATA_DIR, X_TRAIN_PARQUET, X_TEST_PARQUET, 
    MAX_LOOKAHEAD_STOPS, MAX_TRAVEL_TIME_S, MODELS_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess_konya")

STATE_FILE = MODELS_DIR / "historical_state_konya.pkl"

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
        f"?latitude=37.87&longitude=32.49"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,precipitation,snowfall"
        f"&timezone=UTC"
    )
    logger.info(f"Fetching Open-Meteo data for Konya: {start_date} to {end_date}")
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

def load_stops() -> pd.DataFrame:
    stops_file = RAW_DATA_DIR / "gtfs_static" / "gtfs_11_2025" / "stops.txt"
    logger.info(f"Loading stops from {stops_file}")
    stops_df = pd.read_csv(stops_file, usecols=["stop_id", "stop_lat", "stop_lon"], dtype={"stop_id": str})
    return stops_df

def load_and_prepare_base() -> pd.DataFrame:
    log_files = glob.glob(str(RAW_DATA_DIR / "arrival_logs" / "*.csv"))
    if not log_files:
        raise FileNotFoundError(f"No arrival logs found in {RAW_DATA_DIR / 'arrival_logs'}")
    
    logger.info(f"Found {len(log_files)} arrival log files.")
    dfs = []
    for f in log_files:
        chunk = pd.read_csv(
            f, sep=";", 
            dtype={"baslangic_durak_no": str, "bitis_durak_no": str, "arac_no": str, "ana_hat_no": str}
        )
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Loaded {len(df)} rows. Cleaning dates and formatting...")
    df["cikis_zaman"] = pd.to_datetime(df["cikis_zaman"], errors="coerce")
    df["varis_zaman"] = pd.to_datetime(df["varis_zaman"], errors="coerce")
    df = df.dropna(subset=["cikis_zaman", "varis_zaman"])
    
    df = df.rename(columns={
        "ana_hat_no": "route_id",
        "baslangic_durak_no": "stop_id_origen",
        "bitis_durak_no": "stop_id_destino"
    })
    
    df["stop_id_origen"] = df["stop_id_origen"].str.strip()
    df["stop_id_destino"] = df["stop_id_destino"].str.strip()
    df = df[(df["stop_id_origen"] != "") & (df["stop_id_destino"] != "")]
    
    logger.info("Collapsing high-density GPS telemetry into macro-stops...")
    df = df.sort_values(by=["arac_no", "cikis_zaman"]).reset_index(drop=True)
    
    # Identify contiguous blocks
    block_changed = (
        (df["arac_no"] != df["arac_no"].shift(1)) |
        (df["route_id"] != df["route_id"].shift(1)) |
        (df["stop_id_origen"] != df["stop_id_origen"].shift(1)) |
        (df["stop_id_destino"] != df["stop_id_destino"].shift(1))
    )
    df["block_id"] = block_changed.cumsum()
    
    # Aggregate contiguous GPS pings into single physical hops
    aggs = {
        "cikis_zaman": "min",
        "varis_zaman": "max",
        "arac_no": "first",
        "route_id": "first",
        "stop_id_origen": "first",
        "stop_id_destino": "first"
    }
    df = df.groupby("block_id", as_index=False).agg(aggs)
    df = df.drop(columns=["block_id"])
    
    df["tiempo_viaje_base"] = (df["varis_zaman"] - df["cikis_zaman"]).dt.total_seconds().astype(np.float32)
    # Filter noise: hops taking less than 30 seconds are likely bad data or overlapping waypoints
    df = df[df["tiempo_viaje_base"] >= 30]
    
    logger.info(f"Collapsed dataset down to {len(df)} physical trips between stops.")
    
    return df

def engineer_base_features(df: pd.DataFrame, stops_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging stops for base Haversine distance...")
    df = df.merge(stops_df.rename(columns={"stop_id": "stop_id_origen", "stop_lat": "lat_o", "stop_lon": "lon_o"}), on="stop_id_origen", how="inner")
    df = df.merge(stops_df.rename(columns={"stop_id": "stop_id_destino", "stop_lat": "lat_d", "stop_lon": "lon_d"}), on="stop_id_destino", how="inner")
    
    df["distancia_base"] = calc_haversine_vectorized(df["lat_o"], df["lon_o"], df["lat_d"], df["lon_d"]).astype(np.float32)
    df = df.dropna(subset=["distancia_base"])
    
    logger.info("Calculating velocity lags (vel_lag_1, 2, 3) per arac_no...")
    df = df.sort_values(by=["arac_no", "cikis_zaman"])
    df["speed_ms"] = df["distancia_base"] / df["tiempo_viaje_base"]
    df["speed_ms"] = df["speed_ms"].replace([np.inf, -np.inf], np.nan)
    
    df["vel_lag_1"] = df.groupby("arac_no")["speed_ms"].shift(1).astype(np.float32).fillna(0.0)
    df["vel_lag_2"] = df.groupby("arac_no")["speed_ms"].shift(2).astype(np.float32).fillna(0.0)
    df["vel_lag_3"] = df.groupby("arac_no")["speed_ms"].shift(3).astype(np.float32).fillna(0.0)
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
    
    logger.info("Reconstructing Trip ID Proxy...")
    df = df.sort_values(by=["arac_no", "cikis_zaman"]).reset_index(drop=True)
    arac_changed = df["arac_no"] != df["arac_no"].shift(1)
    route_changed = df["route_id"] != df["route_id"].shift(1)
    time_gap = (df["cikis_zaman"] - df["varis_zaman"].shift(1)).dt.total_seconds() > (30 * 60)
    
    new_trip_mask = arac_changed | route_changed | time_gap
    df["trip_id"] = new_trip_mask.cumsum()
    logger.info(f"Reconstructed {df['trip_id'].nunique()} unique trips.")
    
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
        "trip_id", "cikis_zaman", "stop_id_origen", "lat_o", "lon_o", "route_id",
        "vel_lag_1", "vel_lag_2", "vel_lag_3", "hora_del_dia", "dia_semana", "mes",
        "is_holiday", "temperature_2m", "precipitation", "snowfall"
    ]
    cols_d = [
        "trip_id", "varis_zaman", "stop_id_destino", "lat_d", "lon_d"
    ]
    
    df_origin = df_split[cols_o]
    df_dest = df_split[cols_d]
    triplete = ['route_id', 'stop_id_origen', 'stop_id_destino']
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
        
        # Expanding window logic exactly matching Boston
        chunk.sort_values("cikis_zaman", inplace=True)
        
        keys = list(zip(chunk['route_id'], chunk['stop_id_origen'], chunk['stop_id_destino']))
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
            idx = (row.route_id, row.stop_id_origen, row.stop_id_destino)
            old_sum, old_count = historical_state.get(idx, (0.0, 0))
            historical_state[idx] = (old_sum + row.sum_val, old_count + row.count_val)
            
        final_cols = [
            "tiempo_viaje_segundos",
            "hora_del_dia", "dia_semana", "mes",
            "temperature_2m", "precipitation", "snowfall",
            "is_holiday",
            "route_id", "stop_id_origen", "stop_id_destino",
            "distancia_proyectada",
            "vel_lag_1", "vel_lag_2", "vel_lag_3",
            "num_paradas_salto", "tiempo_promedio_historico"
        ]
        chunk = chunk[final_cols]
        for col in ["route_id", "stop_id_origen", "stop_id_destino"]:
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
    logger.info("KONYA PREPROCESSING PIPELINE (SHIFT-LOOP)")
    logger.info("="*70)
    
    df = load_and_prepare_base()
    
    min_date = df["cikis_zaman"].min().strftime("%Y-%m-%d")
    max_date = df["cikis_zaman"].max().strftime("%Y-%m-%d")
    weather_df = fetch_weather(min_date, max_date)
    
    stops_df = load_stops()
    
    df = engineer_base_features(df, stops_df, weather_df)
    
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
