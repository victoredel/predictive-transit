"""
train_model.py — Entrenamiento Out-of-Time XGBoost (Año Completo 2024)
=======================================================================
Estrategia de Validación: Cross-Year (Out-of-Time)
  - ENTRENAMIENTO: Todo el año 2024 (12 meses, ~3 GB de datos).
  - EVALUACIÓN:    Todo el año 2025 (ejecutar evaluate_model.py).

Arquitectura del Pipeline:
  1. fetch_weather_history()  → Open-Meteo Historical API (2024-01-01 → 2024-12-31)
  2. load_mbta_data()          → glob "*2024*.csv" sobre datasets/MBTA.../
  3. merge_weather()           → pd.merge by (year, month, day, hour)
  4. engineer_features()       → shift(-1) Node-to-Node + mes del año
  5. clean_data()              → outliers + nulos con logging exhaustivo
  6. prepare_features()        → dtype float32 / category para XGBoost
  7. train()                   → XGBRegressor + early stopping (train/val 80/20)
  8. save_model()              → transit_xgboost_model.json

Observabilidad:
  - Uso de RAM (psutil) en cada etapa.
  - Filas retenidas/descartadas en cada filtro.
  - RMSE de entrenamiento y validación cada 50 árboles.
  - Tiempo de ejecución de cada paso.

Optimización de Memoria:
  - dtype_map al cargar CSV (evita inferencia y down-casting manual).
  - pd.concat con lista y GC explícito entre archivos.
  - Eliminación del DataFrame fuente antes del entrenamiento.

Uso:
  python train_model.py

Salida:
  ai-engine/transit_xgboost_model.json
"""

import gc
import glob
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import requests_cache
import xgboost as xgb
from retry_requests import retry
import openmeteo_requests
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logger — ISO 8601, nivel INFO, stdout
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Constantes de Configuración
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent
DATA_DIR  = BASE_DIR / "datasets" / "MBTA_Bus_Arrival_Departure_Times_2024"
MODEL_PATH = BASE_DIR / "transit_xgboost_model.json"

# Coordenadas de Boston (Logan Airport / City Center)
BOSTON_LAT = 42.3601
BOSTON_LON = -71.0589

# Ventana temporal de entrenamiento (año completo 2024)
TRAIN_START = "2024-01-01"
TRAIN_END   = "2024-12-31"

# Patrón glob para seleccionar SOLO archivos de 2024
GLOB_PATTERN = "*2024*.csv"

# Columnas mínimas requeridas del CSV MBTA
REQUIRED_COLS = [
    "service_date", "route_id", "direction_id",
    "half_trip_id", "stop_id", "time_point_order",
    "scheduled", "actual",
]

# Límites de sanity check para el target
MIN_TRAVEL_TIME_S = 0
MAX_TRAVEL_TIME_S = 3_600   # 1 hora máximo entre paradas

# Features categóricas (XGBoost native categorical)
CATEGORICAL_FEATURES = ["route_id", "direction_id", "stop_id", "next_stop_id"]

# Features numéricas (escalares)
NUMERIC_FEATURES = ["hora_del_dia", "dia_semana", "mes",
                    "temperature_2m", "precipitation", "snowfall"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET       = "tiempo_viaje_segundos"

# Hiperparámetros XGBoost optimizados para dataset grande (~12M filas)
XGB_PARAMS = {
    "n_estimators":          600,
    "early_stopping_rounds": 30,
    "max_depth":             7,
    "learning_rate":         0.05,
    "subsample":             0.80,
    "colsample_bytree":      0.80,
    "min_child_weight":      10,      # más conservador para datasets grandes
    "reg_alpha":             0.1,     # L1
    "reg_lambda":            1.0,     # L2
    "objective":             "reg:squarederror",
    "tree_method":           "hist",  # más rápido en CPU
    "enable_categorical":    True,
    "random_state":          42,
    "eval_metric":           "rmse",
    "n_jobs":                -1,      # todos los cores
    "device":                "cpu",
}


# ---------------------------------------------------------------------------
# Utilidades de Observabilidad
# ---------------------------------------------------------------------------

def log_memory_usage(context: str) -> None:
    """Loguea el uso actual de RAM (RSS) del proceso en MB."""
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 ** 2)
    logger.info("🧠 RAM [%s]: %.1f MB", context, rss_mb)


def log_dataframe_stats(df: pd.DataFrame, label: str) -> None:
    """Loguea shape y consumo de RAM del DataFrame."""
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(
        "📊 [%s] shape=%s | memoria_df=%.1f MB",
        label, df.shape, mem_mb,
    )


def log_drop(before: int, after: int, reason: str) -> None:
    """Loguea exactamente cuántas filas fueron descartadas y el porcentaje."""
    dropped = before - after
    pct = (dropped / before * 100) if before > 0 else 0.0
    logger.info(
        "🗑️  Filtro [%s]: %d filas eliminadas (%.2f%%) | quedan %d",
        reason, dropped, pct, after,
    )


# ---------------------------------------------------------------------------
# PASO 1: Fetch del Clima Histórico (Open-Meteo API)
# ---------------------------------------------------------------------------

def fetch_weather_history() -> pd.DataFrame:
    """
    Descarga datos climáticos horarios de Boston para el año 2024 completo
    usando la API Historical de Open-Meteo (gratuita, sin key).

    Variables extraídas (horarias):
      - temperature_2m : Temperatura del aire a 2m (°C)
      - precipitation  : Precipitación total (mm/h)
      - snowfall       : Nevadas (cm/h)

    Retorna:
        DataFrame con columna 'datetime' (UTC) + las 3 variables climáticas.
        El 'datetime' se descompone en year, month, day, hour para el merge.

    Nota:
        Usa requests_cache para evitar re-descargar en múltiples ejecuciones.
    """
    logger.info("=" * 70)
    logger.info("PASO 1: FETCH CLIMA HISTÓRICO Open-Meteo (Boston, 2024)")
    logger.info("=" * 70)
    logger.info("  Rango: %s → %s", TRAIN_START, TRAIN_END)
    logger.info("  Coord: lat=%.4f, lon=%.4f", BOSTON_LAT, BOSTON_LON)

    t0 = time.time()

    # Configurar caché de HTTP (evita re-downloads en re-ejecuciones)
    cache_session = requests_cache.CachedSession(
        cache_name=str(BASE_DIR / ".weather_cache_2024"),
        backend="sqlite",
        expire_after=-1,   # caché permanente (datos históricos no cambian)
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude":   BOSTON_LAT,
        "longitude":  BOSTON_LON,
        "start_date": TRAIN_START,
        "end_date":   TRAIN_END,
        "hourly":     ["temperature_2m", "precipitation", "snowfall"],
        "timezone":   "America/New_York",   # zona horaria de Boston
    }

    try:
        responses = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )
        response = responses[0]
        hourly   = response.Hourly()

        # Construir rango de timestamps (índice horario)
        dt_index = pd.date_range(
            start=pd.Timestamp(hourly.Time(),      unit="s", tz="America/New_York"),
            end=  pd.Timestamp(hourly.TimeEnd(),   unit="s", tz="America/New_York"),
            freq=pd.tseries.frequencies.to_offset(f"{hourly.Interval()}s"),
            inclusive="left",
        )

        weather_df = pd.DataFrame({
            "datetime":      dt_index,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy().astype("float32"),
            "precipitation":  hourly.Variables(1).ValuesAsNumpy().astype("float32"),
            "snowfall":       hourly.Variables(2).ValuesAsNumpy().astype("float32"),
        })

    except Exception as exc:
        logger.error("❌ Error en Open-Meteo API: %s. Continuando SIN datos climáticos.", exc)
        logger.warning("   Las columnas climáticas se rellenarán con 0.")
        # DataFrame vacío como fallback
        return pd.DataFrame(columns=["datetime", "temperature_2m", "precipitation", "snowfall"])

    elapsed = time.time() - t0
    logger.info(
        "  ✅ Clima descargado: %d horas en %.1fs | temp rango: [%.1f°C, %.1f°C]",
        len(weather_df), elapsed,
        weather_df["temperature_2m"].min(), weather_df["temperature_2m"].max(),
    )

    # Descomponer datetime para hacer join con los datos MBTA
    weather_df["year"]  = weather_df["datetime"].dt.year.astype("int16")
    weather_df["month"] = weather_df["datetime"].dt.month.astype("int8")
    weather_df["day"]   = weather_df["datetime"].dt.day.astype("int8")
    weather_df["hour"]  = weather_df["datetime"].dt.hour.astype("int8")

    # Rellenar nulos (horas sin lectura → 0)
    weather_df[["temperature_2m", "precipitation", "snowfall"]] = (
        weather_df[["temperature_2m", "precipitation", "snowfall"]].fillna(0.0)
    )

    logger.info(
        "  📋 Días con precipitación > 0: %d | Días con nieve > 0: %d",
        (weather_df["precipitation"] > 0).sum() // 24,
        (weather_df["snowfall"] > 0).sum() // 24,
    )

    return weather_df[["year", "month", "day", "hour",
                        "temperature_2m", "precipitation", "snowfall"]]


# ---------------------------------------------------------------------------
# PASO 2: Carga de Datos MBTA 2024 (12 Archivos, ~3 GB)
# ---------------------------------------------------------------------------

def load_mbta_data() -> pd.DataFrame:
    """
    Carga TODOS los archivos CSV del año 2024 usando glob, uno por uno,
    para controlar el uso de memoria.

    Estrategia de eficiencia:
      - dtype_map explícito al leer (evita inferencia costosa).
      - Solo las columnas REQUIRED_COLS (usecols).
      - pd.concat al final con lista preacumulada.
      - GC entre archivos.

    Retorna:
        DataFrame concatenado de los 12 meses de 2024.

    Lanza:
        FileNotFoundError: Si no se encuentra ningún archivo 2024.
        ValueError:        Si faltan columnas requeridas.
    """
    logger.info("=" * 70)
    logger.info("PASO 2: CARGA DE DATOS MBTA 2024 (12 meses)")
    logger.info("=" * 70)
    logger.info("  Directorio: %s", DATA_DIR)
    logger.info("  Patrón glob: %s", GLOB_PATTERN)

    log_memory_usage("antes de carga CSV")

    # Ordenar archivos para carga reproducible (enero → diciembre)
    files = sorted(glob.glob(str(DATA_DIR / GLOB_PATTERN)))

    if not files:
        raise FileNotFoundError(
            f"No se encontró ningún archivo con el patrón '{GLOB_PATTERN}' "
            f"en el directorio '{DATA_DIR}'. "
            "Verifica que los datasets 2024 estén descargados."
        )

    logger.info("  ✅ Archivos encontrados: %d", len(files))
    for f in files:
        logger.info("     → %s (%.0f MB)", Path(f).name, Path(f).stat().st_size / 1e6)

    # dtype_map de alta eficiencia de memoria
    # Int64 (pandas nullable) para columnas que pueden tener NaN
    dtype_map = {
        "route_id":         "str",
        "direction_id":     "str",
        "half_trip_id":     "str",
        "stop_id":          "str",
        "time_point_order": "Int64",  # nullable int (puede tener NaN en MBTA)
    }

    dfs = []
    total_rows_loaded = 0

    for fpath in files:
        fname = Path(fpath).name
        logger.info("  📂 Cargando: %s ...", fname)
        t0 = time.time()

        try:
            chunk_df = pd.read_csv(
                fpath,
                usecols=REQUIRED_COLS,
                dtype=dtype_map,
                low_memory=False,
                # No parsear fechas aquí; lo hacemos en engineer_features
                # para mayor control sobre errores.
            )
        except Exception as exc:
            logger.error("  ⚠️  Error leyendo '%s': %s. Saltando archivo.", fname, exc)
            continue

        rows = len(chunk_df)
        elapsed = time.time() - t0
        total_rows_loaded += rows
        logger.info(
            "     → %d filas en %.2fs (%.0f MB en disco)",
            rows, elapsed, Path(fpath).stat().st_size / 1e6,
        )
        dfs.append(chunk_df)

        # Liberación de memoria explícita entre archivos
        del chunk_df
        gc.collect()

    if not dfs:
        raise ValueError("No se pudo cargar ningún archivo CSV correctamente.")

    logger.info("  🔗 Concatenando %d DataFrames ...", len(dfs))
    t0 = time.time()
    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    logger.info("  ✅ Concat en %.2fs | Total filas cargadas: %d", time.time() - t0, total_rows_loaded)

    # Validar columnas
    missing = set(REQUIRED_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en los CSVs: {missing}")

    log_memory_usage("después de carga CSV")
    log_dataframe_stats(df, "POST_CARGA_MBTA")
    return df


# ---------------------------------------------------------------------------
# PASO 3: Feature Engineering Node-to-Node + Mes del Año
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de Feature Engineering:

      1. Parseo de 'actual' y 'scheduled' a datetime.
      2. Extracción de: hora_del_dia, dia_semana, MES (1-12).
      3. Extracción de year, month, day, hour para el merge climático.
      4. Ordenamiento por (half_trip_id, time_point_order).
      5. shift(-1) para calcular next_actual_time y next_scheduled_time.
      6. Cálculo del target tiempo_viaje_segundos = next_actual - actual.
      7. Cálculo de tiempo_programado_segundos = next_scheduled - scheduled.

    Args:
        df: DataFrame crudo de MBTA.

    Retorna:
        DataFrame con features y target calculados.
    """
    logger.info("=" * 70)
    logger.info("PASO 3: FEATURE ENGINEERING (Node-to-Node + Temporal)")
    logger.info("=" * 70)

    # ── 3a. Parseo de timestamps ─────────────────────────────────────────────
    logger.info("  Parseando timestamps 'actual' y 'scheduled' ...")
    t0 = time.time()
    for col in ("actual", "scheduled"):
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        n_null = int(df[col].isna().sum())
        if n_null > 0:
            logger.warning("  ⚠️  '%s': %d valores no parseables → NaT", col, n_null)
    logger.info("  ✅ Parseo completado en %.2fs", time.time() - t0)

    # ── 3b. Features temporales ──────────────────────────────────────────────
    logger.info("  Extrayendo features temporales ...")
    df["hora_del_dia"] = df["actual"].dt.hour.astype("Int64")        # 0-23
    df["dia_semana"]   = df["actual"].dt.dayofweek.astype("Int64")   # 0=Lun, 6=Dom
    df["mes"]          = df["actual"].dt.month.astype("Int64")        # 1-12 ← NUEVO

    # Keys de merge con la tabla climática
    df["year_merge"]  = df["actual"].dt.year.astype("Int64")
    df["month_merge"] = df["actual"].dt.month.astype("Int64")
    df["day_merge"]   = df["actual"].dt.day.astype("Int64")
    df["hour_merge"]  = df["actual"].dt.hour.astype("Int64")

    logger.info(
        "  ✅ hora_del_dia∈[%d,%d] | dia_semana∈[%d,%d] | mes∈[%d,%d]",
        int(df["hora_del_dia"].min()), int(df["hora_del_dia"].max()),
        int(df["dia_semana"].min()),   int(df["dia_semana"].max()),
        int(df["mes"].min()),          int(df["mes"].max()),
    )

    # ── 3c. Ordenar para garantizar secuencia de paradas ────────────────────
    logger.info("  Ordenando por (half_trip_id, time_point_order) ...")
    t0 = time.time()
    df.sort_values(["half_trip_id", "time_point_order"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("  ✅ Ordenamiento en %.2fs", time.time() - t0)

    # ── 3d. shift(-1) para siguiente parada ─────────────────────────────────
    logger.info("  Calculando next_actual_time, next_scheduled_time, next_stop_id ...")
    t0 = time.time()
    grouped = df.groupby("half_trip_id", sort=False)
    df["next_actual_time"]    = grouped["actual"].shift(-1)
    df["next_scheduled_time"] = grouped["scheduled"].shift(-1)
    df["next_stop_id"]        = grouped["stop_id"].shift(-1)
    logger.info("  ✅ shift(-1) completado en %.2fs", time.time() - t0)
    logger.info(
        "  ℹ️  NaT en next_actual_time (última parada/viaje): %d",
        int(df["next_actual_time"].isna().sum()),
    )

    # ── 3e. Calcular targets ─────────────────────────────────────────────────
    logger.info("  Calculando targets ...")
    df["tiempo_viaje_segundos"] = (
        df["next_actual_time"] - df["actual"]
    ).dt.total_seconds()

    df["tiempo_programado_segundos"] = (
        df["next_scheduled_time"] - df["scheduled"]
    ).dt.total_seconds()

    df["retraso_actual_segundos"] = (
        df["actual"] - df["scheduled"]
    ).dt.total_seconds()

    logger.info(
        "  ✅ target stats: min=%.1f | media=%.1f | mediana=%.1f | max=%.1f",
        float(df["tiempo_viaje_segundos"].min()),
        float(df["tiempo_viaje_segundos"].mean()),
        float(df["tiempo_viaje_segundos"].median()),
        float(df["tiempo_viaje_segundos"].max()),
    )

    log_dataframe_stats(df, "POST_FEATURE_ENGINEERING")
    return df


# ---------------------------------------------------------------------------
# PASO 4: Merge con Datos Climáticos
# ---------------------------------------------------------------------------

def merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza un left join entre los datos MBTA y el clima horario de Boston.

    La clave de join es: (year, month, day, hour) en zona horaria America/New_York.
    Los nulos climáticos resultantes del merge se rellenan con 0 (indica
    condiciones normales — sin precipitación ni nieve).

    Args:
        df:         DataFrame MBTA con columns year_merge, month_merge, day_merge, hour_merge.
        weather_df: DataFrame de Open-Meteo con columnas year, month, day, hour + variables.

    Retorna:
        DataFrame enriquecido con columnas climáticas.
    """
    logger.info("=" * 70)
    logger.info("PASO 4: MERGE CON DATOS CLIMÁTICOS")
    logger.info("=" * 70)

    if weather_df.empty:
        logger.warning("  ⚠️  DataFrame climático vacío. Añadiendo columnas climáticas en 0.")
        df["temperature_2m"] = np.float32(0.0)
        df["precipitation"]  = np.float32(0.0)
        df["snowfall"]       = np.float32(0.0)
        return df

    rows_before = len(df)
    t0 = time.time()

    # Renombrar claves del clima para el merge
    weather_df = weather_df.rename(columns={
        "year":  "year_merge",
        "month": "month_merge",
        "day":   "day_merge",
        "hour":  "hour_merge",
    })

    # Asegurar tipos consistentes en las claves de join
    for col in ["year_merge", "month_merge", "day_merge", "hour_merge"]:
        df[col]         = df[col].astype("Int64")
        weather_df[col] = weather_df[col].astype("int64")

    df = df.merge(
        weather_df,
        on=["year_merge", "month_merge", "day_merge", "hour_merge"],
        how="left",
    )

    elapsed = time.time() - t0
    rows_after = len(df)

    # Validar que el merge no introdujo duplicados (debería ser 1:1 llave temporal)
    if rows_after != rows_before:
        logger.warning(
            "  ⚠️  Merge introdujo filas extra: %d → %d. "
            "Posible duplicado en el índice climático.",
            rows_before, rows_after,
        )

    # Rellenar nulos climáticos (horas sin cobertura) con 0
    climate_cols = ["temperature_2m", "precipitation", "snowfall"]
    n_nulls = df[climate_cols].isna().any(axis=1).sum()
    if n_nulls > 0:
        logger.info("  🔧 Rellenando %d filas sin coincidencia climática con 0.", n_nulls)
    df[climate_cols] = df[climate_cols].fillna(0.0)

    # Asegurar float32 para eficiencia de memoria
    for col in climate_cols:
        df[col] = df[col].astype("float32")

    # Limpiar columnas auxiliares de merge
    df.drop(columns=["year_merge", "month_merge", "day_merge", "hour_merge"],
            inplace=True, errors="ignore")

    logger.info(
        "  ✅ Merge completado en %.2fs | "
        "Días lluviosos (prec>0mm): %d | Días con nieve: %d",
        elapsed,
        (df["precipitation"] > 0).sum() // 24,
        (df["snowfall"] > 0).sum() // 24,
    )
    log_dataframe_stats(df, "POST_MERGE_CLIMA")
    return df


# ---------------------------------------------------------------------------
# PASO 5: Limpieza y Filtrado con Observabilidad Total
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline de limpieza exhaustivo. Loguea exactamente cuántas filas
    se descartan en cada paso.

    Filtros aplicados (en orden):
      1. NaT en actual/scheduled.
      2. NaT en next_actual_time / next_stop_id (última parada del viaje).
      3. NaT en tiempo_viaje_segundos.
      4. Outliers en target: tiempo_viaje_segundos ∉ (0, 3600].
      5. Outliers en retraso: |retraso_actual_segundos| > 7200s.
      6. NaT en time_point_order.
      7. Nulos en features del modelo.

    Retorna:
        DataFrame limpio.
    """
    logger.info("=" * 70)
    logger.info("PASO 5: LIMPIEZA Y FILTRADO DE OUTLIERS")
    logger.info("=" * 70)

    # 5a. NaT en timestamps de entrada
    n = len(df)
    df = df.dropna(subset=["actual", "scheduled"])
    log_drop(n, len(df), "NaT en actual/scheduled")

    # 5b. NaT en next_actual_time (última parada de cada viaje → sin target)
    n = len(df)
    df = df.dropna(subset=["next_actual_time", "next_stop_id"])
    log_drop(n, len(df), "next_actual_time/next_stop_id nulos (última parada)")

    # 5c. NaT en target calculado
    n = len(df)
    df = df.dropna(subset=["tiempo_viaje_segundos"])
    log_drop(n, len(df), "tiempo_viaje_segundos nulo")

    # 5d. Outliers en target: tiempos negativos o irrealmente largos
    n = len(df)
    df = df[
        (df["tiempo_viaje_segundos"] > MIN_TRAVEL_TIME_S) &
        (df["tiempo_viaje_segundos"] <= MAX_TRAVEL_TIME_S)
    ]
    log_drop(n, len(df), f"tiempo_viaje_segundos ∉ ({MIN_TRAVEL_TIME_S}, {MAX_TRAVEL_TIME_S}]")

    # 5e. Outliers extremos en retraso (>2h indica datos corruptos)
    n = len(df)
    df = df[df["retraso_actual_segundos"].abs() <= 7_200]
    log_drop(n, len(df), "|retraso_actual_segundos| > 7200s")

    # 5f. Nulos en time_point_order
    n = len(df)
    df = df.dropna(subset=["time_point_order"])
    log_drop(n, len(df), "time_point_order nulo")

    # 5g. Nulos en cualquier feature del modelo
    n = len(df)
    df = df.dropna(subset=ALL_FEATURES)
    log_drop(n, len(df), "nulos en features del modelo")

    df = df.reset_index(drop=True)
    log_dataframe_stats(df, "POST_LIMPIEZA")

    # Distribución del target para validación de sanidad
    logger.info(
        "  📈 Target final | min=%.1fs | p25=%.1fs | p50=%.1fs | p75=%.1fs | max=%.1fs",
        float(df[TARGET].quantile(0.00)),
        float(df[TARGET].quantile(0.25)),
        float(df[TARGET].quantile(0.50)),
        float(df[TARGET].quantile(0.75)),
        float(df[TARGET].quantile(1.00)),
    )

    return df


# ---------------------------------------------------------------------------
# PASO 6: Preparación de Features para XGBoost
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepara la matriz de features X y el vector target y para XGBoost.

    Transformaciones:
      - Numéricas → float32 (eficiencia de memoria).
      - Categóricas → dtype 'category' (requerido por enable_categorical=True).

    Retorna:
        (X, y): Tuple con el DataFrame de features y la Serie del target.
    """
    logger.info("=" * 70)
    logger.info("PASO 6: PREPARACIÓN DE FEATURES PARA XGBOOST")
    logger.info("=" * 70)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    # Numéricas → float32
    for col in NUMERIC_FEATURES:
        X[col] = X[col].astype("float32")
        logger.info(
            "  🔢 '%s' → float32 | rango: [%.2f, %.2f]",
            col, float(X[col].min()), float(X[col].max()),
        )

    # Categóricas → category (XGBoost native)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
        n_cats = X[col].nunique()
        logger.info("  🏷️  '%s' → category | cardinalidad: %d", col, n_cats)

    logger.info("  ✅ X.shape=%s | y: media=%.2fs, std=%.2fs", X.shape, float(y.mean()), float(y.std()))
    return X, y


# ---------------------------------------------------------------------------
# PASO 7: Entrenamiento con Early Stopping
# ---------------------------------------------------------------------------

def train(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """
    Divide los datos en Train/Validación (80/20) dentro del año 2024
    y entrena el XGBRegressor con early stopping.

    Nota:
        La evaluación OUT-OF-TIME real se hace con datos de 2025
        en evaluate_model.py. Este split interno sirve solo para
        calibrar early stopping sin contaminar el test set.

    Args:
        X: DataFrame de features.
        y: Serie del target.

    Retorna:
        Modelo XGBRegressor entrenado.
    """
    logger.info("=" * 70)
    logger.info("PASO 7: ENTRENAMIENTO XGBOOST CON EARLY STOPPING")
    logger.info("=" * 70)

    # Split 80/20: shuffle=True para representatividad interna
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True,
    )
    logger.info(
        "  📊 Train: %d filas | Validación interna: %d filas | Split 80/20",
        len(X_train), len(X_val),
    )

    model = xgb.XGBRegressor(**XGB_PARAMS)
    logger.info("  🤖 Hiperparámetros:")
    for k, v in XGB_PARAMS.items():
        logger.info("     %-30s = %s", k, v)

    logger.info("  🚀 Iniciando entrenamiento (verbose cada 50 árboles) ...")
    t_start = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,
    )

    elapsed = time.time() - t_start
    logger.info("  ✅ Entrenamiento completado en %.1fs (%.1f min)", elapsed, elapsed / 60)
    logger.info(
        "  🏆 Mejor iteración: %d | Mejor RMSE validación interna: %.4f",
        model.best_iteration, model.best_score,
    )

    log_memory_usage("post-entrenamiento")
    return model


# ---------------------------------------------------------------------------
# PASO 8: Guardar Modelo
# ---------------------------------------------------------------------------

def save_model(model: xgb.XGBRegressor) -> None:
    """
    Guarda el modelo entrenado en formato JSON nativo de XGBoost.
    El formato JSON es portátil, versionable con git y legible para debug.
    """
    logger.info("=" * 70)
    logger.info("PASO 8: GUARDADO DEL MODELO")
    logger.info("=" * 70)

    model.save_model(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    logger.info("  ✅ Modelo guardado: %s (%.2f MB)", MODEL_PATH, size_mb)
    logger.info("  ℹ️  Formato: JSON nativo XGBoost | Features: %s", ALL_FEATURES)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()
    logger.info("╔═══════════════════════════════════════════════════════════════════╗")
    logger.info("║  MBTA TRANSIT — XGBoost Out-of-Time Training (2024 → Test 2025)  ║")
    logger.info("╚═══════════════════════════════════════════════════════════════════╝")
    logger.info("Timestamp inicio: %s", pd.Timestamp.now().isoformat())
    log_memory_usage("arranque")

    try:
        # 1. Fetch clima histórico 2024
        weather_df = fetch_weather_history()

        # 2. Cargar MBTA 2024 (12 archivos)
        df = load_mbta_data()

        # 3. Feature Engineering
        df = engineer_features(df)

        # 4. Merge con clima
        df = merge_weather(df, weather_df)
        del weather_df
        gc.collect()
        log_memory_usage("post-merge")

        # 5. Limpieza con logging granular
        df = clean_data(df)

        # Guard: si queda poco dato, el modelo no valdrá nada
        if len(df) < 50_000:
            logger.critical(
                "❌ Solo %d filas tras la limpieza. "
                "Revisa los datasets 2024 y los umbrales de filtrado.", len(df),
            )
            sys.exit(1)

        # 6. Preparar features
        X, y = prepare_features(df)

        # Liberar df original (ya no se necesita)
        del df
        gc.collect()
        log_memory_usage("post-liberación df")

        # 7. Entrenamiento
        model = train(X, y)
        del X, y
        gc.collect()

        # 8. Guardar modelo
        save_model(model)

        total_elapsed = time.time() - total_start
        logger.info("=" * 70)
        logger.info(
            "🎉 Pipeline completado en %.1fs (%.1f min)",
            total_elapsed, total_elapsed / 60,
        )
        logger.info("   Modelo listo: %s", MODEL_PATH)
        logger.info("   Siguiente paso: ejecutar evaluate_model.py con datos 2025")

    except FileNotFoundError as exc:
        logger.critical("❌ Archivo no encontrado: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.critical("❌ Error de validación: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("❌ Error inesperado: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
