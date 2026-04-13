"""
evaluate_model.py — Evaluación Out-of-Time (Año Completo 2025)
===============================================================
Estrategia de Validación: Cross-Year (Out-of-Time)
  - MODELO CARGADO:  transit_xgboost_model.json (entrenado en 2024).
  - DATOS DE TEST:   Todo el año 2025 (12 meses, nunca vistos en entrenamiento).

Esta separación temporal estricta es la máxima garantía de generalización:
el modelo debe predecir condiciones de un año futuro completo.

Pipeline de Evaluación:
  1. load_model()              → Carga transit_xgboost_model.json.
  2. fetch_weather_test()      → Open-Meteo Historical API (2025-01-01 → 2025-12-31).
  3. load_mbta_test()          → glob "*2025*.csv" sobre datasets/MBTA.../2025/.
  4. engineer_features()       → Mismo pipeline que en train_model.py.
  5. merge_weather()           → pd.merge by (year, month, day, hour).
  6. clean_data()              → Mismo pipeline de limpieza que en training.
  7. prepare_features()        → Categóricas → category, numéricas → float32.
  8. evaluate()                → MAE, RMSE, R², MAPE + distribución de errores.
  9. report_feature_importance() → Ranking Gain ordenado (precipitación vs. stop_id).

Uso:
  python evaluate_model.py

Prerequisito:
  Haber ejecutado train_model.py para generar transit_xgboost_model.json
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_model")

# ---------------------------------------------------------------------------
# Constantes (DEBEN ser idénticas a train_model.py)
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "datasets" / "MBTA_Bus_Arrival_Departure_Times_2025"
MODEL_PATH = BASE_DIR / "transit_xgboost_model.json"

BOSTON_LAT = 42.3601
BOSTON_LON = -71.0589

# Ventana temporal del conjunto de prueba (año completo 2025)
TEST_START = "2025-01-01"
TEST_END   = "2025-12-31"

# Patrón glob para seleccionar SOLO archivos de 2025
GLOB_PATTERN = "*2025*.csv"

REQUIRED_COLS = [
    "service_date", "route_id", "direction_id",
    "half_trip_id", "stop_id", "time_point_order",
    "scheduled", "actual",
]

CATEGORICAL_FEATURES = ["route_id", "direction_id", "stop_id", "next_stop_id"]
NUMERIC_FEATURES     = ["hora_del_dia", "dia_semana", "mes",
                        "temperature_2m", "precipitation", "snowfall"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET               = "tiempo_viaje_segundos"

MIN_TRAVEL_TIME_S = 0
MAX_TRAVEL_TIME_S = 3_600


# ---------------------------------------------------------------------------
# Utilidades de Observabilidad
# ---------------------------------------------------------------------------

def log_memory_usage(context: str) -> None:
    """Loguea el uso de RAM (RSS) del proceso en MB."""
    proc = psutil.Process(os.getpid())
    rss_mb = proc.memory_info().rss / (1024 ** 2)
    logger.info("🧠 RAM [%s]: %.1f MB", context, rss_mb)


def log_drop(before: int, after: int, reason: str) -> None:
    """Loguea filas eliminadas y porcentaje retenido."""
    dropped = before - after
    pct = (dropped / before * 100) if before > 0 else 0.0
    logger.info(
        "🗑️  [%s]: %d filas eliminadas (%.2f%%) | quedan %d",
        reason, dropped, pct, after,
    )


# ---------------------------------------------------------------------------
# PASO 1: Carga del Modelo
# ---------------------------------------------------------------------------

def load_model() -> xgb.XGBRegressor:
    """
    Carga el modelo XGBoost desde transit_xgboost_model.json.
    Falla de forma crítica si el archivo no existe.

    Retorna:
        XGBRegressor listo para inferencia.
    """
    logger.info("=" * 70)
    logger.info("PASO 1: CARGA DEL MODELO (entrenado en 2024)")
    logger.info("=" * 70)

    if not MODEL_PATH.exists():
        logger.critical(
            "❌ MODELO NO ENCONTRADO: '%s'. "
            "Ejecuta train_model.py primero.",
            MODEL_PATH,
        )
        sys.exit(1)

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)

    logger.info("  ✅ Modelo cargado: %s (%.2f MB)", MODEL_PATH.name, size_mb)
    params = model.get_params()
    for k in ["n_estimators", "max_depth", "learning_rate", "tree_method", "enable_categorical"]:
        logger.info("     %-30s = %s", k, params.get(k, "N/A"))

    return model


# ---------------------------------------------------------------------------
# PASO 2: Fetch del Clima de Test (Open-Meteo, 2025)
# ---------------------------------------------------------------------------

def fetch_weather_test() -> pd.DataFrame:
    """
    Descarga datos climáticos horarios de Boston para el año 2025 completo.
    Reutiliza la misma API Historical de Open-Meteo (los datos de 2025
    ya están disponibles en el archivo histórico).

    Retorna:
        DataFrame con year, month, day, hour + temperature_2m, precipitation, snowfall.
    """
    logger.info("=" * 70)
    logger.info("PASO 2: FETCH CLIMA TEST Open-Meteo (Boston, 2025)")
    logger.info("=" * 70)
    logger.info("  Rango: %s → %s", TEST_START, TEST_END)

    t0 = time.time()

    cache_session = requests_cache.CachedSession(
        cache_name=str(BASE_DIR / ".weather_cache_2025"),
        backend="sqlite",
        expire_after=-1,
    )
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    om = openmeteo_requests.Client(session=retry_session)

    params = {
        "latitude":   BOSTON_LAT,
        "longitude":  BOSTON_LON,
        "start_date": TEST_START,
        "end_date":   TEST_END,
        "hourly":     ["temperature_2m", "precipitation", "snowfall"],
        "timezone":   "America/New_York",
    }

    try:
        responses = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive", params=params
        )
        response = responses[0]
        hourly   = response.Hourly()

        dt_index = pd.date_range(
            start=pd.Timestamp(hourly.Time(),    unit="s", tz="America/New_York"),
            end=  pd.Timestamp(hourly.TimeEnd(), unit="s", tz="America/New_York"),
            freq=pd.tseries.frequencies.to_offset(f"{hourly.Interval()}s"),
            inclusive="left",
        )

        weather_df = pd.DataFrame({
            "datetime":       dt_index,
            "temperature_2m": hourly.Variables(0).ValuesAsNumpy().astype("float32"),
            "precipitation":  hourly.Variables(1).ValuesAsNumpy().astype("float32"),
            "snowfall":       hourly.Variables(2).ValuesAsNumpy().astype("float32"),
        })

    except Exception as exc:
        logger.error("❌ Error en Open-Meteo API: %s. Continuando SIN datos climáticos.", exc)
        return pd.DataFrame(columns=["datetime", "temperature_2m", "precipitation", "snowfall"])

    logger.info(
        "  ✅ Clima 2025 descargado: %d horas en %.1fs",
        len(weather_df), time.time() - t0,
    )

    weather_df["year"]  = weather_df["datetime"].dt.year.astype("int16")
    weather_df["month"] = weather_df["datetime"].dt.month.astype("int8")
    weather_df["day"]   = weather_df["datetime"].dt.day.astype("int8")
    weather_df["hour"]  = weather_df["datetime"].dt.hour.astype("int8")

    weather_df[["temperature_2m", "precipitation", "snowfall"]] = (
        weather_df[["temperature_2m", "precipitation", "snowfall"]].fillna(0.0)
    )

    return weather_df[["year", "month", "day", "hour",
                        "temperature_2m", "precipitation", "snowfall"]]


# ---------------------------------------------------------------------------
# PASO 3: Carga de Datos MBTA 2025 (12 Archivos)
# ---------------------------------------------------------------------------

def load_mbta_test() -> pd.DataFrame:
    """
    Carga TODOS los archivos CSV del año 2025 usando glob.
    Aplica el mismo pipeline de Feature Engineering, Merge y Limpieza
    que en train_model.py para garantizar coherencia estricta del pipeline.

    Retorna:
        DataFrame crudo de MBTA 2025 listo para feature engineering.
    """
    logger.info("=" * 70)
    logger.info("PASO 3: CARGA DE DATOS MBTA TEST (2025, 12 meses)")
    logger.info("=" * 70)
    logger.info("  Directorio: %s", DATA_DIR)

    log_memory_usage("antes de carga CSV 2025")

    files = sorted(glob.glob(str(DATA_DIR / GLOB_PATTERN)))

    if not files:
        raise FileNotFoundError(
            f"No se encontró ningún archivo con el patrón '{GLOB_PATTERN}' "
            f"en '{DATA_DIR}'. Verifica que los datasets 2025 estén descargados."
        )

    logger.info("  ✅ Archivos encontrados: %d", len(files))
    for f in files:
        logger.info("     → %s (%.0f MB)", Path(f).name, Path(f).stat().st_size / 1e6)

    dtype_map = {
        "route_id":         "str",
        "direction_id":     "str",
        "half_trip_id":     "str",
        "stop_id":          "str",
        "time_point_order": "Int64",
    }

    dfs = []
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
            )
        except Exception as exc:
            logger.error("  ⚠️  Error leyendo '%s': %s. Saltando.", fname, exc)
            continue

        logger.info("     → %d filas en %.2fs", len(chunk_df), time.time() - t0)
        dfs.append(chunk_df)
        del chunk_df
        gc.collect()

    if not dfs:
        raise ValueError("No se pudo cargar ningún archivo CSV de 2025.")

    df = pd.concat(dfs, ignore_index=True)
    del dfs
    gc.collect()
    logger.info("  ✅ MBTA 2025 total: %d filas", len(df))
    log_memory_usage("post-carga CSV 2025")
    return df


# ---------------------------------------------------------------------------
# PASO 4: Feature Engineering (idéntico a train_model.py)
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replica el pipeline de feature engineering de train_model.py.
    Debe ser 100% idéntico para garantizar la coherencia del pipeline.
    """
    logger.info("=" * 70)
    logger.info("PASO 4: FEATURE ENGINEERING (2025 test set)")
    logger.info("=" * 70)

    for col in ("actual", "scheduled"):
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)

    df["hora_del_dia"] = df["actual"].dt.hour.astype("Int64")
    df["dia_semana"]   = df["actual"].dt.dayofweek.astype("Int64")
    df["mes"]          = df["actual"].dt.month.astype("Int64")

    df["year_merge"]  = df["actual"].dt.year.astype("Int64")
    df["month_merge"] = df["actual"].dt.month.astype("Int64")
    df["day_merge"]   = df["actual"].dt.day.astype("Int64")
    df["hour_merge"]  = df["actual"].dt.hour.astype("Int64")

    df.sort_values(["half_trip_id", "time_point_order"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    grouped = df.groupby("half_trip_id", sort=False)
    df["next_actual_time"]    = grouped["actual"].shift(-1)
    df["next_scheduled_time"] = grouped["scheduled"].shift(-1)
    df["next_stop_id"]        = grouped["stop_id"].shift(-1)

    df["tiempo_viaje_segundos"]     = (df["next_actual_time"] - df["actual"]).dt.total_seconds()
    df["tiempo_programado_segundos"]= (df["next_scheduled_time"] - df["scheduled"]).dt.total_seconds()
    df["retraso_actual_segundos"]   = (df["actual"] - df["scheduled"]).dt.total_seconds()

    logger.info(
        "  ✅ FE completado | filas=%d | mes∈[%d,%d]",
        len(df), int(df["mes"].min()), int(df["mes"].max()),
    )
    return df


# ---------------------------------------------------------------------------
# PASO 5: Merge con Clima 2025
# ---------------------------------------------------------------------------

def merge_weather(df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Replica el merge climático de train_model.py."""
    logger.info("=" * 70)
    logger.info("PASO 5: MERGE CON DATOS CLIMÁTICOS 2025")
    logger.info("=" * 70)

    if weather_df.empty:
        logger.warning("  ⚠️  DataFrame climático vacío. Columnas climáticas = 0.")
        df["temperature_2m"] = np.float32(0.0)
        df["precipitation"]  = np.float32(0.0)
        df["snowfall"]       = np.float32(0.0)
        return df

    rows_before = len(df)

    weather_df = weather_df.rename(columns={
        "year": "year_merge", "month": "month_merge",
        "day":  "day_merge",  "hour":  "hour_merge",
    })
    for col in ["year_merge", "month_merge", "day_merge", "hour_merge"]:
        df[col]         = df[col].astype("Int64")
        weather_df[col] = weather_df[col].astype("int64")

    df = df.merge(
        weather_df,
        on=["year_merge", "month_merge", "day_merge", "hour_merge"],
        how="left",
    )

    climate_cols = ["temperature_2m", "precipitation", "snowfall"]
    df[climate_cols] = df[climate_cols].fillna(0.0)
    for col in climate_cols:
        df[col] = df[col].astype("float32")

    df.drop(columns=["year_merge", "month_merge", "day_merge", "hour_merge"],
            inplace=True, errors="ignore")

    logger.info("  ✅ Merge OK | %d → %d filas", rows_before, len(df))
    return df


# ---------------------------------------------------------------------------
# PASO 6: Limpieza (idéntica a train_model.py)
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Replica el pipeline de limpieza de train_model.py."""
    logger.info("=" * 70)
    logger.info("PASO 6: LIMPIEZA DE DATOS (2025 test set)")
    logger.info("=" * 70)

    n_original = len(df)

    n = len(df)
    df = df.dropna(subset=["actual", "scheduled"])
    log_drop(n, len(df), "NaT en actual/scheduled")

    n = len(df)
    df = df.dropna(subset=["next_actual_time", "next_stop_id"])
    log_drop(n, len(df), "next_actual_time/next_stop_id nulos")

    n = len(df)
    df = df.dropna(subset=["tiempo_viaje_segundos"])
    log_drop(n, len(df), "tiempo_viaje_segundos nulo")

    n = len(df)
    df = df[
        (df["tiempo_viaje_segundos"] > MIN_TRAVEL_TIME_S) &
        (df["tiempo_viaje_segundos"] <= MAX_TRAVEL_TIME_S)
    ]
    log_drop(n, len(df), f"tiempo_viaje_segundos ∉ ({MIN_TRAVEL_TIME_S}, {MAX_TRAVEL_TIME_S}]")

    n = len(df)
    df = df[df["retraso_actual_segundos"].abs() <= 7_200]
    log_drop(n, len(df), "|retraso_actual_segundos| > 7200s")

    n = len(df)
    df = df.dropna(subset=["time_point_order"])
    log_drop(n, len(df), "time_point_order nulo")

    n = len(df)
    df = df.dropna(subset=ALL_FEATURES)
    log_drop(n, len(df), "nulos en features del modelo")

    df = df.reset_index(drop=True)

    n_clean = len(df)
    logger.info(
        "  ✅ Limpieza: %d → %d filas (%.1f%% retenido)",
        n_original, n_clean, (n_clean / n_original * 100) if n_original > 0 else 0,
    )
    return df


# ---------------------------------------------------------------------------
# PASO 7: Preparación de Features para XGBoost
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepara X, y con los mismos tipos que en entrenamiento."""
    logger.info("=" * 70)
    logger.info("PASO 7: PREPARACIÓN DE FEATURES (2025 test set)")
    logger.info("=" * 70)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    for col in NUMERIC_FEATURES:
        X[col] = X[col].astype("float32")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")

    logger.info("  ✅ X.shape=%s | y: media=%.2fs, std=%.2fs",
                X.shape, float(y.mean()), float(y.std()))
    return X, y


# ---------------------------------------------------------------------------
# PASO 8: Evaluación y Reporte de Métricas Rigurosas
# ---------------------------------------------------------------------------

def evaluate(
    model: xgb.XGBRegressor,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    """
    Ejecuta la inferencia sobre el año 2025 completo y calcula métricas
    de evaluación Out-of-Time rigurosas.

    Métricas:
      - MAE   : Error Absoluto Medio (segundos)
      - RMSE  : Raíz del Error Cuadrático Medio (segundos)
      - R²    : Coeficiente de determinación (1.0 = perfecto)
      - MAPE  : Mean Absolute Percentage Error (%)
      - Acc@60s  : % de predicciones con |error| < 60s
      - Acc@120s : % de predicciones con |error| < 120s
      - Distribución de errores (P50, P90, P95, P99)

    Retorna:
        Dict con todas las métricas calculadas.
    """
    logger.info("=" * 70)
    logger.info("PASO 8: INFERENCIA Y MÉTRICAS OUT-OF-TIME (TEST 2025)")
    logger.info("=" * 70)

    logger.info("  🔮 Inferencia sobre %d muestras (año 2025) ...", len(X))
    t0 = time.time()
    y_pred    = model.predict(X)
    t_infer   = time.time() - t0
    throughput = len(X) / t_infer

    logger.info(
        "  ✅ Inferencia: %.3fs | throughput=%.0f registros/s",
        t_infer, throughput,
    )

    # Protección MAPE: excluir ceros exactos del denominador
    y_arr        = y.values
    mask_nonzero = y_arr != 0
    y_nz         = y_arr[mask_nonzero]
    yp_nz        = y_pred[mask_nonzero]

    # Métricas principales
    mae  = mean_absolute_error(y_arr, y_pred)
    rmse = np.sqrt(mean_squared_error(y_arr, y_pred))
    r2   = r2_score(y_arr, y_pred)
    mape = np.mean(np.abs((y_nz - yp_nz) / y_nz)) * 100

    # Umbrales operativos de precisión
    errors_abs = np.abs(y_arr - y_pred)
    acc_60s    = (errors_abs < 60).sum()  / len(y) * 100
    acc_120s   = (errors_abs < 120).sum() / len(y) * 100

    # Percentiles de error absoluto
    p50_err = float(np.percentile(errors_abs, 50))
    p90_err = float(np.percentile(errors_abs, 90))
    p95_err = float(np.percentile(errors_abs, 95))
    p99_err = float(np.percentile(errors_abs, 99))

    metrics = {
        "n_samples":     len(y),
        "mae_s":         float(mae),
        "rmse_s":        float(rmse),
        "r2":            float(r2),
        "mape_pct":      float(mape),
        "acc_60s_pct":   float(acc_60s),
        "acc_120s_pct":  float(acc_120s),
        "p50_error_s":   p50_err,
        "p90_error_s":   p90_err,
        "p95_error_s":   p95_err,
        "p99_error_s":   p99_err,
        "throughput_rps": float(throughput),
    }

    # ── Reporte en consola ────────────────────────────────────────────────────
    div = "─" * 72
    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║       REPORTE OUT-OF-TIME — MBTA XGBoost (Train:2024 / Test:2025)      ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print(f"║  Test set   : Año completo 2025 (nunca visto en entrenamiento)         ║")
    print(f"║  Muestras   : {len(y):,:<58} ║")
    print("╠══════════════════════════════════════════════════════════════════════════╣")
    print("║  MÉTRICAS PRIMARIAS                                                      ║")
    print(div)
    print(f"  {'MAE  — Error Absoluto Medio':<45} {mae:>10.2f} s")
    print(f"  {'RMSE — Raíz Error Cuadrático Medio':<45} {rmse:>10.2f} s")
    print(f"  {'R²   — Coeficiente de Determinación':<45} {r2:>10.4f}")
    print(f"  {'MAPE — Error Porcentual Medio Absoluto':<45} {mape:>10.2f} %")
    print(div)
    print("║  UMBRALES OPERATIVOS                                                     ║")
    print(div)
    print(f"  {'Accuracy < 60s  (1 minuto de precisión)':<45} {acc_60s:>10.2f} %")
    print(f"  {'Accuracy < 120s (2 minutos de precisión)':<45} {acc_120s:>10.2f} %")
    print(div)
    print("║  DISTRIBUCIÓN DEL ERROR ABSOLUTO (Percentiles)                          ║")
    print(div)
    print(f"  {'P50 (mediana del error)':<45} {p50_err:>10.2f} s")
    print(f"  {'P90':<45} {p90_err:>10.2f} s")
    print(f"  {'P95':<45} {p95_err:>10.2f} s")
    print(f"  {'P99':<45} {p99_err:>10.2f} s")
    print(div)
    print(f"  {'Throughput de inferencia':<45} {throughput:>10,.0f} rec/s")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    logger.info(
        "OUT-OF-TIME: MAE=%.2fs | RMSE=%.2fs | R²=%.4f | MAPE=%.2f%%",
        mae, rmse, r2, mape,
    )
    logger.info("Accuracy@60s=%.2f%% | Accuracy@120s=%.2f%%", acc_60s, acc_120s)

    return metrics, y_pred


# ---------------------------------------------------------------------------
# PASO 9: Feature Importance (Ranking Gain)
# ---------------------------------------------------------------------------

def report_feature_importance(model: xgb.XGBRegressor) -> None:
    """
    Imprime el ranking de importancia de features por tipo 'gain'.

    'gain' representa la mejora media en el criterio de separación
    que cada feature aporta — es la métrica más informativa y la que
    mejor responde a la pregunta: "¿cuánto impact tienen precipitation
    y snowfall frente a stop_id y route_id?"

    Se agruparán visualmente los features climáticos vs. espaciales.
    """
    logger.info("=" * 70)
    logger.info("PASO 9: FEATURE IMPORTANCE (Gain) — Climáticos vs. Espaciales")
    logger.info("=" * 70)

    importance_dict = model.get_booster().get_score(importance_type="gain")

    if not importance_dict:
        logger.warning("  ⚠️  No hay datos de importancia (modelo sin árboles?).")
        return

    fi_df = (
        pd.DataFrame.from_dict(importance_dict, orient="index", columns=["gain"])
        .sort_values("gain", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    fi_df["gain_pct"]  = fi_df["gain"] / fi_df["gain"].sum() * 100
    fi_df["rank"]      = range(1, len(fi_df) + 1)

    # Categorizar features para análisis comparativo
    climate_feats  = {"temperature_2m", "precipitation", "snowfall"}
    temporal_feats = {"hora_del_dia", "dia_semana", "mes"}
    spatial_feats  = {"route_id", "direction_id", "stop_id", "next_stop_id"}

    def get_group(feat):
        if feat in climate_feats:  return "🌧️  CLIMÁTICO"
        if feat in temporal_feats: return "🕐 TEMPORAL"
        if feat in spatial_feats:  return "📍 ESPACIAL"
        return "❓ OTRO"

    fi_df["grupo"] = fi_df["feature"].apply(get_group)

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║   FEATURE IMPORTANCE — Gain (mayor = más informativo)               ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"  {'Rank':<5} {'Feature':<20} {'Grupo':<18} {'Gain':>12} {'Gain%':>9}  {'Bar'}")
    print("  " + "─" * 68)

    for _, row in fi_df.iterrows():
        bar = "█" * max(1, int(row["gain_pct"] / 2))
        print(
            f"  {int(row['rank']):<5} {row['feature']:<20} {row['grupo']:<18} "
            f"{row['gain']:>12.2f} {row['gain_pct']:>8.2f}%  {bar}"
        )

    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()

    # Resumen por grupo
    print("  ── Contribución por Grupo (% total de Gain) ──")
    group_summary = fi_df.groupby("grupo")["gain_pct"].sum().sort_values(ascending=False)
    for grupo, pct in group_summary.items():
        print(f"     {grupo:<18}: {pct:.2f}%")
    print()

    logger.info(
        "Top feature: '%s' (gain=%.2f, %.2f%%) | Grupo: %s",
        fi_df.iloc[0]["feature"], fi_df.iloc[0]["gain"],
        fi_df.iloc[0]["gain_pct"],  fi_df.iloc[0]["grupo"],
    )


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    total_start = time.time()
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║  MBTA XGBoost — Evaluación Out-of-Time (Train:2024 / Test:2025) ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("Timestamp inicio: %s", pd.Timestamp.now().isoformat())
    log_memory_usage("arranque evaluación")

    try:
        # 1. Cargar modelo (entrenado en 2024)
        model = load_model()

        # 2. Fetch clima 2025
        weather_df = fetch_weather_test()

        # 3. Cargar MBTA 2025 (12 archivos)
        df = load_mbta_test()

        # 4. Feature Engineering 2025
        df = engineer_features(df)

        # 5. Merge con clima 2025
        df = merge_weather(df, weather_df)
        del weather_df
        gc.collect()
        log_memory_usage("post-merge 2025")

        # 6. Limpieza 2025
        df = clean_data(df)

        if len(df) < 10_000:
            logger.critical(
                "❌ Solo %d filas en test set. Revisa los datasets 2025.", len(df),
            )
            sys.exit(1)

        # 7. Preparar features
        X_test, y_test = prepare_features(df)
        del df
        gc.collect()
        log_memory_usage("post-preparación features")

        # 8. Evaluación Out-of-Time con métricas rigurosas
        metrics, _y_pred = evaluate(model, X_test, y_test)

        # 9. Feature Importance: climáticos vs. espaciales
        report_feature_importance(model)

        total_elapsed = time.time() - total_start
        logger.info("=" * 70)
        logger.info("🎉 Evaluación Out-of-Time completada en %.1fs (%.1f min)",
                    total_elapsed, total_elapsed / 60)

        # Diagnóstico automático de calidad Cross-Year
        mae_val = metrics["mae_s"]
        r2_val  = metrics["r2"]
        print(f"\n  {'─'*50}")
        if mae_val < 60 and r2_val > 0.70:
            print(f"  ✅ MODELO EXCELENTE (Cross-Year): MAE={mae_val:.1f}s | R²={r2_val:.4f}")
        elif mae_val < 120 and r2_val > 0.50:
            print(f"  ⚠️  MODELO ACEPTABLE (Cross-Year): MAE={mae_val:.1f}s | R²={r2_val:.4f}")
        else:
            print(f"  ❌ MODELO POBRE (Cross-Year): MAE={mae_val:.1f}s | R²={r2_val:.4f}")
            print("     → Considera más datos, más features o ajuste de hiperparámetros.")
        print()

    except FileNotFoundError as exc:
        logger.critical("❌ Archivo no encontrado: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("❌ Error inesperado: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
