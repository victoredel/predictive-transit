"""
train_model.py — Pipeline de Entrenamiento XGBoost (MLOps Profesional)
======================================================================
Entrena un XGBRegressor sobre el dataset histórico MBTA Bus Arrival/Departure
Times 2025 usando eventos Node-to-Node (sin coordenadas GPS).

Diseño:
  - Carga solo 2 meses (-01 y -02) para entrenamiento rápido y reproducible.
  - Feature Engineering Node-to-Node: deriva el tiempo real de viaje entre
    paradas consecutivas usando .shift(-1) por half_trip_id.
  - Alta Observabilidad: métricas de RAM, shape del DataFrame, conteo exacto de
    filas descartadas en cada paso de limpieza, y log de pérdida cada 50 árboles.
  - Guarda el modelo en transit_xgboost_model.json (formato nativo XGBoost).

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
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import xgboost as xgb
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuración del Logger — formato ISO con nivel, nombre y mensaje
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_model")

# ---------------------------------------------------------------------------
# Constantes de configuración
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "MBTA_Bus_Arrival_Departure_Times_2025"
MODEL_PATH = BASE_DIR / "transit_xgboost_model.json"

# Meses de entrenamiento: Enero y Febrero 2025
TRAIN_MONTHS = ["2025-01", "2025-02"]

# Columnas mínimas requeridas en el CSV
REQUIRED_COLS = [
    "service_date", "route_id", "direction_id",
    "half_trip_id", "stop_id", "time_point_order",
    "scheduled", "actual",
]

# Límites de sanity check para el target (tiempo de viaje en segundos)
MIN_TRAVEL_TIME_S = 0
MAX_TRAVEL_TIME_S = 3_600   # 1 hora máximo entre paradas

# Features categóricas que XGBoost manejará nativamente
CATEGORICAL_FEATURES = ["route_id", "direction_id", "stop_id", "next_stop_id"]

# Features numéricas
NUMERIC_FEATURES = ["hora_del_dia", "dia_semana"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "tiempo_viaje_segundos"

# Hiperparámetros del modelo
XGB_PARAMS = {
    "n_estimators": 500,
    "early_stopping_rounds": 20,
    "max_depth": 7,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_alpha": 0.1,         # L1 regularización
    "reg_lambda": 1.0,        # L2 regularización
    "objective": "reg:squarederror",
    "tree_method": "hist",    # eficiente para CPU y datasets grandes
    "enable_categorical": True,
    "random_state": 42,
    "eval_metric": "rmse",
}


# ---------------------------------------------------------------------------
# Utilidades de Observabilidad
# ---------------------------------------------------------------------------

def log_memory_usage(context: str) -> None:
    """Loguea el uso actual de RAM del proceso en MB."""
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    logger.info("🧠 RAM [%s]: %.1f MB", context, mem_mb)


def log_dataframe_stats(df: pd.DataFrame, label: str) -> None:
    """Loguea shape, columnas y uso de memoria del DataFrame."""
    mem_df_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    logger.info(
        "📊 [%s] shape=%s | columnas=%s | memoria_df=%.1f MB",
        label, df.shape, list(df.columns), mem_df_mb,
    )


def log_drop(rows_before: int, rows_after: int, reason: str) -> None:
    """Loguea exactamente cuántas filas fueron descartadas y por qué."""
    dropped = rows_before - rows_after
    pct = (dropped / rows_before * 100) if rows_before > 0 else 0
    logger.info(
        "🗑️  Limpieza [%s]: %d filas descartadas (%.2f%%) | quedan %d filas",
        reason, dropped, pct, rows_after,
    )


# ---------------------------------------------------------------------------
# Paso 1: Carga de Datos con Profiling
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """
    Carga los CSVs correspondientes a los meses de entrenamiento usando glob.
    Valida que los archivos existan y que las columnas requeridas estén presentes.

    Returns:
        DataFrame concatenado de todos los meses de entrenamiento.

    Raises:
        FileNotFoundError: Si no se encuentra ningún archivo para los meses configurados.
        ValueError: Si algún archivo no contiene las columnas requeridas.
    """
    logger.info("=" * 70)
    logger.info("PASO 1: CARGA DE DATOS")
    logger.info("=" * 70)
    logger.info("Directorio de datos: %s", DATA_DIR)
    logger.info("Meses a cargar: %s", TRAIN_MONTHS)

    log_memory_usage("antes de carga")

    all_files = []
    for month in TRAIN_MONTHS:
        # Ajusta el patrón al nombre real del archivo MBTA
        pattern = str(DATA_DIR / f"*{month}*.csv")
        matched = glob.glob(pattern)
        if not matched:
            raise FileNotFoundError(
                f"No se encontró ningún archivo para el mes '{month}' "
                f"con el patrón '{pattern}'. Verifica que el dataset esté descargado."
            )
        all_files.extend(matched)
        logger.info("  ✅ Encontrado: %s", [Path(f).name for f in matched])

    logger.info("Total de archivos a cargar: %d", len(all_files))

    dfs = []
    for fpath in all_files:
        t0 = time.time()
        logger.info("  📂 Cargando: %s ...", Path(fpath).name)

        # Definir dtypes en carga para eficiencia de memoria
        # 'actual' y 'scheduled' se cargan como string para parseo controlado
        dtype_map = {
            "route_id":         "str",
            "direction_id":     "str",
            "half_trip_id":     "str",
            "stop_id":          "str",
            "time_point_order": "Int64",  # Int64 tolerante a NaN
        }
        chunk_df = pd.read_csv(
            fpath,
            usecols=REQUIRED_COLS,
            dtype=dtype_map,
            low_memory=False,
        )

        elapsed = time.time() - t0
        logger.info(
            "     → %d filas en %.2fs | shape=%s",
            len(chunk_df), elapsed, chunk_df.shape,
        )
        dfs.append(chunk_df)

    df = pd.concat(dfs, ignore_index=True)
    log_memory_usage("después de concat")
    log_dataframe_stats(df, "POST_CARGA")

    # Validar columnas requeridas
    missing_cols = set(REQUIRED_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el dataset: {missing_cols}")

    return df


# ---------------------------------------------------------------------------
# Paso 2: Feature Engineering Node-to-Node
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica el pipeline completo de Feature Engineering para predicción Node-to-Node.

    Pasos:
      1. Conversión de 'actual' y 'scheduled' a datetime.
      2. Extracción de hora_del_dia y dia_semana desde 'actual'.
      3. Ordenamiento por (half_trip_id, time_point_order) para garantizar
         la correcta secuencia de paradas dentro de cada viaje.
      4. shift(-1) por half_trip_id para obtener next_actual_time y next_stop_id.
      5. Cálculo del target: tiempo_viaje_segundos = next_actual_time - actual.

    Args:
        df: DataFrame con las columnas REQUIRED_COLS.

    Returns:
        DataFrame con features ingenieridas y target calculado.
    """
    logger.info("=" * 70)
    logger.info("PASO 2: FEATURE ENGINEERING")
    logger.info("=" * 70)
    rows_start = len(df)

    # ── 2a. Parseo de timestamps ─────────────────────────────────────────────
    logger.info("  Parseando columnas de tiempo 'actual' y 'scheduled'...")
    t0 = time.time()

    # El formato MBTA es "YYYY-MM-DD HH:MM:SS" combinando service_date + tiempo
    # 'actual' y 'scheduled' pueden ya venir como datetime o como strings HH:MM:SS
    # Intentamos parseo directo; si falla, combinamos con service_date.
    for col in ("actual", "scheduled"):
        # Primero intentamos parseo directo (ya vienen como datetime string)
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)

        n_null = df[col].isna().sum()
        if n_null > 0:
            logger.warning(
                "  ⚠️  '%s': %d valores no parseables → NaT (se eliminarán en limpieza)",
                col, n_null,
            )

    logger.info("  ✅ Timestamps parseados en %.2fs", time.time() - t0)

    # ── 2b. Extracción de features temporales ────────────────────────────────
    logger.info("  Extrayendo hora_del_dia y dia_semana ...")
    df["hora_del_dia"] = df["actual"].dt.hour.astype("Int64")
    df["dia_semana"]   = df["actual"].dt.dayofweek.astype("Int64")  # 0=Lun, 6=Dom
    logger.info("  ✅ hora_del_dia ∈ [%d, %d] | dia_semana ∈ [%d, %d]",
                df["hora_del_dia"].min(), df["hora_del_dia"].max(),
                df["dia_semana"].min(),   df["dia_semana"].max())

    # ── 2c. Ordenamiento para asegurar la secuencia de paradas ───────────────
    logger.info("  Ordenando por (half_trip_id, time_point_order) ...")
    t0 = time.time()
    df.sort_values(["half_trip_id", "time_point_order"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    logger.info("  ✅ Ordenamiento completado en %.2fs", time.time() - t0)

    # ── 2d. Obtención de la siguiente parada por viaje (shift(-1)) ───────────
    logger.info("  Calculando next_actual_time y next_stop_id via shift(-1) ...")
    t0 = time.time()

    # Agrupamos y aplicamos shift para mantener la integridad por viaje
    grouped = df.groupby("half_trip_id", sort=False)
    df["next_actual_time"] = grouped["actual"].shift(-1)
    df["next_stop_id"]     = grouped["stop_id"].shift(-1)

    logger.info("  ✅ shift(-1) completado en %.2fs", time.time() - t0)
    logger.info(
        "  ℹ️  next_actual_time nulos (última parada de cada viaje): %d",
        df["next_actual_time"].isna().sum(),
    )

    # ── 2e. Cálculo del target ───────────────────────────────────────────────
    logger.info("  Calculando target: tiempo_viaje_segundos ...")
    df["tiempo_viaje_segundos"] = (
        df["next_actual_time"] - df["actual"]
    ).dt.total_seconds()

    # También calculamos retraso_actual_segundos como feature complementario
    df["retraso_actual_segundos"] = (
        df["actual"] - df["scheduled"]
    ).dt.total_seconds()

    logger.info(
        "  ✅ Target calculado | stats: min=%.1f, max=%.1f, media=%.1f, mediana=%.1f",
        df["tiempo_viaje_segundos"].min(),
        df["tiempo_viaje_segundos"].max(),
        df["tiempo_viaje_segundos"].mean(),
        df["tiempo_viaje_segundos"].median(),
    )

    log_dataframe_stats(df, "POST_FEATURE_ENGINEERING")
    return df


# ---------------------------------------------------------------------------
# Paso 3: Limpieza y Filtrado con Observabilidad Total
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica un pipeline de limpieza exhaustivo, loguando exactamente cuántas filas
    se descartan en cada paso.

    Pasos:
      1. Eliminar filas con NaT en timestamps críticos.
      2. Eliminar filas sin next_actual_time (última parada del viaje).
      3. Eliminar filas con target fuera de rango [MIN, MAX].
      4. Eliminar filas con retraso extremo (outlier en delay).
      5. Eliminar filas con time_point_order nulo.

    Returns:
        DataFrame limpio listo para entrenamiento.
    """
    logger.info("=" * 70)
    logger.info("PASO 3: LIMPIEZA Y FILTRADO")
    logger.info("=" * 70)

    # ── 3a. Nulos en timestamps de entrada ───────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["actual", "scheduled"])
    log_drop(n_before, len(df), "NaT en actual/scheduled")

    # ── 3b. Nulos en next_actual_time (última parada = sin sucesor) ──────────
    n_before = len(df)
    df = df.dropna(subset=["next_actual_time", "next_stop_id"])
    log_drop(n_before, len(df), "next_actual_time/next_stop_id nulos (última parada)")

    # ── 3c. Nulos en target ──────────────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["tiempo_viaje_segundos"])
    log_drop(n_before, len(df), "tiempo_viaje_segundos nulo")

    # ── 3d. Outliers en target: tiempo de viaje fuera de [0, 3600] ──────────
    n_before = len(df)
    df = df[
        (df["tiempo_viaje_segundos"] > MIN_TRAVEL_TIME_S) &
        (df["tiempo_viaje_segundos"] <= MAX_TRAVEL_TIME_S)
    ]
    log_drop(n_before, len(df), f"tiempo_viaje_segundos fuera de ({MIN_TRAVEL_TIME_S}, {MAX_TRAVEL_TIME_S}]")

    # ── 3e. Outliers en retraso: |retraso| > 2 horas (datos corruptos) ───────
    n_before = len(df)
    df = df[df["retraso_actual_segundos"].abs() <= 7_200]
    log_drop(n_before, len(df), "|retraso_actual_segundos| > 7200s")

    # ── 3f. Nulos en time_point_order ────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["time_point_order"])
    log_drop(n_before, len(df), "time_point_order nulo")

    # ── 3g. Nulos en features categóricas ────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    log_drop(n_before, len(df), "nulos en features del modelo")

    # Reset index limpio
    df = df.reset_index(drop=True)
    log_dataframe_stats(df, "POST_LIMPIEZA")

    # Distribución del target para validar sanidad
    logger.info(
        "  📈 Target final | min=%.1fs | p25=%.1fs | p50=%.1fs | p75=%.1fs | max=%.1fs",
        df[TARGET].quantile(0.00),
        df[TARGET].quantile(0.25),
        df[TARGET].quantile(0.50),
        df[TARGET].quantile(0.75),
        df[TARGET].quantile(1.00),
    )

    return df


# ---------------------------------------------------------------------------
# Paso 4: Preparación de Features para XGBoost
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Convierte las variables categóricas al tipo 'category' requerido por
    XGBoost enable_categorical=True, y separa features del target.

    Args:
        df: DataFrame limpio.

    Returns:
        (X, y): Tuple con el DataFrame de features y la Serie del target.
    """
    logger.info("=" * 70)
    logger.info("PASO 4: PREPARACIÓN DE FEATURES")
    logger.info("=" * 70)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    # Convertir a tipo category (necesario para enable_categorical=True en XGBoost)
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
        n_cats = X[col].nunique()
        logger.info("  🏷️  '%s' → category | cardinalidad: %d categorías únicas", col, n_cats)

    # Convertir numéricas a float32 para eficiencia de memoria
    for col in NUMERIC_FEATURES:
        X[col] = X[col].astype("float32")
        logger.info("  🔢 '%s' → float32 | rango: [%.0f, %.0f]", col, X[col].min(), X[col].max())

    logger.info("  ✅ Feature matrix lista: shape=%s", X.shape)
    logger.info("  ✅ Target vector: shape=%s | media=%.2fs | std=%.2fs",
                y.shape, y.mean(), y.std())

    return X, y


# ---------------------------------------------------------------------------
# Paso 5: Split y Entrenamiento
# ---------------------------------------------------------------------------

def train(X: pd.DataFrame, y: pd.Series) -> xgb.XGBRegressor:
    """
    Divide los datos en Train/Validación (80/20) y entrena el XGBRegressor con
    early stopping y logging de eval_metric cada 50 árboles.

    Args:
        X: DataFrame de features.
        y: Serie del target.

    Returns:
        Modelo XGBRegressor entrenado.
    """
    logger.info("=" * 70)
    logger.info("PASO 5: ENTRENAMIENTO CON EARLY STOPPING")
    logger.info("=" * 70)

    # ── Split estratificado no aplica a regresión, usamos shuffle temporal ───
    # Usamos random_state=42 para reproducibilidad.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, shuffle=True
    )
    logger.info(
        "  📊 Train: %d filas | Validación: %d filas | Split: 80/20",
        len(X_train), len(X_val),
    )

    # ── Inicializar modelo ────────────────────────────────────────────────────
    model = xgb.XGBRegressor(**XGB_PARAMS)
    logger.info("  🤖 Hiperparámetros del modelo:")
    for k, v in XGB_PARAMS.items():
        logger.info("     %-30s = %s", k, v)

    # ── Entrenamiento con eval_set y verbose=50 ───────────────────────────────
    # verbose=50 → imprime RMSE cada 50 árboles en stdout del proceso
    logger.info("  🚀 Iniciando entrenamiento XGBoost ...")
    t_start = time.time()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=50,     # log cada 50 árboles: [train-rmse + val-rmse]
    )

    elapsed = time.time() - t_start
    best_iter = model.best_iteration
    best_score = model.best_score

    logger.info("  ✅ Entrenamiento completado en %.1fs", elapsed)
    logger.info("  🏆 Mejor iteración: %d | Mejor RMSE en val: %.4f", best_iter, best_score)

    return model


# ---------------------------------------------------------------------------
# Paso 6: Guardar Modelo
# ---------------------------------------------------------------------------

def save_model(model: xgb.XGBRegressor) -> None:
    """
    Guarda el modelo en formato JSON nativo de XGBoost.
    El formato JSON es portátil, versionable y legible.
    """
    logger.info("=" * 70)
    logger.info("PASO 6: GUARDADO DEL MODELO")
    logger.info("=" * 70)

    model.save_model(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
    logger.info("  ✅ Modelo guardado en: %s (%.2f MB)", MODEL_PATH, size_mb)
    logger.info("  ℹ️  Formato: JSON nativo XGBoost (portable + versionable)")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║     MBTA TRANSIT XGBoost — Pipeline de Entrenamiento MLOps      ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("Timestamp de inicio: %s", pd.Timestamp.now().isoformat())

    try:
        # 1. Carga
        df = load_data()

        # 2. Feature Engineering
        df = engineer_features(df)

        # 3. Limpieza
        df = clean_data(df)

        # Si queda muy poco dato tras la limpieza, abortamos
        if len(df) < 10_000:
            logger.critical(
                "❌ Solo quedan %d filas tras la limpieza. "
                "Revisa el dataset o los umbrales de filtrado.",
                len(df),
            )
            sys.exit(1)

        # 4. Preparación de features
        X, y = prepare_features(df)

        # Liberamos el DataFrame original de memoria
        del df
        gc.collect()
        log_memory_usage("después de liberar df")

        # 5. Entrenamiento
        model = train(X, y)

        # 6. Guardar modelo
        save_model(model)

        total_elapsed = time.time() - total_start
        logger.info("=" * 70)
        logger.info("🎉 Pipeline completado exitosamente en %.1fs (%.1f min)",
                    total_elapsed, total_elapsed / 60)
        logger.info("   Modelo disponible en: %s", MODEL_PATH)

    except FileNotFoundError as exc:
        logger.critical("❌ Archivo no encontrado: %s", exc)
        sys.exit(1)
    except ValueError as exc:
        logger.critical("❌ Error de validación de datos: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("❌ Error inesperado: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
