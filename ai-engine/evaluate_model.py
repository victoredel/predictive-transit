"""
evaluate_model.py — Evaluación Exhaustiva del Modelo XGBoost (Hold-out Test)
=============================================================================
Carga el modelo entrenado (transit_xgboost_model.json) y el mes de Abril 2025
como conjunto de Test completamente no visto durante el entrenamiento.

Métricas reportadas:
  - MAE   : Mean Absolute Error (segundos)
  - RMSE  : Root Mean Squared Error (segundos)
  - R²    : Coeficiente de determinación
  - MAPE  : Mean Absolute Percentage Error (%)
  - Accuracy@60s  : % de predicciones con |error| < 60 segundos
  - Accuracy@120s : % de predicciones con |error| < 120 segundos
  - Feature Importance (tipo "gain") ordenada descendente

Uso:
  python evaluate_model.py

Prerequisito:
  Haber ejecutado train_model.py para generar transit_xgboost_model.json
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuración del Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate_model")

# ---------------------------------------------------------------------------
# Constantes (deben ser idénticas a train_model.py)
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "MBTA_Bus_Arrival_Departure_Times_2025"
MODEL_PATH = BASE_DIR / "transit_xgboost_model.json"

# Mes de Test: Abril 2025 (hold-out, jamás visto en entrenamiento)
TEST_MONTH = "2025-04"

REQUIRED_COLS = [
    "service_date", "route_id", "direction_id",
    "half_trip_id", "stop_id", "time_point_order",
    "scheduled", "actual",
]

CATEGORICAL_FEATURES = ["route_id", "direction_id", "stop_id", "next_stop_id"]
NUMERIC_FEATURES     = ["hora_del_dia", "dia_semana"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET               = "tiempo_viaje_segundos"

MIN_TRAVEL_TIME_S = 0
MAX_TRAVEL_TIME_S = 3_600


# ---------------------------------------------------------------------------
# Carga del Modelo
# ---------------------------------------------------------------------------

def load_model() -> xgb.XGBRegressor:
    """
    Carga el modelo XGBoost desde el archivo JSON.
    Lanza un error crítico y termina el proceso si el archivo no existe,
    ya que no tiene sentido continuar sin modelo entrenado.

    Returns:
        XGBRegressor cargado y listo para predicción.

    Raises:
        SystemExit: Si el archivo del modelo no existe.
    """
    logger.info("=" * 70)
    logger.info("PASO 1: CARGA DEL MODELO")
    logger.info("=" * 70)

    if not MODEL_PATH.exists():
        logger.critical(
            "❌ MODELO NO ENCONTRADO: '%s'. "
            "Ejecuta train_model.py primero para generar el modelo.",
            MODEL_PATH,
        )
        sys.exit(1)

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)

    logger.info("  ✅ Modelo cargado desde: %s (%.2f MB)", MODEL_PATH, size_mb)
    logger.info("  ℹ️  Configuración del modelo cargado:")
    params = model.get_params()
    for k in ["n_estimators", "max_depth", "learning_rate", "tree_method", "enable_categorical"]:
        logger.info("     %-30s = %s", k, params.get(k, "N/A"))

    return model


# ---------------------------------------------------------------------------
# Carga del Test Set (Hold-out)
# ---------------------------------------------------------------------------

def load_test_data() -> pd.DataFrame:
    """
    Carga el archivo CSV del mes de Test (Abril 2025).
    Aplica el mismo pipeline de Feature Engineering y limpieza que en training,
    garantizando la coherencia del pipeline.

    Returns:
        DataFrame limpio con features y target listo para evaluación.

    Raises:
        FileNotFoundError: Si no se encuentra el CSV del mes de test.
    """
    logger.info("=" * 70)
    logger.info("PASO 2: CARGA Y PREPARACIÓN DEL TEST SET (Hold-out)")
    logger.info("=" * 70)

    # Buscar archivo del mes de test
    import glob as glob_mod
    pattern = str(DATA_DIR / f"*{TEST_MONTH}*.csv")
    matched = glob_mod.glob(pattern)

    if not matched:
        raise FileNotFoundError(
            f"No se encontró el archivo del mes de test '{TEST_MONTH}' "
            f"con el patrón '{pattern}'."
        )

    fpath = matched[0]
    logger.info("  📂 Cargando test set: %s", Path(fpath).name)
    t0 = time.time()

    dtype_map = {
        "route_id":         "str",
        "direction_id":     "str",
        "half_trip_id":     "str",
        "stop_id":          "str",
        "time_point_order": "Int64",
    }
    df = pd.read_csv(fpath, usecols=REQUIRED_COLS, dtype=dtype_map, low_memory=False)

    logger.info("  ✅ Cargado: %d filas en %.2fs | shape=%s",
                len(df), time.time() - t0, df.shape)

    # ── Feature Engineering (replicado exactamente de train_model.py) ────────
    logger.info("  Aplicando Feature Engineering ...")

    for col in ("actual", "scheduled"):
        df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)

    df["hora_del_dia"] = df["actual"].dt.hour.astype("Int64")
    df["dia_semana"]   = df["actual"].dt.dayofweek.astype("Int64")

    df.sort_values(["half_trip_id", "time_point_order"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    grouped = df.groupby("half_trip_id", sort=False)
    df["next_actual_time"] = grouped["actual"].shift(-1)
    df["next_stop_id"]     = grouped["stop_id"].shift(-1)

    df["tiempo_viaje_segundos"] = (
        df["next_actual_time"] - df["actual"]
    ).dt.total_seconds()
    df["retraso_actual_segundos"] = (
        df["actual"] - df["scheduled"]
    ).dt.total_seconds()

    # ── Limpieza (replicada exactamente de train_model.py) ────────────────────
    logger.info("  Aplicando limpieza ...")

    n_original = len(df)
    df = df.dropna(subset=["actual", "scheduled"])
    df = df.dropna(subset=["next_actual_time", "next_stop_id"])
    df = df.dropna(subset=["tiempo_viaje_segundos"])
    df = df[(df["tiempo_viaje_segundos"] > MIN_TRAVEL_TIME_S) &
            (df["tiempo_viaje_segundos"] <= MAX_TRAVEL_TIME_S)]
    df = df[df["retraso_actual_segundos"].abs() <= 7_200]
    df = df.dropna(subset=["time_point_order"])
    df = df.dropna(subset=CATEGORICAL_FEATURES + NUMERIC_FEATURES)
    df = df.reset_index(drop=True)

    n_clean = len(df)
    logger.info(
        "  ✅ Limpieza: %d → %d filas | %.1f%% retenido",
        n_original, n_clean, (n_clean / n_original * 100) if n_original > 0 else 0,
    )

    # ── Preparar types para XGBoost ───────────────────────────────────────────
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()

    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("category")
    for col in NUMERIC_FEATURES:
        X[col] = X[col].astype("float32")

    logger.info("  ✅ Test set listo | X.shape=%s | y.shape=%s", X.shape, y.shape)

    return X, y


# ---------------------------------------------------------------------------
# Evaluación y Reporte de Métricas
# ---------------------------------------------------------------------------

def evaluate(model: xgb.XGBRegressor, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Ejecuta la inferencia y calcula el reporte completo de métricas.

    Métricas calculadas:
      - MAE   : Error promedio absoluto en segundos.
      - RMSE  : Penaliza más los errores grandes.
      - R²    : Qué proporción de la varianza explica el modelo.
      - MAPE  : % promedio de desviación respecto al valor real.
      - Acc@60s / Acc@120s : % de predicciones dentro de umbrales operativos.

    Returns:
        Dict con todas las métricas calculadas.
    """
    logger.info("=" * 70)
    logger.info("PASO 3: INFERENCIA Y CÁLCULO DE MÉTRICAS")
    logger.info("=" * 70)

    logger.info("  🔮 Ejecutando predicción sobre %d muestras ...", len(X))
    t0 = time.time()
    y_pred = model.predict(X)
    elapsed_inference = time.time() - t0
    throughput = len(X) / elapsed_inference

    logger.info(
        "  ✅ Inferencia completada: %.3fs | throughput=%.0f registros/s",
        elapsed_inference, throughput,
    )

    # Protección contra división por cero en MAPE
    mask_nonzero = y != 0
    y_true_nz = y[mask_nonzero]
    y_pred_nz = y_pred[mask_nonzero]

    # ── Cálculo de métricas ───────────────────────────────────────────────────
    mae   = mean_absolute_error(y, y_pred)
    rmse  = np.sqrt(mean_squared_error(y, y_pred))
    r2    = r2_score(y, y_pred)
    mape  = np.mean(np.abs((y_true_nz - y_pred_nz) / y_true_nz)) * 100

    errors_abs = np.abs(y.values - y_pred)
    acc_60s    = (errors_abs < 60).sum()  / len(y) * 100
    acc_120s   = (errors_abs < 120).sum() / len(y) * 100

    # ── Distribución de errores ───────────────────────────────────────────────
    p50_err = np.percentile(errors_abs, 50)
    p90_err = np.percentile(errors_abs, 90)
    p95_err = np.percentile(errors_abs, 95)
    p99_err = np.percentile(errors_abs, 99)

    metrics = {
        "n_samples":    len(y),
        "mae_s":        mae,
        "rmse_s":       rmse,
        "r2":           r2,
        "mape_pct":     mape,
        "acc_60s_pct":  acc_60s,
        "acc_120s_pct": acc_120s,
        "p50_error_s":  p50_err,
        "p90_error_s":  p90_err,
        "p95_error_s":  p95_err,
        "p99_error_s":  p99_err,
        "throughput_rps": throughput,
    }

    # ── Reporte en consola (formato de tabla) ─────────────────────────────────
    divider = "─" * 70

    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║        REPORTE DE EVALUACIÓN — MBTA Bus XGBoost Model               ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Test set       : Mes {TEST_MONTH} (Hold-out, nunca visto en training)   ║")
    print(f"║  Muestras       : {len(y):,}                                        ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║  MÉTRICAS PRIMARIAS                                                  ║")
    print(divider)
    print(f"  {'MAE (segundos)':<40} {mae:>10.2f} s")
    print(f"  {'RMSE (segundos)':<40} {rmse:>10.2f} s")
    print(f"  {'R² Score':<40} {r2:>10.4f}")
    print(f"  {'MAPE (% desviación promedio)':<40} {mape:>10.2f} %")
    print(divider)
    print("║  ANÁLISIS DE ERRORES (Umbrales Operativos)                          ║")
    print(divider)
    print(f"  {'Accuracy < 60s  (1 minuto)':<40} {acc_60s:>10.2f} %")
    print(f"  {'Accuracy < 120s (2 minutos)':<40} {acc_120s:>10.2f} %")
    print(divider)
    print("║  DISTRIBUCIÓN DEL ERROR ABSOLUTO (Percentiles)                      ║")
    print(divider)
    print(f"  {'P50 (mediana)':<40} {p50_err:>10.2f} s")
    print(f"  {'P90':<40} {p90_err:>10.2f} s")
    print(f"  {'P95':<40} {p95_err:>10.2f} s")
    print(f"  {'P99':<40} {p99_err:>10.2f} s")
    print(divider)
    print(f"  {'Throughput de inferencia':<40} {throughput:>10,.0f} rec/s")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print("\n")

    # Loguear también vía logger para persistencia en logs del sistema
    logger.info("MAE=%.2fs | RMSE=%.2fs | R²=%.4f | MAPE=%.2f%%", mae, rmse, r2, mape)
    logger.info("Accuracy@60s=%.2f%% | Accuracy@120s=%.2f%%", acc_60s, acc_120s)

    return metrics, y_pred


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def report_feature_importance(model: xgb.XGBRegressor) -> None:
    """
    Imprime una tabla ordenada con la importancia de cada feature según la
    métrica 'gain' (ganancia media por split = señal real del feature).

    'gain' es la métrica más relevante en XGBoost ya que representa la
    mejora en el criterio de separación que cada feature aporta en promedio.
    """
    logger.info("=" * 70)
    logger.info("PASO 4: FEATURE IMPORTANCE (tipo 'gain')")
    logger.info("=" * 70)

    # Obtener importancias tipo 'gain' como dict
    importance_dict = model.get_booster().get_score(importance_type="gain")

    if not importance_dict:
        logger.warning("  ⚠️  No hay datos de importancia disponibles (modelo sin árboles?).")
        return

    # Construir DataFrame ordenado
    fi_df = (
        pd.DataFrame.from_dict(importance_dict, orient="index", columns=["gain"])
        .sort_values("gain", ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    fi_df["gain_pct"] = fi_df["gain"] / fi_df["gain"].sum() * 100

    # Tabla de text en consola
    print("\n")
    print("╔══════════════════════════════════════════════════════╗")
    print("║   FEATURE IMPORTANCE — Tipo: Gain (mayor = mejor)   ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"  {'Rank':<5} {'Feature':<25} {'Gain':>12} {'Gain %':>10}")
    print("  " + "─" * 55)
    for i, row in fi_df.iterrows():
        bar = "█" * int(row["gain_pct"] / 2)  # barra proporcional
        print(f"  {i+1:<5} {row['feature']:<25} {row['gain']:>12.2f} {row['gain_pct']:>9.2f}%  {bar}")
    print("╚══════════════════════════════════════════════════════╝")
    print("\n")

    logger.info("  Top feature: '%s' (gain=%.2f, %.2f%% del total)",
                fi_df.iloc[0]["feature"],
                fi_df.iloc[0]["gain"],
                fi_df.iloc[0]["gain_pct"])


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║   MBTA TRANSIT XGBoost — Evaluación Exhaustiva del Modelo       ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")

    try:
        # 1. Cargar modelo
        model = load_model()

        # 2. Cargar y preparar test set
        X_test, y_test = load_test_data()

        # 3. Evaluar métricas
        metrics, y_pred = evaluate(model, X_test, y_test)

        # 4. Feature importance
        report_feature_importance(model)

        total_elapsed = time.time() - total_start
        logger.info("🎉 Evaluación completada en %.1fs", total_elapsed)

        # Diagnóstico rápido de calidad del modelo
        mae_val = metrics["mae_s"]
        r2_val  = metrics["r2"]

        if mae_val < 60 and r2_val > 0.70:
            logger.info("✅ Modelo de BUENA calidad: MAE<60s y R²>0.70")
        elif mae_val < 120 and r2_val > 0.50:
            logger.warning("⚠️  Modelo ACEPTABLE: MAE<120s pero puede mejorar.")
        else:
            logger.warning(
                "❌ Modelo POBRE: MAE=%.1fs / R²=%.4f. "
                "Considera más datos o ajuste de hiperparámetros.",
                mae_val, r2_val,
            )

    except FileNotFoundError as exc:
        logger.critical("❌ Archivo no encontrado: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.critical("❌ Error inesperado: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
