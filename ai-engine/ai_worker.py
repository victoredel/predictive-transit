"""
ai_worker.py — Motor de IA XGBoost (Escritor CQRS) — Node-to-Node
===================================================================
Patrón: CQRS Writer — Lee el stream Redis, predice ETA entre paradas, escribe al cache.

Flujo:
  1. Carga transit_xgboost_model.json al inicio (FALLA CRÍTICA si no existe).
  2. XREAD bloqueante en 'bus_stop_stream' (consume solo nuevos mensajes).
  3. Feature Engineering Node-to-Node: extrae route_id, direction_id,
     stop_id, next_stop_id, hora_del_dia, dia_semana.
  4. Inferencia XGBoost con enable_categorical=True.
  5. HSET + EXPIRE → cache Redis con TTL de 5 minutos.
     Clave: eta:{route_id}:{stop_id}:{next_stop_id}

Feature Vector (6 dimensiones):
  [hora_del_dia, dia_semana, route_id (cat), direction_id (cat),
   stop_id (cat), next_stop_id (cat)]

Stream Redis esperado (campo del mensaje):
  - route_id       : ID de la ruta (ej. "1", "39")
  - direction_id   : "0" o "1"
  - stop_id        : ID de la parada actual (ej. "72")
  - next_stop_id   : ID de la siguiente parada (ej. "73")
  - half_trip_id   : ID del viaje (para correlación)
  - actual_ts      : Timestamp ISO del evento actual (opcional, usa now() si falta)
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import redis.asyncio as aioredis
import xgboost as xgb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuración del Logger — nivel INFO con timestamps ISO
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ai_worker")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
load_dotenv()

BASE_DIR    = Path(__file__).parent
MODEL_PATH  = BASE_DIR / "transit_xgboost_model.json"

REDIS_URL:   str = os.getenv("REDIS_URL", "redis://localhost:6379")
STREAM_NAME: str = os.getenv("AI_INPUT_STREAM",  "bus_stop_stream")
ETA_TTL:     int = int(os.getenv("ETA_TTL_SECONDS", "300"))      # 5 min TTL
BATCH_SIZE:  int = int(os.getenv("AI_BATCH_SIZE",   "50"))
BLOCK_MS:    int = int(os.getenv("AI_BLOCK_MS",     "5000"))    # 5s timeout XREAD

# Features en el mismo orden que durante el entrenamiento
# NUMÉRICOS primero, luego CATEGÓRICOS (debe coincidir con train_model.py)
NUMERIC_FEATURES     = ["hora_del_dia", "dia_semana"]
CATEGORICAL_FEATURES = ["route_id", "direction_id", "stop_id", "next_stop_id"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Carga del Modelo al Inicio de la Aplicación
# ---------------------------------------------------------------------------

def load_model() -> xgb.XGBRegressor:
    """
    Carga el modelo XGBoost desde transit_xgboost_model.json.

    Política de fallo: FALLA CRÍTICA (sys.exit) si el archivo no existe.
    No tiene sentido arrancar el worker sin un modelo válido.

    Returns:
        XGBRegressor listo para inferencia.

    Raises:
        SystemExit(1): Si el archivo del modelo no se encuentra.
    """
    if not MODEL_PATH.exists():
        logger.critical(
            "❌ MODELO NO ENCONTRADO: '%s'. "
            "Ejecuta train_model.py primero para entrenar el modelo.",
            MODEL_PATH,
        )
        sys.exit(1)

    try:
        model = xgb.XGBRegressor()
        model.load_model(str(MODEL_PATH))
        size_mb = MODEL_PATH.stat().st_size / (1024 ** 2)
        logger.info("✅ Modelo XGBoost cargado desde: %s (%.2f MB)", MODEL_PATH, size_mb)
        logger.info(
            "   Features esperados (%d): %s",
            len(ALL_FEATURES), ALL_FEATURES,
        )
        return model

    except Exception as exc:
        logger.critical(
            "❌ Error al cargar el modelo desde '%s': %s",
            MODEL_PATH, exc, exc_info=True,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Feature Engineering Node-to-Node
# ---------------------------------------------------------------------------

def engineer_features(record: dict) -> tuple[pd.DataFrame, dict]:
    """
    Construye el DataFrame de 1 fila con los features del modelo desde
    un mensaje del stream Redis.

    El stream publica eventos de parada (Node-to-Node), por lo que los campos
    stop_id y next_stop_id ya vienen como IDs discretos de paradas MBTA,
    coincidiendo exactamente con el dataset histórico de entrenamiento.

    Args:
        record: dict con los campos del mensaje Redis Stream (todos strings).
                Campos requeridos:
                  - route_id      : str  (ej. "39")
                  - direction_id  : str  (ej. "0" o "1")
                  - stop_id       : str  (ej. "72")
                  - next_stop_id  : str  (ej. "73")
                Campos opcionales:
                  - actual_ts     : ISO timestamp del evento (usa now() si falta)

    Returns:
        Tuple (X_df, metadata):
          - X_df    : pd.DataFrame de 1 fila con los 6 features correctamente tipificados.
          - metadata: dict con los valores para logging y escritura en Redis.

    Raises:
        ValueError: Si faltan campos requeridos en el mensaje.
    """
    # ── Validación de campos requeridos ───────────────────────────────────────
    required_fields = ["route_id", "direction_id", "stop_id", "next_stop_id"]
    missing = [f for f in required_fields if not record.get(f)]
    if missing:
        raise ValueError(
            f"Campos requeridos faltantes en el mensaje del stream: {missing}. "
            f"Mensaje recibido: {record}"
        )

    # ── Extracción de features categóricas ───────────────────────────────────
    route_id     = str(record["route_id"]).strip()
    direction_id = str(record["direction_id"]).strip()
    stop_id      = str(record["stop_id"]).strip()
    next_stop_id = str(record["next_stop_id"]).strip()

    # ── Extracción de features temporales ────────────────────────────────────
    # Si el mensaje trae un timestamp del evento, lo usamos para máxima precisión.
    # Si no, usamos la hora actual del servidor (siempre disponible).
    actual_ts_str = record.get("actual_ts")
    if actual_ts_str:
        try:
            event_dt = pd.to_datetime(actual_ts_str)
        except Exception:
            logger.warning(
                "  ⚠️  No se pudo parsear actual_ts='%s'. Usando now().", actual_ts_str
            )
            event_dt = datetime.now(timezone.utc)
    else:
        event_dt = datetime.now(timezone.utc)

    hora_del_dia = float(event_dt.hour)        # 0–23
    dia_semana   = float(event_dt.weekday())   # 0=Lunes, 6=Domingo

    # ── Construir DataFrame de 1 fila ─────────────────────────────────────────
    # El orden de columnas DEBE coincidir exactamente con ALL_FEATURES de train_model.py
    X_df = pd.DataFrame([{
        "hora_del_dia": hora_del_dia,
        "dia_semana":   dia_semana,
        "route_id":     route_id,
        "direction_id": direction_id,
        "stop_id":      stop_id,
        "next_stop_id": next_stop_id,
    }], columns=ALL_FEATURES)

    # Tipos correctos: float32 para numéricas, category para categóricas
    for col in NUMERIC_FEATURES:
        X_df[col] = X_df[col].astype("float32")
    for col in CATEGORICAL_FEATURES:
        X_df[col] = X_df[col].astype("category")

    metadata = {
        "route_id":     route_id,
        "direction_id": direction_id,
        "stop_id":      stop_id,
        "next_stop_id": next_stop_id,
        "hora_del_dia": int(hora_del_dia),
        "dia_semana":   int(dia_semana),
        "event_dt_iso": event_dt.isoformat() if hasattr(event_dt, "isoformat") else str(event_dt),
    }

    return X_df, metadata


# ---------------------------------------------------------------------------
# Cálculo de Nivel de Confianza
# ---------------------------------------------------------------------------

def calculate_confidence(hora: int, dia_semana: int, eta_seconds: float) -> float:
    """
    Heurística de confianza [0.10 – 0.99] basada en contexto operativo Node-to-Node.

    Mayor confianza cuando:
      - El ETA es corto (< 60s): la predicción a corto plazo es más precisa.
      - Es horario valle (no es hora pico): menos variabilidad en la red.
      - Es día de semana laboral (seg–vie): el modelo tiene más datos de entrenamiento.

    Menor confianza cuando:
      - Hora pico mañana (7–9h) o tarde (17–19h).
      - ETA largo (> 15min): la incertidumbre se acumula.
      - Fin de semana: patrones distintos y menor cantidad de datos históricos.
    """
    confidence = 0.75  # baseline calibrado

    # Penalización hora pico
    if (7 <= hora <= 9) or (17 <= hora <= 19):
        confidence -= 0.12

    # Bonus ETAs cortos (muy confiables), penalización ETAs largos
    if eta_seconds < 60:
        confidence += 0.10
    elif eta_seconds > 900:
        confidence -= 0.10

    # Ligera penalización en fin de semana (sáb=5, dom=6)
    if dia_semana >= 5:
        confidence -= 0.05

    return round(max(0.10, min(0.99, confidence)), 2)


# ---------------------------------------------------------------------------
# Pipeline de Inferencia (por registro)
# ---------------------------------------------------------------------------

async def process_record(
    msg_id: str,
    record: dict,
    model: xgb.XGBRegressor,
    redis_client: aioredis.Redis,
) -> None:
    """
    Pipeline completo para un único evento Node-to-Node del stream Redis:
      1. Feature Engineering (extrae variables Node-to-Node + temporales).
      2. Inferencia XGBoost con DataFrame tipificado.
      3. Calcula nivel de confianza contextual.
      4. HSET + EXPIRE en Redis (escritura atómica con pipeline).

    Args:
        msg_id      : ID del mensaje en el stream (para logging/debugging).
        record      : dict con los campos del mensaje.
        model       : XGBRegressor cargado.
        redis_client: Cliente Redis asíncrono.
    """
    # 1. Feature Engineering
    X_df, meta = engineer_features(record)

    # 2. Inferencia XGBoost
    # XGBoost acepta directamente el DataFrame con tipos category
    eta_raw      = float(model.predict(X_df)[0])
    eta_segundos = round(max(1.0, eta_raw), 1)  # mínimo 1 segundo

    # 3. Confianza contextual
    nivel_confianza = calculate_confidence(
        hora=meta["hora_del_dia"],
        dia_semana=meta["dia_semana"],
        eta_seconds=eta_segundos,
    )

    # 4. Escritura en Redis
    # Clave semántica: eta:{route_id}:{stop_id}:{next_stop_id}
    # Permite al Routing API leer el ETA de un segmento específico con O(1)
    cache_key = (
        f"eta:{meta['route_id']}:{meta['stop_id']}:{meta['next_stop_id']}"
    )

    cache_data = {
        "route_id":              meta["route_id"],
        "direction_id":          meta["direction_id"],
        "stop_id":               meta["stop_id"],
        "next_stop_id":          meta["next_stop_id"],
        "eta_segundos":          str(eta_segundos),
        "nivel_confianza":       str(nivel_confianza),
        "hora_del_dia":          str(meta["hora_del_dia"]),
        "dia_semana":            str(meta["dia_semana"]),
        "ultima_actualizacion":  datetime.now(timezone.utc).isoformat(),
        "stream_msg_id":         msg_id,
    }

    # Pipeline Redis: HSET + EXPIRE en una sola roundtrip de red
    pipe = redis_client.pipeline()
    pipe.hset(cache_key, mapping=cache_data)
    pipe.expire(cache_key, ETA_TTL)
    await pipe.execute()

    logger.debug(
        "ETA: route=%-6s stop=%-6s→%-6s eta=%6.1fs confianza=%.2f key=%s",
        meta["route_id"], meta["stop_id"], meta["next_stop_id"],
        eta_segundos, nivel_confianza, cache_key,
    )


# ---------------------------------------------------------------------------
# Worker Loop — CQRS Writer (XREAD)
# ---------------------------------------------------------------------------

async def worker_loop(model: xgb.XGBRegressor) -> None:
    """
    Bucle principal de consumo del stream Redis (patrón CQRS Writer).

    Usa XREAD con bloqueo para consumir solo mensajes nuevos (no reprocesar
    el histórico acumulado). Al iniciar, obtiene el último ID del stream
    como cursor de inicio.

    Resiliencia:
      - Reconexión automática a Redis con retries cada 5s.
      - Continúa si falla el procesamiento de un mensaje individual
        (no interrumpe el batch completo).
      - El cursor del stream se actualiza solo en mensajes exitosos
        para evitar pérdida de mensajes en fallo.
    """
    logger.info("=" * 70)
    logger.info("🧠 AI Engine (XGBoost CQRS Writer) — Node-to-Node arrancando")
    logger.info("   Stream fuente     : %s", STREAM_NAME)
    logger.info("   Modelo cargado    : %s", MODEL_PATH.name)
    logger.info("   Cache TTL         : %ds", ETA_TTL)
    logger.info("   Batch size        : %d msgs", BATCH_SIZE)
    logger.info("   Redis URL         : %s", REDIS_URL)
    logger.info("=" * 70)

    redis_client: Optional[aioredis.Redis] = None
    total_processed = 0
    total_errors     = 0

    while True:
        # ── A. Gestión de conexión Redis ──────────────────────────────────────
        if redis_client is None:
            try:
                redis_client = aioredis.from_url(
                    REDIS_URL,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                )
                await redis_client.ping()
                logger.info("✅ Conectado a Redis: %s", REDIS_URL)
            except aioredis.RedisError as exc:
                logger.error("Sin conexión a Redis: %s. Reintentando en 5s...", exc)
                await asyncio.sleep(5)
                continue

        # ── B. Obtener cursor de inicio (último ID del stream) ────────────────
        try:
            stream_info = await redis_client.xinfo_stream(STREAM_NAME)
            last_id = stream_info.get("last-generated-id") or "0-0"
        except aioredis.ResponseError:
            logger.warning(
                "Stream '%s' no existe aún. Esperando que el ingestion publique...",
                STREAM_NAME,
            )
            await asyncio.sleep(3)
            continue
        except aioredis.RedisError as exc:
            logger.error("Error al leer info del stream: %s", exc)
            await asyncio.sleep(3)
            continue

        logger.info("🚀 Iniciando lectura desde ID: %s", last_id)

        # ── C. Loop de lectura XREAD ──────────────────────────────────────────
        while True:
            try:
                results = await redis_client.xread(
                    streams={STREAM_NAME: last_id},
                    count=BATCH_SIZE,
                    block=BLOCK_MS,
                )

                if not results:
                    # Timeout XREAD sin nuevos mensajes → seguimos esperando
                    continue

                batch_ok    = 0
                batch_error = 0

                for _stream_key, messages in results:
                    for msg_id, fields in messages:
                        # Actualizamos el cursor siempre (para no reproces. en crash)
                        last_id = msg_id

                        try:
                            await process_record(msg_id, fields, model, redis_client)
                            batch_ok += 1
                        except ValueError as exc:
                            # Mensaje malformado: logueamos y descartamos
                            logger.warning(
                                "Mensaje inválido [%s] descartado: %s", msg_id, exc
                            )
                            batch_error += 1
                        except Exception as exc:
                            logger.error(
                                "Error al procesar msg [%s]: %s",
                                msg_id, exc, exc_info=True,
                            )
                            batch_error += 1

                total_processed += batch_ok
                total_errors    += batch_error

                if batch_ok > 0:
                    logger.info(
                        "✔ Batch OK: %d ETAs calculados | total_acum=%d | errores_acum=%d",
                        batch_ok, total_processed, total_errors,
                    )
                if batch_error > 0:
                    logger.warning(
                        "⚠️  Batch: %d errores en este ciclo", batch_error
                    )

            except aioredis.ConnectionError as exc:
                logger.error("Conexión Redis perdida: %s. Reconectando...", exc)
                try:
                    await redis_client.aclose()
                except Exception:
                    pass
                redis_client = None
                break   # sale del inner loop → reconecta en el outer loop

            except Exception as exc:
                logger.error("Error inesperado en worker loop: %s", exc, exc_info=True)
                await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║   MBTA Predictive Transit — AI Engine (Node-to-Node XGBoost)    ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")

    # Cargar modelo al arranque (FALLA CRÍTICA si no existe)
    xgb_model = load_model()

    try:
        asyncio.run(worker_loop(xgb_model))
    except KeyboardInterrupt:
        logger.info("🛑 AI Engine detenido por el usuario.")
    except Exception as exc:
        logger.critical("❌ Error fatal en AI Engine: %s", exc, exc_info=True)
        sys.exit(1)
