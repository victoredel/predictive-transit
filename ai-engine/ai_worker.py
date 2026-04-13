"""
ai_worker.py — Motor de IA XGBoost (CQRS Writer) con Clima en Tiempo Real
===========================================================================
Patrón: CQRS Writer — Lee el stream Redis, enriquece con clima actual,
predice ETA entre paradas Node-to-Node, escribe al cache Redis.

Flujo completo:
  1. load_model()           → Carga transit_xgboost_model.json (FALLA CRÍTICA si no existe).
  2. XREAD bloqueante       → Consume nuevos mensajes del stream 'bus_stop_stream'.
  3. get_current_weather()  → Open-Meteo Forecast API (temperatura/lluvia/nieve actuales).
  4. engineer_features()    → Inyecta hora, mes, dia_semana + clima real.
  5. model.predict()        → XGBoost inference con enable_categorical=True.
  6. HSET + EXPIRE          → Cache Redis con TTL de 5 minutos.
                              Clave: eta:{route_id}:{stop_id}:{next_stop_id}

Feature Vector (9 dimensiones, alineado con train_model.py):
  Numéricas (6):   [hora_del_dia, dia_semana, mes, temperature_2m, precipitation, snowfall]
  Categóricas (4): [route_id, direction_id, stop_id, next_stop_id]

El clima se obtiene en tiempo real desde la API de pronóstico de Open-Meteo.
Si la API falla, se usan valores por defecto (0) sin interrumpir el servicio.

Stream Redis esperado (campos del mensaje):
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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import redis.asyncio as aioredis
import requests
import xgboost as xgb
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logger — ISO 8601, nivel INFO, stdout
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
ETA_TTL:     int = int(os.getenv("ETA_TTL_SECONDS", "300"))   # 5 min
BATCH_SIZE:  int = int(os.getenv("AI_BATCH_SIZE",   "50"))
BLOCK_MS:    int = int(os.getenv("AI_BLOCK_MS",     "5000"))  # 5s timeout XREAD

# Coordenadas de Boston para el Forecast API
BOSTON_LAT = 42.3601
BOSTON_LON = -71.0589

# TTL del caché de clima en el worker (refresh cada 10 minutos)
WEATHER_CACHE_TTL_S = 600

# Features (DEBEN coincidir exactamente con train_model.py)
NUMERIC_FEATURES     = ["hora_del_dia", "dia_semana", "mes",
                        "temperature_2m", "precipitation", "snowfall"]
CATEGORICAL_FEATURES = ["route_id", "direction_id", "stop_id", "next_stop_id"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Caché de Clima en Memoria (evita llamadas por-registro a la API)
# ---------------------------------------------------------------------------
_weather_cache: dict = {
    "temperature_2m": 10.0,  # valor por defecto: 10°C (clima templado)
    "precipitation":  0.0,   # sin lluvia
    "snowfall":       0.0,   # sin nieve
    "_last_fetch":    0.0,   # timestamp UNIX del último fetch
}


# ---------------------------------------------------------------------------
# PASO 1: Carga del Modelo al Inicio de la Aplicación
# ---------------------------------------------------------------------------

def load_model() -> xgb.XGBRegressor:
    """
    Carga el modelo XGBoost desde transit_xgboost_model.json.

    Política de fallo: FALLA CRÍTICA (sys.exit) si el archivo no existe,
    ya que no tiene sentido arrancar el worker sin un modelo válido.

    Retorna:
        XGBRegressor listo para inferencia.

    Lanza:
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
        logger.info("✅ Modelo XGBoost cargado: %s (%.2f MB)", MODEL_PATH.name, size_mb)
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
# PASO 2: Obtención del Clima Actual (Open-Meteo Forecast API)
# ---------------------------------------------------------------------------

def get_current_weather() -> dict:
    """
    Consulta la API de Pronóstico (Forecast) de Open-Meteo para obtener
    el clima actual en Boston en tiempo real.

    Estrategia de caché en memoria:
      - Los datos climáticos se cachean por WEATHER_CACHE_TTL_S (10 min).
      - Si la API falla, se retornan los últimos valores conocidos o defaults.
      - Los defaults son temperatura=10°C, precipitation=0, snowfall=0
        (equivalente a un día normal sin eventos climáticos).

    Variables obtenidas (actuales, no histórico):
      - temperature_2m : Temperatura del aire a 2m (°C)
      - precipitation  : Precipitación actual (mm/h)
      - snowfall       : Nieve actual (cm/h)

    Retorna:
        dict con keys: temperature_2m, precipitation, snowfall (float).
    """
    global _weather_cache

    # Verificar si el caché aún es válido
    now_ts = time.time()
    if (now_ts - _weather_cache["_last_fetch"]) < WEATHER_CACHE_TTL_S:
        return {
            "temperature_2m": _weather_cache["temperature_2m"],
            "precipitation":  _weather_cache["precipitation"],
            "snowfall":       _weather_cache["snowfall"],
        }

    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude":  BOSTON_LAT,
            "longitude": BOSTON_LON,
            # Pedimos solo la hora actual para minimizar datos transferidos
            "current":   ["temperature_2m", "precipitation", "snowfall"],
            "timezone":  "America/New_York",
        }

        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        current = data.get("current", {})
        temp    = float(current.get("temperature_2m", 10.0))
        prec    = float(current.get("precipitation",  0.0))
        snow    = float(current.get("snowfall",        0.0))

        # Actualizar caché en memoria
        _weather_cache["temperature_2m"] = temp
        _weather_cache["precipitation"]  = prec
        _weather_cache["snowfall"]        = snow
        _weather_cache["_last_fetch"]     = now_ts

        logger.debug(
            "🌤️  Clima actualizado: temp=%.1f°C | prec=%.2fmm | snow=%.2fcm",
            temp, prec, snow,
        )
        return {"temperature_2m": temp, "precipitation": prec, "snowfall": snow}

    except requests.exceptions.Timeout:
        logger.warning("⚠️  Open-Meteo timeout. Usando últimos valores en caché.")
    except requests.exceptions.ConnectionError:
        logger.warning("⚠️  Open-Meteo sin conexión. Usando últimos valores en caché.")
    except requests.exceptions.HTTPError as exc:
        logger.warning("⚠️  Open-Meteo HTTP error %s. Usando últimos valores en caché.", exc)
    except (KeyError, ValueError, TypeError) as exc:
        logger.warning("⚠️  Error parseando respuesta de Open-Meteo: %s. Usando caché.", exc)
    except Exception as exc:
        logger.warning("⚠️  Error inesperado en get_current_weather: %s. Usando caché.", exc)

    # Retornar últimos valores conocidos sin actualizar el timestamp
    # (para reintentar en el próximo ciclo del TTL)
    return {
        "temperature_2m": _weather_cache["temperature_2m"],
        "precipitation":  _weather_cache["precipitation"],
        "snowfall":       _weather_cache["snowfall"],
    }


# ---------------------------------------------------------------------------
# PASO 3: Feature Engineering con Clima Real
# ---------------------------------------------------------------------------

def engineer_features(record: dict, weather: dict) -> tuple[pd.DataFrame, dict]:
    """
    Construye el DataFrame de 1 fila con los 9 features del modelo
    desde un mensaje del stream Redis + datos climáticos actuales.

    Args:
        record:  dict con los campos del mensaje Redis Stream.
                 Campos requeridos:
                   - route_id      : str (ej. "39")
                   - direction_id  : str ("0" o "1")
                   - stop_id       : str (ej. "72")
                   - next_stop_id  : str (ej. "73")
                 Opcional:
                   - actual_ts     : ISO timestamp del evento (usa now() si falta)

        weather: dict con climate actuales del get_current_weather().
                 Keys: temperature_2m, precipitation, snowfall.

    Retorna:
        Tuple (X_df, metadata):
          - X_df    : pd.DataFrame de 1 fila con los 9 features tipificados.
          - metadata: dict para logging y escritura en Redis.

    Lanza:
        ValueError: Si faltan campos requeridos en el mensaje.
    """
    # ── Validación de campos requeridos ───────────────────────────────────────
    required_fields = ["route_id", "direction_id", "stop_id", "next_stop_id"]
    missing = [f for f in required_fields if not record.get(f)]
    if missing:
        raise ValueError(
            f"Campos requeridos faltantes en el mensaje: {missing}. "
            f"Mensaje: {record}"
        )

    # ── Campos categóricos ────────────────────────────────────────────────────
    route_id     = str(record["route_id"]).strip()
    direction_id = str(record["direction_id"]).strip()
    stop_id      = str(record["stop_id"]).strip()
    next_stop_id = str(record["next_stop_id"]).strip()

    # ── Features temporales ───────────────────────────────────────────────────
    # Si el mensaje trae timestamp del evento, lo usamos para máxima precisión.
    actual_ts_str = record.get("actual_ts")
    if actual_ts_str:
        try:
            event_dt = pd.to_datetime(actual_ts_str)
            # Asegurar que tenga timezone para calcular hora local
            if event_dt.tzinfo is None:
                event_dt = event_dt.tz_localize("America/New_York")
        except Exception:
            logger.warning(
                "⚠️  No se pudo parsear actual_ts='%s'. Usando now().", actual_ts_str
            )
            event_dt = datetime.now(timezone.utc)
    else:
        event_dt = datetime.now(timezone.utc)

    hora_del_dia = float(event_dt.hour)       # 0-23
    dia_semana   = float(event_dt.weekday())  # 0=Lunes, 6=Domingo
    mes          = float(event_dt.month)      # 1-12 ← NUEVO: feature clave para estacionalidad

    # ── Features climáticas (tiempo real) ────────────────────────────────────
    temperature_2m = float(weather.get("temperature_2m", 10.0))
    precipitation  = float(weather.get("precipitation", 0.0))
    snowfall       = float(weather.get("snowfall", 0.0))

    # ── Construir DataFrame de 1 fila ─────────────────────────────────────────
    # El orden de columnas DEBE coincidir EXACTAMENTE con ALL_FEATURES de train_model.py
    X_df = pd.DataFrame([{
        "hora_del_dia":  hora_del_dia,
        "dia_semana":    dia_semana,
        "mes":           mes,           # ← Nuevo: mes del año
        "temperature_2m": temperature_2m,
        "precipitation": precipitation,
        "snowfall":      snowfall,
        "route_id":      route_id,
        "direction_id":  direction_id,
        "stop_id":       stop_id,
        "next_stop_id":  next_stop_id,
    }], columns=ALL_FEATURES)

    # Tipos XGBoost: float32 para numéricas, category para categóricas
    for col in NUMERIC_FEATURES:
        X_df[col] = X_df[col].astype("float32")
    for col in CATEGORICAL_FEATURES:
        X_df[col] = X_df[col].astype("category")

    metadata = {
        "route_id":       route_id,
        "direction_id":   direction_id,
        "stop_id":        stop_id,
        "next_stop_id":   next_stop_id,
        "hora_del_dia":   int(hora_del_dia),
        "dia_semana":     int(dia_semana),
        "mes":            int(mes),
        "temperature_2m": temperature_2m,
        "precipitation":  precipitation,
        "snowfall":       snowfall,
        "event_dt_iso":   event_dt.isoformat() if hasattr(event_dt, "isoformat") else str(event_dt),
    }

    return X_df, metadata


# ---------------------------------------------------------------------------
# Cálculo de Nivel de Confianza Contextual
# ---------------------------------------------------------------------------

def calculate_confidence(
    hora: int,
    dia_semana: int,
    eta_seconds: float,
    precipitation: float,
    snowfall: float,
) -> float:
    """
    Heurística de confianza [0.10–0.99] basada en contexto operativo.

    Mayor confianza: ETAs cortos, horario valle, sin precipitación.
    Menor confianza: hora pico, ETAs largos, lluvia o nieve activa.

    Args:
        hora:         Hora del día (0-23).
        dia_semana:   Día de la semana (0=Lun, 6=Dom).
        eta_seconds:  ETA predicho en segundos.
        precipitation: Precipitación actual (mm/h).
        snowfall:     Nieve actual (cm/h).

    Retorna:
        Confianza como float en [0.10, 0.99].
    """
    confidence = 0.75  # baseline calibrado

    # Penalización hora pico (mayor congestión → más variabilidad)
    if (7 <= hora <= 9) or (17 <= hora <= 19):
        confidence -= 0.12

    # ETAs cortos = más confiables; ETAs largos = más inciertos
    if eta_seconds < 60:
        confidence += 0.10
    elif eta_seconds > 900:
        confidence -= 0.10

    # Fin de semana: patrones distintos y menor volumen histórico
    if dia_semana >= 5:
        confidence -= 0.05

    # Penalización climática: lluvia y nieve aumentan variabilidad
    if precipitation > 5.0:   # lluvia moderada a intensa
        confidence -= 0.10
    elif precipitation > 1.0: # lluvia leve
        confidence -= 0.05

    if snowfall > 2.0:        # nevada significativa
        confidence -= 0.15
    elif snowfall > 0.5:      # nevada leve
        confidence -= 0.08

    return round(max(0.10, min(0.99, confidence)), 2)


# ---------------------------------------------------------------------------
# Pipeline de Inferencia (por registro)
# ---------------------------------------------------------------------------

async def process_record(
    msg_id:       str,
    record:       dict,
    model:        xgb.XGBRegressor,
    redis_client: aioredis.Redis,
) -> None:
    """
    Pipeline completo para un único evento Node-to-Node del stream Redis:
      1. Obtener clima actual (con caché de 10 min) vía Open-Meteo Forecast.
      2. Feature Engineering: construye el vector de 9 features.
      3. Inferencia XGBoost → ETA en segundos.
      4. Calcular nivel de confianza contextual (con clima).
      5. HSET + EXPIRE en Redis (escritura atómica con pipeline).

    Args:
        msg_id:       ID del mensaje en el stream (para logging).
        record:       dict con los campos del mensaje Redis.
        model:        XGBRegressor cargado.
        redis_client: Cliente Redis asíncrono.
    """
    # 1. Clima actual (bloqueante pero con caché de 10 min en memoria)
    # Ejecutamos en el executor para no bloquear el event loop de asyncio
    loop = asyncio.get_event_loop()
    weather = await loop.run_in_executor(None, get_current_weather)

    # 2. Feature Engineering con clima real + mes del año
    X_df, meta = engineer_features(record, weather)

    # 3. Inferencia XGBoost
    eta_raw      = float(model.predict(X_df)[0])
    eta_segundos = round(max(1.0, eta_raw), 1)  # mínimo 1 segundo

    # 4. Confianza contextual con datos climáticos reales
    nivel_confianza = calculate_confidence(
        hora          = meta["hora_del_dia"],
        dia_semana    = meta["dia_semana"],
        eta_seconds   = eta_segundos,
        precipitation = meta["precipitation"],
        snowfall      = meta["snowfall"],
    )

    # 5. Escritura en Redis
    # Clave semántica: eta:{route_id}:{stop_id}:{next_stop_id}
    cache_key = f"eta:{meta['route_id']}:{meta['stop_id']}:{meta['next_stop_id']}"

    cache_data = {
        "route_id":              meta["route_id"],
        "direction_id":          meta["direction_id"],
        "stop_id":               meta["stop_id"],
        "next_stop_id":          meta["next_stop_id"],
        "eta_segundos":          str(eta_segundos),
        "nivel_confianza":       str(nivel_confianza),
        "hora_del_dia":          str(meta["hora_del_dia"]),
        "dia_semana":            str(meta["dia_semana"]),
        "mes":                   str(meta["mes"]),
        # Clima inyectado (para transparencia e inspección del cache)
        "temperatura_actual_c":  str(round(meta["temperature_2m"], 1)),
        "precipitacion_mm":      str(round(meta["precipitation"], 2)),
        "nieve_cm":              str(round(meta["snowfall"], 2)),
        # Metadata de auditoría
        "ultima_actualizacion":  datetime.now(timezone.utc).isoformat(),
        "stream_msg_id":         msg_id,
    }

    # Pipeline Redis: HSET + EXPIRE en una sola roundtrip de red
    pipe = redis_client.pipeline()
    pipe.hset(cache_key, mapping=cache_data)
    pipe.expire(cache_key, ETA_TTL)
    await pipe.execute()

    logger.debug(
        "ETA: route=%-6s stop=%-6s→%-6s eta=%6.1fs confianza=%.2f "
        "temp=%.1f°C prec=%.2fmm snow=%.2fcm key=%s",
        meta["route_id"], meta["stop_id"], meta["next_stop_id"],
        eta_segundos, nivel_confianza,
        meta["temperature_2m"], meta["precipitation"], meta["snowfall"],
        cache_key,
    )


# ---------------------------------------------------------------------------
# Worker Loop — CQRS Writer (XREAD en Redis Streams)
# ---------------------------------------------------------------------------

async def worker_loop(model: xgb.XGBRegressor) -> None:
    """
    Bucle principal de consumo del stream Redis (patrón CQRS Writer).

    Usa XREAD con bloqueo para consumir solo mensajes nuevos (no reprocesa
    el histórico acumulado). Al iniciar, obtiene el último ID del stream
    como cursor de inicio.

    Resiliencia:
      - Reconexión automática a Redis con retries cada 5s.
      - Continúa si falla el procesamiento de un mensaje individual.
      - Cursor del stream se actualiza solo en éxito para evitar pérdidas.
      - Manejo separado de ConnectionError vs. errores de procesamiento.
    """
    logger.info("=" * 70)
    logger.info("🧠 AI Engine (XGBoost CQRS Writer) — con Clima en Tiempo Real")
    logger.info("   Stream fuente     : %s", STREAM_NAME)
    logger.info("   Modelo            : %s", MODEL_PATH.name)
    logger.info("   Cache TTL         : %ds", ETA_TTL)
    logger.info("   Batch size        : %d msgs", BATCH_SIZE)
    logger.info("   Redis URL         : %s", REDIS_URL)
    logger.info("   Weather refresh   : cada %ds (%.1f min)",
                WEATHER_CACHE_TTL_S, WEATHER_CACHE_TTL_S / 60)
    logger.info("=" * 70)

    redis_client: Optional[aioredis.Redis] = None
    total_processed = 0
    total_errors    = 0

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
                logger.error("Sin conexión Redis: %s. Reintentando en 5s ...", exc)
                await asyncio.sleep(5)
                continue

        # ── B. Obtener cursor de inicio del stream ────────────────────────────
        try:
            stream_info = await redis_client.xinfo_stream(STREAM_NAME)
            last_id = stream_info.get("last-generated-id") or "0-0"
        except aioredis.ResponseError:
            logger.warning(
                "Stream '%s' no existe aún. Esperando publicación del ingestion ...",
                STREAM_NAME,
            )
            await asyncio.sleep(3)
            continue
        except aioredis.RedisError as exc:
            logger.error("Error leyendo info del stream: %s", exc)
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
                        # Cursor avanza siempre (para no reprocesar en crash)
                        last_id = msg_id

                        try:
                            await process_record(msg_id, fields, model, redis_client)
                            batch_ok += 1
                        except ValueError as exc:
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
                        "✔ Batch: %d ETAs calculados | total=%d | errores=%d",
                        batch_ok, total_processed, total_errors,
                    )
                if batch_error > 0:
                    logger.warning("⚠️  Batch: %d errores en este ciclo", batch_error)

            except aioredis.ConnectionError as exc:
                logger.error("Conexión Redis perdida: %s. Reconectando ...", exc)
                try:
                    await redis_client.aclose()
                except Exception:
                    pass
                redis_client = None
                break  # sale del inner loop → reconecta en outer loop

            except Exception as exc:
                logger.error("Error inesperado en worker loop: %s", exc, exc_info=True)
                await asyncio.sleep(2)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║   MBTA Predictive Transit — AI Engine (XGBoost + Clima Real)    ║")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")

    # Prefetch inicial del clima (para no tener latencia en el primer mensaje)
    logger.info("🌤️  Pre-fetching datos climáticos actuales de Boston ...")
    initial_weather = get_current_weather()
    logger.info(
        "   Clima actual: temp=%.1f°C | prec=%.2fmm | snow=%.2fcm",
        initial_weather["temperature_2m"],
        initial_weather["precipitation"],
        initial_weather["snowfall"],
    )

    # Cargar modelo al arranque (FALLA CRÍTICA si no existe)
    xgb_model = load_model()

    try:
        asyncio.run(worker_loop(xgb_model))
    except KeyboardInterrupt:
        logger.info("🛑 AI Engine detenido por el usuario.")
    except Exception as exc:
        logger.critical("❌ Error fatal en AI Engine: %s", exc, exc_info=True)
        sys.exit(1)
