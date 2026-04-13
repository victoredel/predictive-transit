"""
main.py — Routing API (Lector CQRS) con FastAPI
================================================
Patrón: CQRS Reader — Solo lee del cache Redis. NUNCA recalcula nada.

El Motor de IA (ai_worker.py) pre-calcula y escribe ETAs en Redis cada vez
que llegan nuevas posiciones del stream. Esta API simplemente sirve esos
resultados con latencia O(1) (un único HGET a Redis).

Endpoints:
  GET /health               → Health check para Docker/load balancer
  GET /api/eta/{route}/{v}  → ETA pre-calculado del vehículo

Diseño:
  - FastAPI asíncrono (no bloques síncronos)
  - Conexión Redis compartida por el ciclo de vida de la app (lifespan)
  - Respuesta 404 descriptiva si el AI Engine aún no procesó ese vehículo
  - Respuesta 503 si Redis no está disponible
"""

import logging
import os
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("routing_api")

load_dotenv()
REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")


# ---------------------------------------------------------------------------
# Ciclo de vida de la aplicación — Conexión Redis compartida
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestiona la conexión Redis como recurso del ciclo de vida de la app.
    Se crea UNA sola vez al arrancar y se cierra al apagar el servidor.
    """
    logger.info("🚀 Routing API arrancando — Conectando a Redis...")
    app.state.redis = aioredis.from_url(
        REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        socket_connect_timeout=5,
        socket_timeout=5,
        max_connections=20,     # pool de conexiones para múltiples workers
    )
    try:
        await app.state.redis.ping()
        logger.info("✅ Redis conectado: %s", REDIS_URL)
    except aioredis.RedisError as exc:
        # Arrancamos de todas formas; el endpoint devolverá 503 hasta que Redis esté listo
        logger.error("⚠️ Redis no disponible al arrancar: %s", exc)

    yield   # ← La aplicación vive aquí

    logger.info("Cerrando conexión Redis...")
    await app.state.redis.aclose()


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Predictive Transit — Routing API",
    description=(
        "API de consulta de ETAs pre-calculados por el Motor de IA XGBoost. "
        "Implementa el rol de **Lector CQRS**: complejidad O(1) por operación. "
        "Nunca realiza inferencia, siempre lee del cache Redis."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Infraestructura"])
async def health_check(request: Request):
    """
    Health check para Docker HEALTHCHECK y load balancers.
    Verifica que tanto el servidor como Redis estén operativos.
    """
    try:
        await request.app.state.redis.ping()
        redis_status = "ok"
    except aioredis.RedisError:
        redis_status = "unavailable"

    return {
        "status": "ok",
        "redis": redis_status,
        "service": "routing-api",
        "pattern": "CQRS Reader",
    }


@app.get(
    "/api/eta/{route_id}/{vehicle_id}",
    summary="Consultar ETA de un vehículo",
    description=(
        "Lee el ETA pre-calculado por el Motor de IA desde el cache Redis. "
        "**Complejidad O(1)** — un único HGETALL sobre la clave `eta:{route_id}:{vehicle_id}`. "
        "Si el cache no existe, el AI Engine aún no procesó ese vehículo."
    ),
    tags=["ETA"],
    responses={
        200: {"description": "ETA disponible en cache"},
        404: {"description": "ETA no calculado aún por el Motor de IA"},
        503: {"description": "Redis no disponible"},
    },
)
async def get_eta(route_id: str, vehicle_id: str, request: Request):
    """
    **Lector CQRS puro:**
    - Construye la clave `eta:{route_id}:{vehicle_id}`
    - Hace HGETALL → responde en JSON
    - Si la clave no existe → 404 descriptivo
    - Si Redis falla → 503

    Parámetros:
    - **route_id**: ID de la línea de tránsito (ej: `Red`, `1`, `Green-B`)
    - **vehicle_id**: ID del vehículo (ej: `R-547E13EF`, `y1234`)
    """
    cache_key = f"eta:{route_id}:{vehicle_id}"

    # ── Leer del cache Redis (O(1)) ───────────────────────────────────────
    try:
        data: dict = await request.app.state.redis.hgetall(cache_key)
    except aioredis.RedisError as exc:
        logger.error("Error de Redis leyendo clave '%s': %s", cache_key, exc)
        raise HTTPException(
            status_code=503,
            detail={
                "error": "redis_unavailable",
                "message": "El servicio de cache no está disponible temporalmente.",
            },
        )

    # ── Cache miss → el AI Engine aún no procesó este vehículo ───────────
    if not data:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "eta_not_found",
                "message": "ETA no disponible aún. El Motor de IA está procesando.",
                "route_id": route_id,
                "vehicle_id": vehicle_id,
                "cache_key": cache_key,
                "hint": "Reintenta en unos segundos. El sistema actualiza ETAs cada ~15s.",
            },
        )

    # ── Cache hit → convertir tipos y devolver ────────────────────────────
    logger.debug("Cache HIT: %s", cache_key)

    return JSONResponse(
        content={
            "vehicle_id":           data.get("vehicle_id"),
            "route_id":             data.get("route_id"),
            "eta_segundos":         _safe_float(data.get("eta_segundos")),
            "nivel_confianza":      _safe_float(data.get("nivel_confianza")),
            "current_status":       data.get("current_status"),
            "distancia_a_parada_m": _safe_float(data.get("distancia_a_parada_m")),
            "velocidad_actual_kmh": _safe_float(data.get("velocidad_actual_kmh")),
            "ultima_actualizacion": data.get("ultima_actualizacion"),
            "_meta": {
                "cache_key":  cache_key,
                "ttl_s":      300,
                "pattern":    "CQRS Reader — O(1) Redis HGETALL",
                "model":      "XGBoost (ai-engine)",
            },
        }
    )


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _safe_float(value: str | None, default: float = 0.0) -> float:
    """Convierte string a float de forma segura, devuelve default si falla."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default
