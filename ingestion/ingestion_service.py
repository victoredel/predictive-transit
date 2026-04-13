"""
ingestion_service.py — Ingesta en Tiempo Real Adaptable (Strategy Pattern)
================================================================================
Microservicio de ingesta basado en el patrón Adapter (Strategy).
Permite enchufar distintas fuentes de datos de tránsito de forma agnóstica.

Actualmente implementa el feed público GTFS-Realtime de MBTA (Boston).

Flujo interno:
  1. Instanciamos un adaptador (MbtaAdapter).
  2. En cada intervalo (POLL_INTERVAL), el adaptador descarga y parsea el feed.
  3. Devuelve los datos en un esquema común: list[NormalizedPosition].
  4. Publicamos al Redis Stream 'bus_gps_stream'.
"""

import abc
import asyncio
import logging
import os
import time
from typing import Optional, TypedDict

import aiohttp
import redis.asyncio as aioredis
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("ingestion_service")

# ---------------------------------------------------------------------------
# Configuración / Variables de entorno
# ---------------------------------------------------------------------------
load_dotenv()

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
POLL_INTERVAL: int = int(os.getenv("POLL_INTERVAL", "15"))

STREAM_NAME: str = "bus_gps_stream"
STREAM_MAXLEN: int = 10_000


# ---------------------------------------------------------------------------
# Estructura Base (El Estándar Interno)
# ---------------------------------------------------------------------------

class NormalizedPosition(TypedDict):
    """
    Esquema estandarizado que el sistema y Redis siempre esperarán,
    independientemente de qué ciudad origen provea la información.
    (NOTA: Todos los valores deben publicarse como strings en Redis Streams)
    """
    vehicle_id: str
    route_id: str
    latitude: str
    longitude: str
    timestamp: str
    current_status: str


class BaseTransitAdapter(abc.ABC):
    """
    Clase Abstracta Base (Adapter/Strategy).
    Provee la interfaz común que deben respetar todas las fuentes de datos.
    """

    @abc.abstractmethod
    async def fetch_and_normalize(
        self, session: aiohttp.ClientSession
    ) -> list[NormalizedPosition]:
        """
        Descarga la data nativa de la API/Feed y la decodifica/normaliza.
        Retorna la lista de posiciones normalizadas lista para inyectarse a Redis.
        """
        pass


# ---------------------------------------------------------------------------
# El Adaptador Concreto (Boston MBTA)
# ---------------------------------------------------------------------------

class MbtaAdapter(BaseTransitAdapter):
    """
    Adaptador GTFS-Realtime público de MBTA (Boston).
    No requiere tokens ni cookies. Devuelve formato Protobuf binario.
    """

    FEED_URL = "https://cdn.mbta.com/realtime/VehiclePositions.pb"

    async def fetch_and_normalize(
        self, session: aiohttp.ClientSession
    ) -> list[NormalizedPosition]:
        try:
            async with session.get(
                self.FEED_URL,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "MBTA Feed respondió HTTP %d. Ignorando este ciclo.",
                        resp.status
                    )
                    return []
                raw_bytes = await resp.read()

        except aiohttp.ClientError as exc:
            logger.error("Error de red conectando al feed MBTA: %s", exc)
            return []
        except asyncio.TimeoutError:
            logger.error("Timeout leyendo feed MBTA (>10s)")
            return []

        feed = gtfs_realtime_pb2.FeedMessage()
        try:
            feed.ParseFromString(raw_bytes)
        except Exception as exc:
            logger.error("Error crítico decodificando Protobuf de MBTA: %s", exc)
            return []

        normalized_data: list[NormalizedPosition] = []
        server_time = int(time.time())

        for entity in feed.entity:
            if not entity.HasField("vehicle"):
                continue

            vp = entity.vehicle
            pos = vp.position

            status_str = "IN_TRANSIT_TO"
            if vp.HasField("current_status"):
                if vp.current_status == gtfs_realtime_pb2.VehiclePosition.STOPPED_AT:
                    status_str = "STOPPED_AT"
                elif vp.current_status == gtfs_realtime_pb2.VehiclePosition.INCOMING_AT:
                    status_str = "INCOMING_AT"

            record: NormalizedPosition = {
                "vehicle_id": str(vp.vehicle.id or entity.id),
                "route_id": str(vp.trip.route_id),
                "latitude": str(pos.latitude),
                "longitude": str(pos.longitude),
                "timestamp": str(int(vp.timestamp) if vp.timestamp else server_time),
                "current_status": status_str,
            }
            normalized_data.append(record)

        return normalized_data


# ---------------------------------------------------------------------------
# Capa de Publicación (Agnóstica a la Fuente)
# ---------------------------------------------------------------------------

async def publish_to_stream(
    redis_client: aioredis.Redis,
    positions: list[NormalizedPosition],
) -> int:
    published = 0
    for record in positions:
        try:
            await redis_client.xadd(
                STREAM_NAME,
                record,
                maxlen=STREAM_MAXLEN,
                approximate=True,
            )
            published += 1
        except aioredis.RedisError as exc:
            logger.error(
                "Error en Redis XADD (vehicle_id=%s): %s",
                record.get("vehicle_id", "?"),
                exc,
            )
    return published


# ---------------------------------------------------------------------------
# El Ingestor Principal (Polling Loop)
# ---------------------------------------------------------------------------

async def polling_loop() -> None:
    logger.info("🚌 Real-Time Ingestion arrancando (Adapter Mode) ...")
    logger.info("   Redis URL    : %s", REDIS_URL)
    logger.info("   Poll interval: %ds", POLL_INTERVAL)

    adapter: BaseTransitAdapter = MbtaAdapter()
    logger.info("   Adaptador act: %s", adapter.__class__.__name__)

    async with aiohttp.ClientSession() as session:
        redis_client: Optional[aioredis.Redis] = None

        while True:
            cycle_start = asyncio.get_event_loop().time()

            if redis_client is None:
                try:
                    redis_client = aioredis.from_url(
                        REDIS_URL,
                        encoding="utf-8",
                        decode_responses=True,
                        socket_connect_timeout=5,
                        socket_timeout=5,
                    )
                    await redis_client.ping()
                    logger.info("✅ Conectado a Redis broker (%s)", STREAM_NAME)
                except aioredis.RedisError as exc:
                    logger.error("Sin conexión a Redis: %s", exc)
                    redis_client = None

            positions = await adapter.fetch_and_normalize(session)
            logger.info("[%s] Vehículos parseados: %d", adapter.__class__.__name__, len(positions))

            if positions and redis_client is not None:
                try:
                    published = await publish_to_stream(redis_client, positions)
                    logger.info("✔ Publicados %d/%d registros.", published, len(positions))
                except aioredis.ConnectionError as exc:
                    logger.error("Perdida conexión Redis durante XADD: %s", exc)
                    await redis_client.aclose()
                    redis_client = None

            elapsed = asyncio.get_event_loop().time() - cycle_start
            sleep_time = max(0.0, POLL_INTERVAL - elapsed)
            await asyncio.sleep(sleep_time)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        asyncio.run(polling_loop())
    except KeyboardInterrupt:
        logger.info("🛑 Ingestion Service detenido por el usuario.")
