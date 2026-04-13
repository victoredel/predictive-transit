FROM python:3.11-slim

# Metadatos
LABEL maintainer="predictive-transit"
LABEL description="Ingestion Service — Olho Vivo São Paulo → Redis Stream"

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos dependencias primero (mejor caché de Docker layers)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el código del servicio
COPY ingestion_service.py .

# Usuario no-root por buenas prácticas de seguridad
RUN useradd -m -u 1001 transit
USER transit

# El script es el entrypoint directo
CMD ["python", "-u", "ingestion_service.py"]
