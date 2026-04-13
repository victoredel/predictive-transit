FROM python:3.11-slim

LABEL maintainer="predictive-transit"
LABEL description="Ingestion Service — MBTA GTFS-RT → Redis Stream"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingestion_service.py .

RUN useradd -m -u 1001 transit
USER transit

CMD ["python", "-u", "ingestion_service.py"]
