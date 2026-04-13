FROM python:3.11-slim

LABEL maintainer="predictive-transit"
LABEL description="Routing API — FastAPI CQRS Reader (Redis GET → JSON)"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

RUN useradd -m -u 1001 transit
USER transit

# uvicorn escucha en 0.0.0.0 para que Docker enrute el tráfico correctamente
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
