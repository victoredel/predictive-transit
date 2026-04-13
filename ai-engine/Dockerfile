FROM python:3.11-slim

LABEL maintainer="predictive-transit"
LABEL description="AI Engine — XGBoost CQRS Writer (XREAD → predict → Redis cache)"

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ai_worker.py .

RUN useradd -m -u 1001 transit
USER transit

CMD ["python", "-u", "ai_worker.py"]
