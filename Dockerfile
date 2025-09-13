FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (ensure builds work if wheels unavailable)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Data directory for SQLite persistence
RUN mkdir -p /data

# Default DB path inside container can be overridden via env
ENV REVIEW_DB_PATH=/data/reviews.db

# Drop privileges
RUN useradd -m appuser && chown -R appuser:appuser /app /data
USER appuser

CMD ["python", "-u", "main.py"]

