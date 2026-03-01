# 1. Create base.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONBUFFERED=1

WORKDIR /app

# --no-install-recommends to reduce image size by ~60%
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m ensurepip --upgrade
RUN pip install --root-user-action=ignore --no-cache-dir --default-timeout=100 -r requirements.txt

COPY customer_churn_dataset.csv .
COPY train_model.py .

RUN python train_model.py

# 2. Final image.
FROM python:3.12-slim

WORKDIR /app

COPY --from=base /usr/local /usr/local
COPY --from=base /app/models/ ./models/
COPY ./src /app/src
COPY ./config /app/config
COPY ./tests /app/tests

ARG APP_VERSION=v1.0.0
ENV VERSION=$APP_VERSION

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/

# Exposing the FastAPI port
EXPOSE 8000

# Non-root user for security
RUN useradd -m dev
USER dev


HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
