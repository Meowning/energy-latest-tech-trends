# ===== Stage 1: Build dependencies =====
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y \
    git gcc g++ \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./worker/nlp/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===== Stage 2: Runtime =====
FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y \
    git \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY ./common ./common
COPY ./worker/nlp ./worker/nlp

ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "worker.nlp.tasks", "worker", "--loglevel=info"]
