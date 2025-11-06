# ======================================================
# Base Image
# ======================================================
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1

# 시스템 기본 패키지 (ocr 등 확장 가능)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 공통 requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common /app/common

# ======================================================
# API Stage
# ======================================================
FROM base AS api
COPY api /app/api
RUN pip install --no-cache-dir -r /app/api/requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ======================================================
# OCR Worker Stage
# ======================================================
FROM base AS ocr
COPY worker/ocr /app/worker/ocr
RUN pip install --no-cache-dir -r /app/worker/ocr/requirements.txt
CMD ["python", "worker/ocr/tasks.py"]

# ======================================================
# NLP Worker Stage
# ======================================================
FROM base AS nlp
COPY worker/nlp /app/worker/nlp
RUN pip install --no-cache-dir -r /app/worker/nlp/requirements.txt
CMD ["python", "worker/nlp/tasks.py"]

# ======================================================
# Crawler Stage
# ======================================================
FROM base AS crawler
COPY crawler /app/crawler
RUN pip install --no-cache-dir -r /app/crawler/requirements.txt
CMD ["python", "crawler/main.py"]

# ======================================================
# Watcher Stage
# ======================================================
FROM base AS watcher
COPY watcher /app/watcher
RUN pip install --no-cache-dir -r /app/watcher/requirements.txt
CMD ["python", "watcher/run.py"]

# ======================================================
# Final Stage (ENTRYPOINT 선택형)
# ======================================================
FROM base AS final
ARG ROLE=api
COPY --from=${ROLE} /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=${ROLE} /app /app
WORKDIR /app

CMD ["bash", "-c", "\
if [ \"$ROLE\" = 'api' ]; then \
    uvicorn api.main:app --host 0.0.0.0 --port 8000; \
elif [ \"$ROLE\" = 'ocr' ]; then \
    python worker/ocr/tasks.py; \
elif [ \"$ROLE\" = 'nlp' ]; then \
    python worker/nlp/tasks.py; \
elif [ \"$ROLE\" = 'crawler' ]; then \
    python crawler/main.py; \
elif [ \"$ROLE\" = 'watcher' ]; then \
    python watcher/run.py; \
else \
    echo 'Unknown ROLE: '$ROLE; exit 1; \
fi"]
