FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc g++ \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./watcher/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./common ./common
COPY ./watcher ./watcher

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "watcher.run"]
