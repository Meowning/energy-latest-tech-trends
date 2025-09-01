# -*- coding: utf-8 -*-
# backend/common/config.py
import os
from pathlib import Path

# ==================== 파일 경로 루트 ====================
FILE_ROOT = Path(os.getenv("FILE_ROOT", "/data/files")).resolve()
INCOMING_DIR = FILE_ROOT / "incoming"
PROCESSED_DIR = FILE_ROOT / "processed"
FAILED_DIR = FILE_ROOT / "failed"

for p in (FILE_ROOT, INCOMING_DIR, PROCESSED_DIR, FAILED_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ==================== DB (MySQL/MariaDB) ====================
MYSQL_HOST = os.getenv("MYSQL_HOST", "db")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_DB   = os.getenv("MYSQL_DB", "energy")
MYSQL_USER = os.getenv("MYSQL_USER", "energy")
MYSQL_PW   = os.getenv("MYSQL_PW", "energy")
MYSQL_URL = os.getenv(
    "MYSQL_URL",
    f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PW}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
)

# ==================== Redis  ====================
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")

# (선택) 경로 토큰 전용 Redis URL 분리
# 기본적으로 REDIS는 샐러리와 공용으로 이용함
PATH_TOKEN_REDIS_URL = os.getenv("PATH_TOKEN_REDIS_URL", REDIS_URL)
# 토큰 정책 (TTL, 원타임)
PATH_TOKEN_TTL_SEC = int(os.getenv("PATH_TOKEN_TTL_SEC", "600"))  # 10분
PATH_TOKEN_ONE_TIME = os.getenv("PATH_TOKEN_ONE_TIME", "1").lower() in ("1", "true", "yes")

# ==================== 저메모리 세팅 ====================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ==================== 요약 모델 ====================
T5_REMOTE_ID = os.getenv("T5_REMOTE_ID", "eenzeenee/t5-small-korean-summarization")
SBERT_REMOTE_ID = os.getenv("SBERT_REMOTE_ID", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
T5_QUANTIZE = os.getenv("T5_QUANTIZE", "1")  # CPU dynamic quantize

# ==================== Celery 워커 타임리밋 ====================
CELERY_TIME_LIMIT = int(os.getenv("CELERY_TIME_LIMIT", "900"))          # hard 15m
CELERY_SOFT_TIME_LIMIT = int(os.getenv("CELERY_SOFT_TIME_LIMIT", "840"))  # soft 14m

# ==================== 워처 스케줄 ====================
CRAWL_HOURS = list(range(9, 18))[:-1]  # 9..17 정각
CRAWL_WEEKDAYS = {0, 1, 2, 3, 4}       # 월~금
KST_TZ = os.getenv("TZ", "Asia/Seoul")

# ==================== Logging ====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()                 # INFO|DEBUG|WARNING|ERROR
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "0").lower() in ("1","true","yes")
LOG_FILE_NAME = os.getenv("LOG_FILE_NAME", "app.log")
LOG_FILE_MAX_BYTES = int(os.getenv("LOG_FILE_MAX_BYTES", "10485760"))  # 10MB
LOG_FILE_BACKUPS = int(os.getenv("LOG_FILE_BACKUPS", "5"))

# ==================== Mail (SMTP) ====================
MAIL_ENABLED = os.getenv("MAIL_ENABLED", "0").lower() in ("1", "true", "yes")

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))  # STARTTLS 기본 포트
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_TIMEOUT = int(os.getenv("SMTP_TIMEOUT", "15"))

# 연결 방식: 둘 다 1이면 SSL 우선
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "0").lower() in ("1", "true", "yes")
SMTP_USE_STARTTLS = os.getenv("SMTP_USE_STARTTLS", "1").lower() in ("1", "true", "yes")

MAIL_FROM = os.getenv("MAIL_FROM", "noreply@example.com")
MAIL_FROM_NAME = os.getenv("MAIL_FROM_NAME", "Energy Digest")
MAIL_SUBJECT_PREFIX = os.getenv("MAIL_SUBJECT_PREFIX", "[Energy]")
