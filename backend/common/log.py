# backend/common/log.py
from __future__ import annotations
import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import (
    FILE_ROOT,
    LOG_LEVEL,
    LOG_TO_FILE,
    LOG_FILE_NAME,
    LOG_FILE_MAX_BYTES,
    LOG_FILE_BACKUPS,
)

"""
로깅 유틸
- LOG_TO_FILE=1 이면 /data/files/logs/app.log에 로그 기록
- LOG_JSON=1 이면 JSON 라인 포맷, 기본은 key=value 스타일
- contextvars 로 요청/태스크 컨텍스트를 로그에 자동 포함
- Celery 신호에 바인딩하면 task_id, task_name 자동 포함 가능
"""

# 로거 설정 플래그 (첫 호출 확인용)
_LOG_CONFIGURED = False

def setup_logging(level: str | None = None) -> None:
    """
    싱글톤처럼 동작하는 로거 세팅
    (프로세스 간 공유하지 않아서 싱글톤은 아님)
    """
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return

    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    lvl = getattr(logging, (level or LOG_LEVEL), logging.INFO)

    root = logging.getLogger()
    root.setLevel(lvl)

    # 콘솔 핸들러 (stdout)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
    root.addHandler(sh)

    # 파일 핸들러
    if LOG_TO_FILE:
        log_dir: Path = (FILE_ROOT / "logs").resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            filename=str(log_dir / LOG_FILE_NAME),
            maxBytes=LOG_FILE_MAX_BYTES,
            backupCount=LOG_FILE_BACKUPS,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(fh)

    _LOG_CONFIGURED = True

# 기록을 위해 다른 곳에서 호출하는 함수
def get_logger(name: str | None = None) -> logging.Logger:
    if not _LOG_CONFIGURED:
        setup_logging()
    return logging.getLogger(name if name else __name__)