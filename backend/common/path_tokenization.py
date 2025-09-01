# backend/common/path_tokenization.py

from __future__ import annotations
import secrets
from functools import lru_cache
from pathlib import Path
import redis
from .config import (
    FILE_ROOT,
    PATH_TOKEN_REDIS_URL,
    PATH_TOKEN_TTL_SEC,
    PATH_TOKEN_ONE_TIME,
)
from .hashing import is_under_file_root

"""
파일 경로 노출을 막기 위한 토큰화(인코딩, 디코딩) 유틸
- encode_path(path)  -> 토큰 생성(Redis에 실제 경로 저장, TTL 있음)
- decode_path(token) -> 토큰을 실제 경로로 복원
- Celery의 Redis와 별개로 작동
"""

_PREFIX = "pathtok:"

@lru_cache(maxsize=1)
def _redis() -> "redis.Redis":
    return redis.Redis.from_url(
        PATH_TOKEN_REDIS_URL,
        decode_responses=True,
        health_check_interval=30,
        socket_keepalive=True,
    )

def encode_path(path: str | Path,
                *,
                ttl_sec: int = PATH_TOKEN_TTL_SEC,
                one_time: bool = PATH_TOKEN_ONE_TIME) -> str:
    """
    파일의 실제 경로를 외부에 노출하지 않고 토큰화하여 작업
    - ttl_sec: 유효 시간 (초 단위)
    - one_time: 일회용 옵션
    """
    p = Path(path).resolve()
    if not is_under_file_root(p):
        raise ValueError("경로가 파일 루트 디렉토리 바깥임")
    if not p.exists():
        raise FileNotFoundError(str(p))

    # URL-safe, 충분한 엔트로피
    token = secrets.token_urlsafe(16)
    r = _redis()
    key = _PREFIX + token
    r.set(key, str(p), ex=ttl_sec)
    if one_time:
        r.set(key + ":once", "1", ex=ttl_sec)
    return token

def decode_path(token: str) -> Path:
    """
    토큰을 실제 경로로 복호화
    - 만료/무효 토큰이면 ValueError
    - one_time 토큰이면 복원 즉시 삭제
    """
    if not token or not isinstance(token, str):
        raise ValueError("사용할 수 없는 토큰입니다.")

    r = _redis()
    key = _PREFIX + token
    s = r.get(key)
    if s is None:
        raise ValueError("만료되거나 사용할 수 없는 토큰입니다.")

    if r.get(key + ":once") == "1":
        r.delete(key, key + ":once")

    p = Path(s).resolve()
    if not is_under_file_root(p):
        raise RuntimeError("복호화된 경로가 파일 루트 디렉토리 바깥임")
    return p

# 디버깅용
def remaining_ttl(token: str) -> int:
    r = _redis()
    return int(r.ttl(_PREFIX + token))

def is_valid(token: str) -> bool:
    r = _redis()
    return r.exists(_PREFIX + token) == 1
