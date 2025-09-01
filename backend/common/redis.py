# backend/common/redis.py
"""
singleton 패턴 Redis 세팅
"""

from __future__ import annotations
import redis
from .config import REDIS_URL

_redis: "redis.Redis | None" = None

def get_redis() -> "redis.Redis":
    global _redis
    if _redis is None:
        _redis = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _redis

def ping() -> bool:
    """Redis 연결 상태 확인용"""
    try:
        return bool(get_redis().ping())
    except Exception:
        return False
