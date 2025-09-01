# backend/common/hashing.py
import hashlib
import time
import os
from pathlib import Path
from .config import INCOMING_DIR, PROCESSED_DIR

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sha256_file(path: Path, buf: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(buf)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def wait_file_stable(path: Path, checks: int = 2, interval: float = 1.0) -> bool:
    """
    size/mtime이 연속 2회 동일하게 나오면 다운 완료된거
    """
    last = None
    stable = 0
    for _ in range(60):  # 최대 60초
        try:
            stat = path.stat()
        except FileNotFoundError:
            time.sleep(interval)
            continue
        sig = (stat.st_size, stat.st_mtime)
        if sig == last:
            stable += 1
            if stable >= checks:
                return True
        else:
            stable = 0
            last = sig
        time.sleep(interval)
    return False

def move_to_processed(src: Path) -> Path:
    dst = PROCESSED_DIR / src.name
    i = 1
    while dst.exists():
        dst = PROCESSED_DIR / f"{src.stem}_{i}{src.suffix}"
        i += 1
    src.replace(dst)
    return dst
