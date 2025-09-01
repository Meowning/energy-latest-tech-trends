# -*- coding: utf-8 -*-
# backend/worker/ocr/tasks.py
from __future__ import annotations
from pathlib import Path

from celery import shared_task
from common.ocr import perform_ocr_pages

@shared_task(name="worker.ocr.tasks.ocr_extract")
def ocr_extract(source: str, file_path: str) -> dict:
    """
    입력:
      - source: 기관명
      - file_path: PDF 파일 경로

    출력:
      { "source": ..., "ocr_text": ... }

    실패 시 ValueError or RuntimeError
    """
    try:
        if not file_path:
            raise ValueError("파일 경로가 비어있음")
        p = Path(file_path)
        if not p.exists():
            raise ValueError(f"파일을 찾을 수 없음: {p}")

        data = p.read_bytes()
        ocr_text = perform_ocr_pages(data, source)
        return {"source": source, "ocr_text": ocr_text}

    except (ValueError, FileNotFoundError) as e:
        raise RuntimeError(f"OCR 입력 오류: {e}") from e
    except Exception as e:
        raise RuntimeError(f"OCR 실패: {e}") from e
