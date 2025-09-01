# -*- coding: utf-8 -*-
# backend/worker/nlp/tasks.py
from __future__ import annotations

from celery import shared_task

from common.nlp import (
    split_korean_sentences,
    generate_summary,
)

@shared_task(name="worker.nlp.tasks.nlp_summarize")
def nlp_summarize(prev: dict, abstractive_lines: int = 3) -> dict:
    """
    입력:
      ocr과 동일한 값 {source, ocr_text}
    출력:
      {"source": ..., "ocr_text": ..., "summary": ...}

    - CPU-only 환경 가정. 모델 로딩은 common.nlp 내부에서 lazy-init.
    - 실패 시 RuntimeError
    """
    try:
        if not isinstance(prev, dict):
            raise ValueError("nlp_summarize 함수는 dict 타입의 입력을 요구함")
        text = prev.get("ocr_text")
        if not text:
            raise ValueError("prev['ocr_text'] 값이 없음")

        # 문장 분리
        sents = split_korean_sentences(text)

        # 생성 요약만 수행
        abstractive_summary = generate_summary(sents, lines=abstractive_lines)

        prev.update({
            "summary": abstractive_summary,
        })
        return prev

    except (ValueError,) as e:
        # nlp 로직 내 입력 오류이므로 RuntimeError로 분류
        raise RuntimeError(f"NLP 입력 오류: {e}") from e
    except Exception as e:
        raise RuntimeError(f"NLP 실패: {e}") from e
