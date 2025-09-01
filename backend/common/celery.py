# backend/common/celery_app.py
from celery import Celery
from kombu import Queue
from .config import REDIS_URL, CELERY_TIME_LIMIT, CELERY_SOFT_TIME_LIMIT, KST_TZ

# Celery는 비동기 작업 큐/작업 큐를 위한 분산 시스템
# 주로 백그라운드에서 작업을 처리하는 데 사용

# Celery 앱 생성
celery_app = Celery(
    "energy_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "worker.ocr.tasks",
        "worker.nlp.tasks",
    ],
)

# 설정 업데이트
celery_app.conf.update(
    timezone=KST_TZ,
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_time_limit=CELERY_TIME_LIMIT,
    task_soft_time_limit=CELERY_SOFT_TIME_LIMIT,
    task_reject_on_worker_lost=True,           # 워커 크래시 뜨면 재 큐잉
    task_acks_on_failure_or_timeout=True,      # 실패 또는 타임아웃 시 작업 확인
    task_routes={
        "worker.ocr.tasks.*": {"queue": "ocr"},
        "worker.nlp.tasks.*": {"queue": "nlp"},
    },
    task_queues=(
        Queue("ocr"),
        Queue("nlp"),
    ),
)

# 설명 : OCR 결과 -> NLP 입력
def ocr_to_nlp_chain(source: str, file_path: str, abstractive_lines: int = 3):
    from celery import chain
    from worker.ocr.tasks import ocr_extract
    from worker.nlp.tasks import nlp_summarize
    return chain(
        ocr_extract.s(source=source, file_path=file_path),
        nlp_summarize.s(abstractive_lines=abstractive_lines),
    )