import collections
import collections.abc
collections.Sequence = collections.abc.Sequence
import os
import threading
import tempfile
import numpy as np
import re
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Enum as SqlEnum
from sqlalchemy.orm import declarative_base, sessionmaker
import pytesseract
import cv2
import fitz
from PIL import Image
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import ocrmypdf

from enums import StatusEnum, SourceEnum

# 환경 변수 설정
DB_PATH     = os.getenv("DB_PATH", "./ocr.db")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
T5_MODEL    = os.getenv("T5_MODEL", "t5-small")

# 데이터베이스 초기화
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id                  = Column(Integer, primary_key=True, index=True)
    created_at          = Column(DateTime, default=datetime.utcnow)
    source              = Column(Text, nullable=False)
    ocr_text            = Column(Text, nullable=False)
    extractive_summary  = Column(Text, nullable=False)
    abstractive_summary = Column(Text, nullable=True)
    status              = Column(SqlEnum(StatusEnum), nullable=False, default=StatusEnum.PENDING)

Base.metadata.create_all(bind=engine)

# T5 모델 불러오기
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
model = T5ForConditionalGeneration.from_pretrained(T5_MODEL)
model.eval()

doc_semaphore = threading.Semaphore(MAX_WORKERS)
app = FastAPI()

# 텍스트 전처리
def preprocess_text(text: str) -> str:
    cleaned = re.sub(r"[^0-9가-힣\s\.,\?\!]", " ", text)
    return re.sub(r"\s+", " ", cleaned).strip()

# OCR: PDF 또는 이미지에서 페이지별 텍스트 추출
def perform_ocr_pages(file_bytes: bytes) -> list[str]:
    pages_text = []

    if file_bytes.startswith(b"%PDF"):
        # OCRmyPDF로 PDF에 텍스트 레이어 삽입
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf_in:
            tf_in.write(file_bytes)
            in_path = tf_in.name
        out_path = in_path.replace(".pdf", "_ocr.pdf")
        try:
            print("[3] OCRmyPDF 실행")
            ocrmypdf.ocr(
                in_path,
                out_path,
                language="kor",
                skip_text=True,                    # 기존 텍스트 레이어는 건너뛰기
                force_ocr=False,                   # 강제 OCR 비활성
                color_conversion_strategy="RGB",   # 비표준 색 공간은 RGB로 변환
                output_type="pdf",                 # PDF/A 변환 없이 원본 색상 유지
                deskew=True,                       # 기울기 보정
                remove_background=True,            # 배경 노이즈 제거
                jobs=4                             # CPU 코어 수에 맞춰 병렬 처리
            )
            
            # OCR 완료된 PDF 열어 텍스트 추출
            doc = fitz.open(out_path)
            pages_text = []
            for i in range(doc.page_count):
                pages_text.append(doc[i].get_text("text"))
            doc.close()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"OCRmyPDF 처리 실패: {e}")
        finally:
            for path in (in_path, out_path):
                try:
                    os.remove(path)
                except OSError:
                    pass
        return pages_text

    # 이미지 단일 페이지 처리
    arr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise HTTPException(status_code=400, detail="지원되지 않는 파일 형식")
    print("[3] OCR 처리 중: 이미지 파일")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return [pytesseract.image_to_string(rgb, lang="kor")]

# 추출적 요약: 순위 기반으로 추출된 순서대로 출력
def extractive_summary(pages_text: list[str], num_sentences: int = 3) -> str:
    combined = "\n".join(pages_text)
    cleaned = preprocess_text(combined)
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.summarizers.text_rank import TextRankSummarizer
    from sumy.nlp.tokenizers import Tokenizer

    parser = PlaintextParser.from_string(cleaned, Tokenizer("korean"))
    summarizer = TextRankSummarizer()
    ranked_sentences = summarizer(parser.document, num_sentences + 2)

    results = []
    for sent_obj in ranked_sentences:
        sentence = str(sent_obj)
        if any(r['sentence'] == sentence for r in results):
            continue
        for page_no, text in enumerate(pages_text, start=1):
            if sentence in text:
                results.append({'page': page_no, 'sentence': sentence})
                break
        if len(results) >= num_sentences:
            break

    summary_lines = [f"[Page {r['page']}] {r['sentence']}" for r in results]
    for idx, line in enumerate(summary_lines, start=1):
        print(f"[5] 요약 문장 {idx}: {line}")
    return "\n".join(summary_lines)

# 생성적 요약
def generate_summary(text: str) -> str:
    cleaned = preprocess_text(text)
    inputs = tokenizer.encode("summarize: " + cleaned, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_beams=2, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 생성적 요약 백그라운드 태스크
def background_generation(doc_id: int):
    doc_semaphore.acquire()
    try:
        db = SessionLocal()
        doc = db.get(Document, doc_id)
        if not doc:
            return
        doc.status = StatusEnum.PARTIAL
        db.commit()
        doc.abstractive_summary = generate_summary(doc.ocr_text)
        doc.status = StatusEnum.DONE
        db.commit()
    finally:
        doc_semaphore.release()
        db.close()

# 메인 엔드포인트: /process
@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    source: SourceEnum = Form(...),
    file: UploadFile = File(...),
    extractive_sentences: int = Form(3),
    do_generation: bool = Form(False)
):
    print(f"[1] 요청 수신: source={source.value}")
    content = await file.read()
    print(f"[2] 파일 수신 완료 (크기: {len(content)} bytes)")

    pages = perform_ocr_pages(content)
    with open("example.txt", "w") as file:
        file.write("\n".join(pages))
    print(f"[6] 전체 페이지 OCR 완료: {len(pages)} 페이지 처리됨")

    extractive = extractive_summary(pages, extractive_sentences)
    print("[7] 추출적 요약 완료")

    db = SessionLocal()
    doc = Document(
        source=source.value,
        ocr_text="\n".join(pages),
        extractive_summary=extractive,
        status=StatusEnum.PENDING
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    print(f"[8] DB 저장 완료: doc_id={doc.id}")

    if do_generation:
        print(f"[9] 생성적 요약 태스크 예약: doc_id={doc.id}")
        background_tasks.add_task(background_generation, doc.id)

    db.close()
    print(f"[10] 응답 전송: doc_id={doc.id}")
    return {
        "doc_id": doc.id,
        "ocr_text": "\n".join(pages),
        "extractive_summary": extractive,
        "generation_started": do_generation
    }

# 상태 조회 엔드포인트
@app.get("/status/{doc_id}")
async def get_status(doc_id: int):
    db = SessionLocal()
    doc = db.get(Document, doc_id)
    db.close()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "doc_id": doc.id,
        "source": doc.source,
        "ocr_text": doc.ocr_text,
        "extractive_summary": doc.extractive_summary,
        "abstractive_summary": doc.abstractive_summary,
        "status": doc.status.value
    }

# 헬스 체크 엔드포인트
@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
