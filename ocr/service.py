# -*- coding: utf-8 -*-
import os
import threading
import tempfile
import re
import numpy as np
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Enum as SqlEnum
from sqlalchemy.orm import declarative_base, sessionmaker
import pytesseract
import cv2
import fitz
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import ocrmypdf
from enums import StatusEnum, SourceEnum
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
import kss
from pykospacing import Spacing
from hanspell import spell_checker

# — 전역 초기화 —————————————————————————————————————————————
spacing = Spacing()                                                      # O(1)
okt     = Okt()                                                          # O(1)
sbert   = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2") # O(1)

# T5 모델 로드                                                        
T5_MODEL  = os.getenv("T5_MODEL", "t5-small")                          # O(1)
tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)                       # O(1)
model     = T5ForConditionalGeneration.from_pretrained(T5_MODEL)         # O(1)
model.eval()                                                             # O(1)

# DB 설정                                                               
DB_PATH      = os.getenv("DB_PATH", "./ocr.db")                       # O(1)
MAX_WORKERS  = int(os.getenv("MAX_WORKERS", "1"))                     # O(1)
engine       = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False}) # O(1)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)                      # O(1)
Base         = declarative_base()                                         # O(1)

doc_semaphore = threading.Semaphore(MAX_WORKERS)                          # O(1)
app           = FastAPI()                                                 # O(1)

class Document(Base):
    __tablename__ = "documents"
    id                  = Column(Integer, primary_key=True, index=True)
    created_at          = Column(DateTime, default=datetime.utcnow)
    source              = Column(Text, nullable=False)
    ocr_text            = Column(Text, nullable=False)
    extractive_summary  = Column(Text, nullable=False)
    abstractive_summary = Column(Text, nullable=True)
    status              = Column(SqlEnum(StatusEnum), nullable=False, default=StatusEnum.PENDING)

Base.metadata.create_all(bind=engine)                                    # O(1)

STOP_WORDS = {"하","되","하다","되다","되어다","있","있다","없","없다","않"}  # O(1)

# 노이즈 라인 필터링: O(L) where L = len(line)
def is_noise_line(line: str) -> bool:
    text = line.strip()
    if len(text) < 10 or re.fullmatch(r"\d+(?:\s*\d+)*", text):
        return True
    if re.search(r"\b(vol\.?|TEL|FAX|E[- ]?mail|202\d|July|월|호기)\b", text, re.IGNORECASE):
        return True
    tokens = okt.pos(text, norm=True, stem=True)
    return not any(pos == 'Verb' for _, pos in tokens)

# 문장 분리: O(N + S) where N = total chars, S = number of sentences
def split_korean_sentences(text: str) -> list[str]:
    # 1) KSS로 문장 분리
    try:
        raws = kss.split_sentences(
            text, backend='mecab',
            use_quotes_brackets_processing=False,
            ignore_quotes_or_brackets=True
        )
    except Exception:
        raws = re.split(r'(?<=[\.!\?])\s*', text)

    results = []
    for raw in raws:
        sent = raw.strip()
        if not sent or is_noise_line(sent):
            continue

        # 2) 한 문장씩 맞춤법·띄어쓰기 교정
        try:
            result = spell_checker.check(sent)
            sent = result.checked
        except Exception as e:
            print("맞춤법 검사 실패:", e)
            # 검사 실패 시 원본 문장 그대로 사용
            pass

        results.append(sent)

    return results

# OCR 및 공백 복원: O(P + R) where P = pages, R = restored text length
def perform_ocr_pages(file_bytes: bytes) -> str:
    pages = []
    if file_bytes.startswith(b"%PDF"):
        # PDF OCR
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(file_bytes)
            in_path = tf.name
        out_path = in_path.replace(".pdf", "_ocr.pdf")
        try:
            print("[3] OCRmyPDF 실행")
            ocrmypdf.ocr(
                in_path, out_path,
                language="kor", skip_text=True, force_ocr=False,
                color_conversion_strategy="RGB", output_type="pdf",
                deskew=True, remove_background=True, jobs=4
            )
            doc = fitz.open(out_path)
            for i in range(doc.page_count):
                #print(f"[4] 페이지 {i+1}/{doc.page_count} OCR 처리 중")
                pages.append(doc[i].get_text("text"))
            doc.close()
        finally:
            for p in (in_path, out_path):
                try: os.remove(p)
                except: pass
    else:
        # 이미지 OCR
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise HTTPException(400, "지원되지 않는 파일 형식")
        print("[3] OCR 처리 중: 이미지 파일")
        pages.append(
            pytesseract.image_to_string(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lang="kor"
            )
        )

    # 1) 병합
    combined = "".join(pages)


    # 2) 모든 공백(스페이스·탭·줄바꿈) 제거
    combined = re.sub(r'\s+', '', combined)

    # 3) URL 제거 (공백 없이 붙어 있어도, URL 문자셋[A–Z a–z 0–9 및 URL 특수문자] 전까지만)
    combined = re.sub(
        r'(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+',
        '',
        combined
    )

    # 4) 이메일 제거 (변경 없음)
    combined = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', combined)

    combined = re.sub(r'(?<!\d)(?:\d{4}년{0,1}\d{1,2}월\d{1,2}일)(?!\d)', '', combined)

    # 5) 전화번호 및 의미 없이 긴 숫자 제거
    combined = re.sub(
        r'(?<!\d)(?:\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})(?!\d)',
        '',
        combined
    )
    #    - 구분자 없는 9~11자리 연속 숫자 (예: 01012345678)
    combined = re.sub(r'\b\d{9,11}\b', '', combined)
    #    - 의미 없이 긴 숫자열 (12자리 이상) 전부 제거
    combined = re.sub(r'\d{12,}', '', combined)

    # 6) 특수문자 필터: 한글·영문·숫자·.,!? 만 남기고 전부 제거
    combined = re.sub(r'[^0-9A-Za-z가-힣\.\,\!\?]', '', combined)

    # 7) before_spacing.txt 기록
    with open("./processed/before_spacing.txt", "w", encoding="utf-8") as f:
        f.write(combined)    

    # 8) 띄어쓰기 복원
    print("[공백 복원 시작]")
    spaced = spacing(combined)
    print("[공백 복원 완료]")

    # 9) after_spacing.txt 기록
    with open("./processed/after_spacing.txt", "w", encoding="utf-8") as f:
        f.write(spaced)

    return spaced


# 추출적 요약: embeddings O(S·d) + clustering O(S²)
def extractive_summary(text: str, num_sentences: int = 3) -> str:
    sents = split_korean_sentences(text)
    if not sents:
        return ""
    proc = [preprocess_text(s) for s in sents]
    embs = sbert.encode(proc, convert_to_numpy=True, show_progress_bar=False)
    n = min(num_sentences, len(embs))
    clust = AgglomerativeClustering(
        n_clusters=n,
        metric="cosine",
        linkage="average"
    )
    labels = clust.fit_predict(embs)
    summary = []
    for lbl in set(labels):
        idxs = np.where(labels == lbl)[0]
        cent = embs[idxs].mean(axis=0)
        sims = cosine_similarity([cent], embs[idxs])[0]
        best = idxs[sims.argmax()]
        summary.append((best, sents[best]))
    summary.sort(key=lambda x: x[0])
    return "\n".join(f"[{i+1}] {s}" for i, s in summary)

# 텍스트 전처리: O(L)
def preprocess_text(text: str) -> str:
    txt = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    txt = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', txt)
    txt = re.sub(r"[\r\n\t]+", ' ', txt)
    txt = re.sub(r"[^0-9A-Za-z가-힣\s\.\?!]", ' ', txt)
    txt = re.sub(r"\s+", ' ', txt).strip()
    tokens = okt.pos(txt, norm=True, stem=True)
    return " ".join(
        lemma for lemma, pos in tokens
        if pos in {'Noun','Verb','Adjective','Alpha'}
        and not(pos == 'Verb' and lemma in STOP_WORDS)
        and len(lemma) > 1
    )

# 생성적 요약: O(L)
def generate_summary(text: str) -> str:
    inputs = tokenizer.encode("summarize: " + preprocess_text(text), return_tensors="pt")
    with torch.no_grad():
        outs = model.generate(
            inputs, max_length=150, num_beams=2, early_stopping=True
        )
    return tokenizer.decode(outs[0], skip_special_tokens=True)

# 백그라운드 생성 태스크
def background_generation(doc_id: int):
    doc_semaphore.acquire()
    try:
        db  = SessionLocal()
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

@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    source: SourceEnum = Form(...),
    file: UploadFile    = File(...),
    extractive_sentences: int = Form(3),
    do_generation: bool        = Form(False)
):
    print(f"[1] 요청 수신: source={source.value}")
    data   = await file.read()
    print(f"[2] 파일 수신 완료 ({len(data)} bytes)")
    spaced = perform_ocr_pages(data)                       # O(P+R)
    sents  = split_korean_sentences(spaced)                # O(R+S)
    with open("example.txt", "w") as f:
        f.write("\n".join(sents))                       # O(S)
    print(f"[4] example.txt에 {len(sents)}개 문장 저장됨")
    extractive = extractive_summary(spaced, extractive_sentences)
    print("[6] 추출적 요약 완료")
    db  = SessionLocal()
    doc = Document(
        source=source.value,
        ocr_text=spaced,
        extractive_summary=extractive,
        status=StatusEnum.PENDING
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    print(f"[8] DB 저장 완료: id={doc.id}")
    if do_generation:
        print(f"[9] 생성적 요약 태스크 예약: id={doc.id}")
        background_tasks.add_task(background_generation, doc.id)
    db.close()
    print(f"[10] 응답 전송: id={doc.id}")
    return {
        "doc_id": doc.id,
        "extractive_summary": extractive,
        "generation_started": do_generation
    }

@app.get("/status/{doc_id}")
async def get_status(doc_id: int):
    db  = SessionLocal()
    doc = db.get(Document, doc_id)
    db.close()
    if not doc:
        raise HTTPException(404, "Document not found")
    return {
        "doc_id": doc.id,
        "ocr_text": doc.ocr_text,
        "extractive_summary": doc.extractive_summary,
        "abstractive_summary": doc.abstractive_summary,
        "status": doc.status.value
    }

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
