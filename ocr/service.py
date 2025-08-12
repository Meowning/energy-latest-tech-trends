# -*- coding: utf-8 -*-
import os
import re
import threading
import tempfile
from enum import Enum
from datetime import datetime

import numpy as np
import cv2
import fitz  # PyMuPDF
import pytesseract

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Enum as SqlEnum
from sqlalchemy.orm import declarative_base, sessionmaker

import sys
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import ocrmypdf

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import kss
from pykospacing import Spacing


# =========================
# 전역 설정
# =========================
def _offline() -> bool:
    v = (os.getenv("TRANSFORMERS_OFFLINE", "0"), os.getenv("HF_HUB_OFFLINE", "0"))
    return any(str(x).lower() in ("1", "true", "yes") for x in v)


def load_t5():
    repo = os.getenv("T5_REMOTE_ID", "eenzeenee/t5-small-korean-summarization")
    off = _offline()
    tok = AutoTokenizer.from_pretrained(repo, use_fast=True, local_files_only=off)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, local_files_only=off)
    return tok, mdl


tokenizer, model = load_t5()

try:
    model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
except Exception as e:
    print("[warn] dynamic quantization skipped:", e)
model.eval().to("cpu")

# CPU 추론 스레드 최적화
try:
    torch.set_num_threads(max(1, os.cpu_count() or 2))
    torch.set_num_interop_threads(1)
except Exception:
    pass


def load_sbert():
    repo = os.getenv("SBERT_REMOTE_ID", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    off = _offline()
    return SentenceTransformer(repo, device="cpu")


sbert = load_sbert()
spacing = Spacing()

# DB 설정
DB_PATH = os.getenv("DB_PATH", "./ocr.db")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()
doc_semaphore = threading.Semaphore(MAX_WORKERS)
app = FastAPI()


# =========================
# ORM
# =========================
class StatusEnum(str, Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    DONE = "DONE"


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(Text, nullable=False)
    ocr_text = Column(Text, nullable=False)
    extractive_summary = Column(Text, nullable=False)
    abstractive_summary = Column(Text, nullable=True)
    status = Column(SqlEnum(StatusEnum), nullable=False, default=StatusEnum.PENDING)


Base.metadata.create_all(bind=engine)


# =========================
# 전처리 패턴
# =========================
CLEAN_KEYWORDS = r'(FAX|Mail|메일|E[- ]?mail|이메일|주소|초점|목차)'

def clean_full_text(text: str) -> str:
    print(f"[전처리] 원본 텍스트 길이: {len(text)}")

    # 1. 특수공백/개행 정리
    t = text.replace('\r', '\n').replace('\u00a0', ' ').replace('\u200b', ' ')

    # 2. 앞쪽 키워드 컷 (문서 앞 35%까지만 검색)
    search_limit = int(len(t) * 0.35)
    matches = list(re.finditer(CLEAN_KEYWORDS, t[:search_limit], flags=re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        print(f"[전처리] 키워드 '{last_match.group()}' 발견, 위치: {last_match.start()}")
        t = t[last_match.end():]

    # 3. 마지막 마침표 뒤 삭제
    last_dot = t.rfind('.')
    if last_dot != -1:
        print(f"[전처리] 마지막 마침표 위치: {last_dot}")
        t = t[:last_dot+1]

    # 4. "사진 + 숫자" 패턴 제거 (붙어있든 띄어있든 모두)
    t = re.sub(r'사진\s*\d+', '', t, flags=re.IGNORECASE)

    # 5. 단독 알파벳 1글자 삭제
    t = re.sub(r'\b[a-zA-Z]\b', '', t)

    # 6. 공백 제거
    t = re.sub(r'\s+', '', t)

    # 7. 허용문자 외 삭제
    t = re.sub(r'[^0-9A-Za-z가-힣\.\,\!\?…]', '', t)

    print(f"[전처리] 전처리 후 길이: {len(t)}")
    return t


# =========================
# OCR + 전처리
# =========================
def perform_ocr_pages(file_bytes: bytes) -> str:
    pages = []
    if file_bytes.startswith(b"%PDF"):
        print("[OCR] PDF 파일 감지")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(file_bytes)
            in_path = tf.name
        out_path = in_path.replace(".pdf", "_ocr.pdf")
        try:
            ocrmypdf.ocr(
                in_path, out_path, language="kor", skip_text=True, force_ocr=False,
                color_conversion_strategy="RGB", output_type="pdf",
                deskew=True, remove_background=True, jobs=int(os.getenv("OCR_JOBS", "2"))
            )
            doc = fitz.open(out_path)
            for i in range(doc.page_count):
                text_page = doc[i].get_text("text")
                print(f"[OCR] 페이지 {i+1} 텍스트 길이: {len(text_page)}")
                pages.append(text_page)
            doc.close()
        finally:
            for p in (in_path, out_path):
                try:
                    os.remove(p)
                except:
                    pass
    else:
        print("[OCR] 이미지 파일 감지")
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise HTTPException(400, "지원되지 않는 파일 형식")
        ocr_text = pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lang="kor")
        print(f"[OCR] 이미지 OCR 결과 길이: {len(ocr_text)}")
        pages.append(ocr_text)

    raw_text = "".join(pages)
    combined = clean_full_text(raw_text)

    os.makedirs("./processed", exist_ok=True)
    with open("./processed/before_spacing.txt", "w", encoding="utf-8") as f:
        f.write(combined)

    spaced = spacing(combined)
    print(f"[Spacing] 띄어쓰기 적용 후 길이: {len(spaced)}")

    with open("./processed/after_spacing.txt", "w", encoding="utf-8") as f:
        f.write(spaced)

    return spaced


# =========================
# 문장 분리 + 요약
# =========================
def is_noise_line(line: str) -> bool:
    t = line.strip()
    if len(t) < 10:
        return True
    if re.fullmatch(r"\d+(?:\s*\d+)*", t):
        return True
    if re.search(r"\b(vol\.?|TEL|FAX|E[- ]?mail|202\d|July|월|호기)\b", t, re.IGNORECASE):
        return True
    return False


def split_korean_sentences(text: str) -> list[str]:
    # 0. 소숫점 보호 (14.25 → 14<dot>25)
    protected = re.sub(r'(\d)\.(\d)', r'\1<dot>\2', text)

    try:
        raws = kss.split_sentences(protected, use_quotes_brackets_processing=False, ignore_quotes_or_brackets=True)
    except Exception:
        raws = re.split(r'(?<=[\.!\?…])\s*', protected)

    # 1. 복원 (<dot> → .)
    raws = [r.replace('<dot>', '.') for r in raws]

    # 2. 잡음 제거
    sents = [s.strip() for s in raws if s.strip() and not is_noise_line(s)]
    print(f"[문장 분리] 문장 개수: {len(sents)}")
    return sents


def _preprocess_for_embed(text: str) -> str:
    txt = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    txt = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', txt)
    txt = re.sub(r"[\r\n\t]+", ' ', txt)
    txt = re.sub(r"[^0-9A-Za-z가-힣\s\.\?!]", ' ', txt)
    txt = re.sub(r"\s+", ' ', txt).strip()
    return txt


def extractive_summary(sent_list: list[str], num_sentences: int = 3) -> str:
    if not sent_list:
        return ""
    proc = [_preprocess_for_embed(s) for s in sent_list]
    embs = sbert.encode(proc, convert_to_numpy=True, show_progress_bar=False)
    n = max(1, min(num_sentences, len(embs)))
    clust = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
    labels = clust.fit_predict(embs)
    picks = []
    for lbl in sorted(set(labels)):
        idxs = np.where(labels == lbl)[0]
        cent = embs[idxs].mean(axis=0)
        sims = cosine_similarity([cent], embs[idxs])[0]
        best = idxs[sims.argmax()]
        picks.append((best, sent_list[best]))
    picks.sort(key=lambda x: x[0])
    result = "\n".join(f"[{i+1}] {s}" for i, s in picks)
    print(f"[추출 요약] {len(picks)}문장 선택")
    return result


def _chunk_tokens(tokens: list[int], chunk_size: int, overlap: int = 32):
    if chunk_size <= 0:
        yield tokens
        return
    i, n = 0, len(tokens)
    while i < n:
        j = min(i + chunk_size, n)
        yield tokens[i:j]
        if j >= n:
            break
        i = max(0, j - overlap)


# -------------------------
# 생성 요약(속도↑, 진행률, 3문장 마침표) 보조함수
# -------------------------
def _token_len(txt: str) -> int:
    return tokenizer.encode(txt, add_special_tokens=True, return_tensors="pt").shape[1]

def _normalize_periods(txt: str) -> str:
    t = re.sub(r"\s+", " ", txt).strip()
    t = re.sub(r"(?<![\.?!])\s*(다|이다|합니다|했습니다|했다|한다)\s*$", r"\1.", t)
    t = re.sub(r"\.{2,}", ".", t)
    return t

def _to_exact_n_sentences(txt: str, n: int) -> list[str]:
    protected = re.sub(r"(\d)\.(\d)", r"\1<dot>\2", txt)
    parts = [p.strip() for p in re.split(r"(?<=\.)\s+", protected) if p.strip()]
    parts = [p.replace("<dot>", ".") for p in parts]
    if not parts:
        return []
    while len(parts) < n:
        parts[-1] = parts[-1].rstrip(".") + "."
        parts.append(parts[-1])
    parts = parts[:n]
    parts = [p if p.endswith(".") else p + "." for p in parts]
    return parts

def _decode_generate(input_ids, decoding_kwargs):
    with torch.inference_mode():
        outs = model.generate(input_ids, **decoding_kwargs)
    return tokenizer.decode(outs[0], skip_special_tokens=True)

def _summarize_stuff(base_text: str, lines: int, decoding_kwargs: dict) -> str:
    prompt = f"Summarize the following text into exactly {lines} sentences in Korean. Use periods only: "
    ids = tokenizer.encode(prompt + base_text, return_tensors="pt", add_special_tokens=True)
    cand = _decode_generate(ids, decoding_kwargs)
    cand = _normalize_periods(cand)
    parts = _to_exact_n_sentences(cand, lines)
    return "\n".join(parts) if parts else ""

def _refine_step(summary: str, chunk_text: str, lines: int, decoding_kwargs: dict) -> str:
    prompt = (
        f"You are refining an existing Korean summary into exactly {lines} sentences using periods only.\n"
        f"CURRENT SUMMARY:\n{summary}\n\n"
        f"NEW CONTEXT (keep only key facts):\n{chunk_text}\n\n"
        f"Provide an improved Korean summary (exactly {lines} sentences, with periods only)."
    )
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    cand = _decode_generate(ids, decoding_kwargs)
    return _normalize_periods(cand)

def _precompress_ultra_long(sent_list: list[str], max_tokens: int) -> list[str]:
    kept = []
    cur = ""
    step = 1
    if len(sent_list) > 3000:
        step = 2
    for s in sent_list[::step]:
        nxt = (cur + " " + s).strip()
        if _token_len(nxt) > max_tokens:
            break
        cur = nxt
        kept.append(s)
    return kept if kept else sent_list[: max(1, int(len(sent_list) * 0.1))]

def _summarize_refine(base_text: str, lines: int, decoding_kwargs: dict) -> str:
    # 큰 청크 / 작은 오버랩 → 호출 횟수 감소(속도↑)
    CHUNK_TOKENS = int(os.getenv("REFINE_CHUNK_TOKENS", "768"))
    OVERLAP = int(os.getenv("REFINE_CHUNK_OVERLAP", "24"))

    all_tokens = tokenizer.encode(base_text, add_special_tokens=True)
    chunks = list(_chunk_tokens(all_tokens, CHUNK_TOKENS, overlap=OVERLAP))
    total_chunks = len(chunks)

    # 1) 첫 청크: Stuff
    first_text = tokenizer.decode(chunks[0], skip_special_tokens=True)
    summary = _summarize_stuff(first_text, lines, decoding_kwargs)

    # 2) 나머지 청크: refine + 진행률 한 줄 갱신
    for i, tok_chunk in enumerate(chunks[1:], start=2):
        sys.stdout.write(f"\r[Refine 진행률] {i}/{total_chunks} 청크 처리 중...")
        sys.stdout.flush()

        chunk_text = tokenizer.decode(tok_chunk, skip_special_tokens=True)

        # 프롬프트 길이 제어: NEW CONTEXT는 512토큰까지만 사용
        c_ids = tokenizer.encode(chunk_text, add_special_tokens=False)
        if len(c_ids) > 512:
            c_ids = c_ids[:512]
            chunk_text = tokenizer.decode(c_ids, skip_special_tokens=True)

        summary = _refine_step(summary, chunk_text, lines, decoding_kwargs)

    sys.stdout.write("\n")
    sys.stdout.flush()

    parts = _to_exact_n_sentences(summary, lines)
    return "\n".join(parts) if parts else ""

def generate_summary(sent_list: list[str], lines: int = 3) -> str:
    if not sent_list:
        return ""

    # 초장문 사전 압축 (속도 핵심)
    ULTRA_LONG_TOKENS = int(os.getenv("ULTRA_LONG_TOKENS", "8000"))
    PRECOMPRESS_TO = int(os.getenv("PRECOMPRESS_TO_TOKENS", "4000"))

    base_text_full = re.sub(r"\s+", " ", " ".join(sent_list)).strip()
    in_len_full = _token_len(base_text_full)

    if in_len_full > ULTRA_LONG_TOKENS:
        reduced_sents = _precompress_ultra_long(sent_list, PRECOMPRESS_TO)
        base_text = re.sub(r"\s+", " ", " ".join(reduced_sents)).strip()
        print(f"[사전압축] {in_len_full}→{_token_len(base_text)} 토큰, 문장 {len(sent_list)}→{len(reduced_sents)}")
    else:
        base_text = base_text_full

    # 디코딩(가볍게)
    DECODING = dict(
        max_new_tokens=int(os.getenv("T5_MAX_NEW_TOKENS", "120")),
        min_new_tokens=int(os.getenv("T5_MIN_NEW_TOKENS", "45")),
        num_beams=int(os.getenv("T5_NUM_BEAMS", "2")),
        do_sample=False,
        use_cache=True,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    # Stuff/Refine 스위칭
    STUFF_MAX_INPUT = int(os.getenv("STUFF_MAX_INPUT_TOKENS", "550"))
    in_len = _token_len(base_text)

    if in_len <= STUFF_MAX_INPUT:
        cand = _summarize_stuff(base_text, lines, DECODING)
        print(f"[생성 요약] Stuff 사용 (입력 토큰 {in_len})")
        return cand
    else:
        cand = _summarize_refine(base_text, lines, DECODING)
        print(f"[생성 요약] Refine 사용 (입력 토큰 {in_len})")
        return cand


# =========================
# 백그라운드 태스크 & API
# =========================
def background_generation(doc_id: int):
    doc_semaphore.acquire()
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            return
        doc.status = StatusEnum.PARTIAL
        db.commit()
        sents = [s for s in (doc.ocr_text or "").split("\n") if s.strip()]
        doc.abstractive_summary = generate_summary(sents, lines=3)
        doc.status = StatusEnum.DONE
        print(f"[생성 요약] 완료: {doc.abstractive_summary}")
        db.commit()
    finally:
        doc_semaphore.release()
        db.close()


@app.post("/process")
async def process(background_tasks: BackgroundTasks,
                  source: str = Form(...),
                  file: UploadFile = File(...),
                  extractive_sentences: int = Form(3),
                  do_generation: bool = Form(False)):
    data = await file.read()
    spaced = perform_ocr_pages(data)
    sents = split_korean_sentences(spaced)
    with open("./example.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sents))
    extractive = extractive_summary(sents, extractive_sentences)
    db = SessionLocal()
    try:
        doc = Document(source=source,
                       ocr_text="\n".join(sents),
                       extractive_summary=extractive,
                       status=StatusEnum.PENDING)
        db.add(doc)
        db.commit()
        db.refresh(doc)
        if do_generation:
            background_tasks.add_task(background_generation, doc.id)
        return {"doc_id": doc.id, "extractive_summary": extractive, "generation_started": do_generation}
    finally:
        db.close()


@app.get("/status/{doc_id}")
async def get_status(doc_id: int):
    db = SessionLocal()
    try:
        doc = db.get(Document, doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        return {"doc_id": doc.id, "ocr_text": doc.ocr_text,
                "extractive_summary": doc.extractive_summary,
                "abstractive_summary": doc.abstractive_summary,
                "status": doc.status.value}
    finally:
        db.close()


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
