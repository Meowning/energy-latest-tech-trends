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


def _pick_device(env_key: str, default_auto: bool = True) -> str:
    """
    env_key로 원하는 디바이스를 받되, 사용 불가하면 안전하게 CPU로 폴백.
    지원: "cuda", "mps", "cpu". 기본은 자동 감지(cuda>mps>cpu).
    """
    want = os.getenv(env_key, "").strip().lower()
    if want in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        print(f"[warn] {env_key}=cuda 요청됐지만 CUDA 미사용 → cpu로 폴백")
        return "cpu"
    if want == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        print(f"[warn] {env_key}=mps 요청됐지만 MPS 미사용 → cpu로 폴백")
        return "cpu"
    if want == "cpu":
        return "cpu"

    if default_auto:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        return "cpu"
    return "cpu"


def load_t5():
    repo = os.getenv("T5_REMOTE_ID", "eenzeenee/t5-small-korean-summarization")
    off = _offline()
    device = _pick_device("T5_DEVICE")  # auto: cuda>mps>cpu

    tok = AutoTokenizer.from_pretrained(repo, use_fast=True, local_files_only=off)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, local_files_only=off)

    # CPU 전용 동적 양자화 (GPU/MPS에서는 금지)
    want_quant = os.getenv("T5_QUANTIZE", "1").lower() in ("1", "true", "yes")
    if device == "cpu" and want_quant:
        try:
            mdl = torch.quantization.quantize_dynamic(mdl, {torch.nn.Linear}, dtype=torch.qint8)
            print("[info] T5 dynamic quantization applied (CPU)")
        except Exception as e:
            print("[warn] dynamic quantization skipped:", e)

    mdl.eval().to(device)
    print(f"[info] T5 loaded on {device.upper()} (repo={repo})")
    return tok, mdl


tokenizer, model = load_t5()

# CPU 추론 스레드 최적화 (GPU/MPS이면 무시)
try:
    if next(model.parameters()).device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 2))
        torch.set_num_interop_threads(1)
        print(f"[info] torch threads set: {torch.get_num_threads()} / interop 1")
except Exception:
    pass


def load_sbert():
    repo = os.getenv("SBERT_REMOTE_ID", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    off = _offline()
    device = _pick_device("SBERT_DEVICE")  # 기본 T5와 동일한 자동 선택

    # SentenceTransformer는 device 인자로 CPU/GPU 전환
    model = SentenceTransformer(repo, device=device, cache_folder=None)  # cache는 기본 경로 사용
    print(f"[info] SBERT loaded on {device.upper()} (repo={repo})")
    return model


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
CLEAN_KEYWORDS = (
    r"(FAX|Mail|메일|E[- ]?mail|이메일|주소|초점|목차|"
    r"제\s*\d{4}\s*\d{1,2}\s*\d{4}\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])\.?)"
)
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
                # print(f"[OCR] 페이지 {i+1} 텍스트 길이: {len(text_page)}")
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
    with open("./processed/ocr.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)
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
# 문장 분리 + 요약(추출/임베딩)
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
        raws = re.split(r'(?<=[\.\!\?…])\s*', protected)

    # 1. 복원 (<dot> → .)
    raws = [r.replace('<dot>', '.') for r in raws]

    # 2. 잡음 제거
    sents = [s.strip() for s in raws if s.strip() and not is_noise_line(s)]
    print(f"[문장 분리] 문장 개수: {len(sents)}")
    return sents


def _preprocess_for_embed(text: str) -> str:
    txt = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    txt = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", ' ', txt)
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
    result = "\n".join(f"[{i+1}] {s}" for i, s in enumerate([p[1] for p in picks]))
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
# 생성 요약 유틸(형식/중복)
# -------------------------
def _token_len(txt: str) -> int:
    return tokenizer.encode(txt, add_special_tokens=True, return_tensors="pt").shape[1]


def _normalize_periods(txt: str) -> str:
    t = re.sub(r"\s+", " ", txt).strip()
    t = re.sub(r"(?<![\.!?])\s*(다|이다|합니다|했습니다|했다|한다)\s*$", r"\1.", t)
    t = re.sub(r"\.{2,}", ".", t)
    return t


def _dedup_keep_order(seq: list[str]) -> list[str]:
    seen = set(); out = []
    for s in seq:
        key = re.sub(r"\s+", " ", s).strip()
        if key and key not in seen:
            seen.add(key); out.append(key)
    return out


def _parse_numbered_or_periods(cand: str, n: int) -> list[str]:
    # (a) 번호 목록 형식
    lines = [x.strip() for x in cand.splitlines() if x.strip()]
    items = []
    for ln in lines:
        ln = re.sub(r'^\s*(?:\d+[\.\)]|[-•·∙ㆍ‧])\s*', '', ln).strip()
        if ln:
            items.append(ln)
    items = _dedup_keep_order(items)
    if len(items) >= n:
        return [(s if s.endswith('.') else s + '.') for s in items[:n]]

    # (b) 마침표 기준
    protected = re.sub(r"(\d)\.(\d)", r"\1<dot>\2", cand)
    parts = [p.strip().replace("<dot>", ".") for p in re.split(r"(?<=\.)\s+", protected) if p.strip()]
    parts = _dedup_keep_order(parts)
    if len(parts) >= n:
        return parts[:n]

    # (c) 세미콜론/쉼표/접속어 보조 분할
    tmp = parts[:] if parts else [cand]
    clauses = []
    for t in tmp:
        chunks = re.split(r';\s*', t)
        if len(chunks) == 1:
            chunks = re.split(r',\s*(?=(그리고|또한|따라서|하지만|그러나)\b)', t)
        for c in chunks:
            c = c.strip(' ,;')
            if len(c) >= 8:
                clauses.append(c if c.endswith('.') else c + '.')
            if len(clauses) >= n:
                break
        if len(clauses) >= n:
            break
    return _dedup_keep_order(clauses)[:n]


def _ensure_n_distinct_sentences(cand: str, n: int, decoding_kwargs: dict, outline: list[str]) -> list[str]:
    parts = _parse_numbered_or_periods(_normalize_periods(cand), n)
    if len(parts) >= n:
        return parts[:n]

    # 1차: 짧은 재생성(현행 유지)
    prompt = (
        f"아래 요약을 바탕으로 서로 다른 핵심 포인트 {n}가지를 한 문장씩 써라.\n"
        f"출력 형식: '1. ...' 줄바꿈, 각 문장은 마침표로 끝낼 것. 중복 금지.\n"
        f"모든 문장은 서로 다른 정보를 담아 반드시 {n}문장을 출력할 것.\n\n"
        f"[요약]\n{cand}"
    )
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    with torch.inference_mode():
        outs = model.generate(ids, **decoding_kwargs)
    retry = tokenizer.decode(outs[0], skip_special_tokens=True)
    parts = _parse_numbered_or_periods(_normalize_periods(retry), n)
    parts = _dedup_keep_order(parts)

    # 2차: 남는 칸은 outline에서 채우기 (복붙 금지)
    if len(parts) < n:
        for s in outline:
            ss = _normalize_periods(s)
            if ss and ss not in parts:
                parts.append(ss if ss.endswith('.') else ss + '.')
            if len(parts) >= n:
                break

    return parts[:n]

# -------------------------
# 디코딩/프롬프트 구성
# -------------------------
def _decoding_cfg():
    return dict(
        max_new_tokens=int(os.getenv("T5_MAX_NEW_TOKENS", "150")),
        min_new_tokens=int(os.getenv("T5_MIN_NEW_TOKENS", "40")),
        num_beams=int(os.getenv("T5_NUM_BEAMS", "2")),
        do_sample=False,
        use_cache=True,
        no_repeat_ngram_size=int(os.getenv("NO_REPEAT_NGRAM", "4")),
        repetition_penalty=float(os.getenv("REPETITION_PENALTY", "1.1")),
        early_stopping=True,
    )


def _decode_generate(input_ids, decoding_kwargs):
    with torch.inference_mode():
        outs = model.generate(input_ids, **decoding_kwargs)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


# -------------------------
# 전역 아웃라인(임베딩 기반 대표 문장) & 적응형 청크
# -------------------------
def _build_global_outline(sent_list: list[str], max_items: int = 10) -> list[str]:
    if not sent_list:
        return []
    sents = sent_list[:4000]  # 안전장치
    proc = [_preprocess_for_embed(s) for s in sents]
    embs = sbert.encode(proc, convert_to_numpy=True, show_progress_bar=False)

    k = min(max_items, max(3, int(len(embs) ** 0.5)))  # √N 수준
    clust = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
    labels = clust.fit_predict(embs)

    picks = []
    for lbl in sorted(set(labels)):
        idxs = np.where(labels == lbl)[0]
        cent = embs[idxs].mean(axis=0)
        sims = cosine_similarity([cent], embs[idxs])[0]
        best = idxs[sims.argmax()]
        picks.append((best, sents[best]))
    picks.sort(key=lambda x: x[0])
    return [x[1] for x in picks]


def _adaptive_chunk_params(total_tokens: int) -> tuple[int, int]:
    target_steps = int(os.getenv("REFINE_TARGET_STEPS", "24"))
    chunk = max(384, min(1024, total_tokens // max(1, target_steps)))
    overlap = 48 if chunk <= 640 else 32
    return chunk, overlap


# -------------------------
# Stuff/Refine 본체 (Outline 주입)
# -------------------------
def _summarize_stuff_with_outline(base_text: str, outline: list[str], lines: int, decoding_kwargs: dict) -> str:
    outline_text = "\n".join(f"- {s}" for s in outline[:12])
    prompt = (
        f"다음 텍스트를 한국어로 서로 다른 {lines}개의 핵심 문장으로 요약하라.\n"
        f"각 문장은 15~40자, 정보 중복 금지. 전역 윤곽을 우선 반영.\n"
        f"출력 형식: '1. ...' 줄바꿈, 마침표로 끝낼 것.\n"
        f"모든 문장은 서로 다른 핵심 정보를 담아야 하며, 반드시 {lines}문장을 모두 작성할 것.\n"
        f"동일하거나 유사한 내용의 문장은 금지하며, 중복 시 다른 내용으로 대체할 것.\n\n"
        f"[전역 윤곽]\n{outline_text}\n\n"
        f"[본문]\n{base_text}"
    )
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    cand = _decode_generate(ids, decoding_kwargs)
    parts = _ensure_n_distinct_sentences(cand, lines, decoding_kwargs, outline)
    return "\n".join(parts)


def _summarize_refine_with_outline(full_text: str, outline: list[str], lines: int) -> str:
    total_tokens = _token_len(full_text)
    CHUNK_TOKENS, OVERLAP = _adaptive_chunk_params(total_tokens)
    decoding_kwargs = _decoding_cfg()

    all_tokens = tokenizer.encode(full_text, add_special_tokens=True)
    chunks = list(_chunk_tokens(all_tokens, CHUNK_TOKENS, overlap=OVERLAP))
    total_chunks = len(chunks)

    # 1) 첫 청크: Stuff(+Outline)
    first_text = tokenizer.decode(chunks[0], skip_special_tokens=True)
    summary = _summarize_stuff_with_outline(first_text, outline, lines, decoding_kwargs)

    # 2) 이후 청크: Refine(+Outline) + 진행률
    outline_text = "\n".join(f"- {s}" for s in outline[:12])
    for i, tok_chunk in enumerate(chunks[1:], start=2):
        sys.stdout.write(f"\r[Refine 진행률] {i}/{total_chunks} 청크 처리 중...")
        sys.stdout.flush()

        chunk_text = tokenizer.decode(tok_chunk, skip_special_tokens=True)
        c_ids = tokenizer.encode(chunk_text, add_special_tokens=False)
        if len(c_ids) > 640:
            c_ids = c_ids[:640]
            chunk_text = tokenizer.decode(c_ids, skip_special_tokens=True)

        prompt = (
            f"전역 윤곽을 유지하며 현재 요약을 새로운 컨텍스트로 보강하라.\n"
            f"결과는 서로 다른 {lines}문장, 각 15~40자, 중복 금지.\n"
            f"출력 형식: '1. ...' 줄바꿈, 마침표로 끝낼 것.\n\n"
            f"[전역 윤곽]\n{outline_text}\n\n"
            f"[현재 요약]\n{summary}\n\n"
            f"[새 컨텍스트]\n{chunk_text}"
        )
        ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        cand = _decode_generate(ids, decoding_kwargs)
        parts = _ensure_n_distinct_sentences(cand, lines, decoding_kwargs, outline)
        summary = "\n".join(parts)

    sys.stdout.write("\n"); sys.stdout.flush()
    return summary


# -------------------------
# (비상용) 초장문 압축 유틸
# -------------------------
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


# -------------------------
# 공개 함수: generate_summary (Outline 우선, 압축은 비상용)
# -------------------------
def generate_summary(sent_list: list[str], lines: int = 3) -> str:
    if not sent_list:
        return ""

    base_text_full = re.sub(r"\s+", " ", " ".join(sent_list)).strip()
    in_len_full = _token_len(base_text_full)

    # 비상: 정말 큰 입력만 사전 압축 (기본 40k 이상)
    ULTRA_LONG_EMERGENCY = int(os.getenv("ULTRA_LONG_EMERGENCY_TOKENS", "40000"))
    if in_len_full > ULTRA_LONG_EMERGENCY:
        print(f"[경고] 입력 {in_len_full} 토큰: 비상 축약 수행")
        budget = int(os.getenv("PRECOMPRESS_TO_TOKENS", "8000"))
        reduced = _precompress_ultra_long(sent_list, budget)
        base_text = re.sub(r"\s+", " ", " ".join(reduced)).strip()
    else:
        base_text = base_text_full

    # 전역 아웃라인(대표 문장) 추출 → 모든 단계에 주입
    outline = _build_global_outline(sent_list, max_items=int(os.getenv("OUTLINE_ITEMS", "10")))

    # 길이 기준으로 Stuff/Refine 스위칭
    STUFF_MAX_INPUT = int(os.getenv("STUFF_MAX_INPUT_TOKENS", "550"))
    if _token_len(base_text) <= STUFF_MAX_INPUT:
        out = _summarize_stuff_with_outline(base_text, outline, lines, _decoding_cfg())
        print(f"[생성 요약] Stuff(+Outline) 사용 (입력 토큰 {_token_len(base_text)})")
        return out
    else:
        out = _summarize_refine_with_outline(base_text, outline, lines)
        print(f"[생성 요약] Refine(+Outline) 사용 (입력 토큰 {_token_len(base_text)})")
        return out


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
