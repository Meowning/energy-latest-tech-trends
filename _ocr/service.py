# -*- coding: utf-8 -*-
import os
import re
import threading
import tempfile
from enum import Enum
from datetime import datetime
import unicodedata
from math import ceil

import numpy as np
import cv2
import fitz  # PyMuPDF
import pytesseract

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Enum as SqlEnum
from sqlalchemy.orm import declarative_base, sessionmaker

import sys
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
    device = _pick_device("T5_DEVICE")

    tok = AutoTokenizer.from_pretrained(repo, use_fast=True, local_files_only=off)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(repo, local_files_only=off)

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
    device = _pick_device("SBERT_DEVICE")
    model = SentenceTransformer(repo, device=device, cache_folder=None)
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
# 전처리 패턴 (source별 스위칭 지원)
# =========================
# 숫자와 함께 보존할 단위 (연/분기/반기 포함)
PRESERVE_UNITS = (
    # --- 일반 한국어 수량/시간 ---
    r"(건|개|명|가구|년|년도|연도|월|일|시|분|초|차|회|분기|반기|상반기|하반기|%)"

    # --- 온도 ---
    r"|(?:°C|℃|°F|℉|K|degC|degF)"

    # --- 길이 ---
    r"|(?:nm|μm|um|mm|cm|m|km|Å)"

    # --- 면적 ---
    r"|(?:mm2|cm2|m2|km2|mm²|cm²|m²|km²|㎡|ha)"

    # --- 부피 ---
    r"|(?:mm3|cm3|m3|km3|mm³|cm³|m³|km³|㎥|L|mL|μL|uL|kL|ML|GL|ℓ)"

    # --- 질량 ---
    r"|(?:mg|g|kg|t|kt|Mt|Gt)"

    # --- 농도/밀도/미세먼지 ---
    r"|(?:ppm|ppb|ppt)"
    r"|(?:μg|ug|mg|ng|g)/(?:L|m3|m\^?3|㎥|kg|m2|m\^?2|㎡)"
    r"|(?:PM(?:10|2\.5))"

    # --- 에너지/전력/전력량 ---
    r"|(?:J|kJ|MJ|GJ|TJ|PJ)"
    r"|(?:W|kW|MW|GW|TW)"
    r"|(?:Wh|kWh|MWh|GWh|TWh|PWh)"
    r"|(?:kcal|Cal)"

    # --- 연료환산/석유당량 ---
    r"|(?:toe|ktoe|Mtoe|Gtoe|boe|Mboe)"

    # --- 배출량/강도 ---
    r"|(?:tCO2e|tCO2eq|CO2e|CO2eq|kgCO2e|ktCO2e|MtCO2e)"
    r"|(?:gCO2(?:/kWh|/MJ)|kgCO2(?:/kWh|/MWh))"

    # --- 방사선 ---
    r"|(?:(?:nSv|μSv|uSv|mSv|Sv)(?:/(?:h|hr|d|day|yr|y|a)|·h(?:\^?-?1|⁻¹)|/시간|/년)?)"
    r"|(?:(?:nGy|μGy|uGy|mGy|Gy)(?:/(?:h|hr|d|day|yr|y|a)|·h(?:\^?-?1|⁻¹)|/시간|/년)?)"
    r"|(?:(?:μrem|urem|mrem|rem)(?:/(?:h|hr|d|day|yr|y|a))?)"
    r"|(?:[kMGT]?Bq(?:/(?:m3|m\^?3|㎥|L|kg|m2|m\^?2|㎡))?)"
    r"|(?:Ci|mCi|μCi|uCi|kCi)"
    r"|(?:cpm|cps|dpm)"

    # --- 압력 ---
    r"|(?:Pa|kPa|MPa|GPa|bar|mbar|hPa|atm|Torr|mmHg)"

    # --- 유량/유속/회전수 ---
    r"|(?:m3/s|m3/h|Nm3/h|Sm3/h|L/s|L/min|L/h|mL/min)"
    r"|(?:m/s|km/h|rpm|rps)"

    # --- 주파수/음압 ---
    r"|(?:Hz|kHz|MHz|GHz|THz|dB)"

    # --- 전기 ---
    r"|(?:V|kV|A|mA|μA|uA)"
    r"|(?:Ω|ohm|kΩ|MΩ)"
    r"|(?:S|mS|μS|uS)"
    r"|(?:F|mF|μF|uF|nF)"
    r"|(?:H|mH|μH|uH)"

    # --- 광/각/휘도/탁도 ---
    r"|(?:rad|mrad|sr)"
    r"|(?:lx|lm|cd|nit|nt)"
    r"|(?:NTU)"

    # --- 통화(한글) ---
    r"|(?:원|천원|만원|백만원|억원|조원)"
)

DEFAULT_CLEAN_KEYWORDS = (
    r"(FAX|Mail|메일|E[- ]?mail|이메일|주소|목차|"
    r"제\s*\d{4}\s*\d{1,2}\s*\d{4}\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])\.?)"
)

CLEAN_KEYWORDS_BY_SOURCE: dict[str, str] = {
    "한국원자력안전재단": r"초점",
    "한국원자력산업협회": r"",
    "에너지경제연구원": r"",
    "한전경영연구원": r"",
    "산업통상자원부": r"",
    "한국원자력연구원": r"",
     "원자력안전위원회": r"(?m)^\s*제\s*(?:1|I|一)\s*장",
}

_COMPILED_CLEAN_REGEX: dict[str, re.Pattern] = {}


def get_clean_keywords(source: str) -> re.Pattern:
    """
    source별 맞춤 정규식 반환.
    - 매핑이 없거나 빈 문자열이면 DEFAULT_CLEAN_KEYWORDS 사용
    - IGNORECASE + MULTILINE로 컴파일
    """
    key = (source or "").strip()
    pattern = CLEAN_KEYWORDS_BY_SOURCE.get(key) or DEFAULT_CLEAN_KEYWORDS

    if pattern not in _COMPILED_CLEAN_REGEX:
        _COMPILED_CLEAN_REGEX[pattern] = re.compile(
            pattern, flags=re.IGNORECASE | re.MULTILINE
        )
    return _COMPILED_CLEAN_REGEX[pattern]

# =========================
# ASCII/유니코드 단위 정규화
# =========================
def normalize_units_for_ascii(t: str) -> str:
    repl = {
        "㎡": "m2", "㎥": "m3", "㎤": "cm3", "㎣": "mm3", "㎦": "km3",
        "㎟": "mm2", "㎠": "cm2", "㎢": "km2",
        "㎜": "mm", "㎝": "cm", "㎞": "km",
        "㎍": "ug", "㎎": "mg", "㎏": "kg",
        "ℓ": "L",
        "℃": "degC", "℉": "degF",
        "Ω": "ohm",
        "㎾": "kW", "㎿": "MW",
        "㎸": "kV", "㎹": "MV", "㎶": "mV",
        "㎄": "kA", "㎃": "mA", "㎂": "uA",
        "㎌": "uF",
        "㎩": "kPa", "㎫": "MPa", "㎬": "GPa", "㎭": "rad",
        "㎧": "m/s", "㎨": "m/s2",
        "㎖": "mL", "㎕": "uL", "㎘": "kL",
        "㎐": "Hz", "㎑": "kHz", "㎒": "MHz", "㎓": "GHz",
        "㏈": "dB",
        "㏃": "Bq", "㏅": "cd", "㏗": "pH", "㏄": "mL",
        "％": "%", "／": "/", "－": "-", "–": "-", "—": "-",
        "µ": "u", "μ": "u",
        "²": "2", "³": "3",
    }
    for k, v in repl.items():
        t = t.replace(k, v)

    t = re.sub(r"\u00B7\s*h(?:\^-?1|⁻1|⁻¹)", r"/h", t)
    t = re.sub(r"\u00B7\s*s(?:\^-?1|⁻1|⁻¹)", r"/s", t)
    t = re.sub(r"\u00B7\s*(?:yr|y|a)(?:\^-?1|⁻1|⁻¹)", r"/yr", t)
    t = re.sub(r"\u00B7\s*h-?1", r"/h", t)
    t = re.sub(r"\u00B7\s*s-?1", r"/s", t)

    t = re.sub(r"([A-Za-z])\s*2\b", r"\g<1>2", t)
    t = re.sub(r"([A-Za-z])\s*3\b", r"\g<1>3", t)
    return t


# =========================
# 페이지 기반 앞부분 컷 (헤더 보존)
# =========================
def _normalize_for_header_match(t: str) -> str:
    # 헤더 매칭 전용 얕은 정규화(인덱스 영향 최소화: 길이 변화 없는 치환 위주)
    t = t.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    t = t.replace("／", "/").replace("－", "-").replace("–", "-").replace("—", "-")
    t = re.sub(r"[\u2000-\u200b\u202f\u205f\u3000]", " ", t)
    return t


def page_based_front_cut(pages: list[str], source: str) -> str:
    if not pages:
        return ""

    head_end = min(max(1, ceil(len(pages) * 0.05)), len(pages))
    ck = get_clean_keywords(source)

    full_text = "".join(pages)
    head_text = "".join(pages[:head_end])

    # (수정) 매칭 전용 정규화: NFKC로 전각/로마숫자/기호를 ASCII로
    def norm(t: str) -> str:
        t = unicodedata.normalize("NFKC", t)
        # 얇은/전각 공백류를 일반 공백으로
        t = re.sub(r"[\u2000-\u200b\u202f\u205f\u3000]", " ", t)
        return t

    head_norm = norm(head_text)
    full_norm  = norm(full_text)

    head_matches = list(ck.finditer(head_norm))
    if head_matches:
        last = head_matches[-1]
        print(f"[앞부분컷] 5% window={head_end}p | source='{source}' | pattern='{ck.pattern}'")
        print(f"[앞부분컷] 마지막 매치: '{last.group()}' (pos={last.start()}~{last.end()}) → end() 이후 시작")
        return full_text[last.end():]

    first_full = ck.search(full_norm)
    if first_full:
        print(f"[앞부분컷] 5% window={head_end}p | source='{source}' | pattern='{ck.pattern}'")
        print(f"[앞부분컷] 전체 첫 매치: '{first_full.group()}' (pos={first_full.start()}) → start()부터 시작")
        return full_text[first_full.start():]

    print(f"[앞부분컷] 5% window={head_end}p | source='{source}' | pattern='{ck.pattern}' → 매치 없음(스킵)")
    return full_text

# (추가) START/END 보호구간(␞␟) 바깥에서만 숫자를 제거
def _remove_digits_outside_protection(t: str) -> str:
    START, END = "\u241E", "\u241F"   # ␞, ␟
    out = []
    i = 0
    n = len(t)
    while i < n:
        s = t.find(START, i)
        if s == -1:
            # 남은 꼬리 전부 바깥 영역 → 숫자 제거
            out.append(re.sub(r"\d+", " ", t[i:]))
            break
        # (i ~ s) 구간: 바깥 영역 → 숫자 제거
        out.append(re.sub(r"\d+", " ", t[i:s]))
        e = t.find(END, s + 1)
        if e == -1:
            # END 못 찾으면 안전하게 나머지 전부 바깥 취급
            out.append(re.sub(r"\d+", " ", t[s:]))
            break
        # (s ~ e] 구간: 보호 영역 → 그대로 둠
        out.append(t[s:e+1])
        i = e + 1
    return "".join(out)

# =========================
# 전처리 본체 (앞부분 컷은 여기서 안 함)
# =========================
def clean_full_text(text: str, source: str) -> str:
    """
    - 숫자는 지우되 '숫자+단위'와 '날짜/연도'는 보존
    - 표/그림/사진 캡션 제거(줄 전체 + 본문 토막)
    - 허용 기호: . , ! ? … / - 유지
    - 마지막엔 공백 제거 (pykospacing에서 복원)
    """
    print(f"[전처리] 원본 텍스트 길이: {len(text)}")
    t = text.replace('\r', '\n').replace('\u00a0', ' ').replace('\u200b', ' ')
    t = normalize_units_for_ascii(t)

    # --- 캡션 제거 ---
    t = re.sub(
        r"(?im)^\s*[\(\[]?\s*(?:표|그림|사진)\s*\d+(?:[.\-–]\d+)*\s*[\)\]]?(?:\s*[.:]\s*)?.*$",
        "", t
    )
    t = re.sub(r"[\(\[]?\s*(?:표|그림|사진)\s*\d+(?:[.\-–]\d+)*\s*[\)\]]?", "", t)
    t = re.sub(r"\b(?:표|그림|사진)\b", "", t, flags=re.IGNORECASE)

    # --- 보호 마커 ---
    START, END = "\u241E", "\u241F"   # ␞, ␟

    # ===== 날짜/연도 먼저 보호 (개행/띄어쓰기 허용) =====
    # 2024년 / 2024년도 / 2024 연도
    pat_year = re.compile(r"(\d{4})\s*(년|년도|연도)")
    # 2024-01-15 / 2024.1.15 / 2024/01(/15)
    pat_date = re.compile(r"(\d{4})[./-](\d{1,2})(?:[./-](\d{1,2}))?")
    # 1월 / 3일
    pat_md   = re.compile(r"(\d{1,2})\s*(월|일)")

    def _wrap(m: re.Match) -> str:
        return f"{START}{m.group(0)}{END}"

    t, n_year = pat_year.subn(_wrap, t)
    t, n_date = pat_date.subn(_wrap, t)
    t, n_md   = pat_md.subn(_wrap, t)
    if any([n_year, n_date, n_md]):
        print(f"[전처리] 날짜/연도 보호: year={n_year}, date={n_date}, md={n_md}")

    # ===== 숫자+단위 보존 =====
    num_re = r"\d+(?:[,\.\u00B7]\d+)*"
    preserve_re = re.compile(
        rf"(?P<full>(?P<num>{num_re})\s*(?P<unit>{PRESERVE_UNITS}))",
        re.IGNORECASE | re.DOTALL
    )
    def _mark_keep(m: re.Match) -> str:
        return f"{START}{m.group('num')}{m.group('unit')}{END}"

    t, n_units = preserve_re.subn(_mark_keep, t)
    if n_units:
        print(f"[전처리] 숫자+단위 보호: {n_units}건")

    # --- 잡음 & 숫자 제거 ---
    t = re.sub(r'\b[a-zA-Z]\b', ' ', t)
    t = _remove_digits_outside_protection(t)

    # --- 보호 복원 ---
    t = t.replace(START, "").replace(END, "")

    # --- 허용문자 필터 + 공백 압축 ---
    t = re.sub(r"[^0-9A-Za-z가-힣\.\,\!\?…/\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # --- 최종: 공백 제거 (Spacing 단계에서 복원) ---
    t = t.replace(" ", "")

    print(f"[전처리] 전처리 후 길이: {len(t)}")
    return t

# =========================
# OCR + 전처리 (컷 → 클린 순서)
# =========================
def perform_ocr_pages(file_bytes: bytes, source: str) -> str:
    pages = []
    if file_bytes.startswith(b"%PDF"):
        print("[OCR] PDF 파일 감지")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(file_bytes)
            in_path = tf.name
        out_path = in_path.replace(".pdf", "_ocr.pdf")
        try:
            ocrmypdf.ocr(
                in_path, out_path,
                language="kor", skip_text=True, force_ocr=False,
                color_conversion_strategy="RGB", output_type="pdf",
                deskew=True, jobs=int(os.getenv("OCR_JOBS", "2")),
                quiet=True  # 로그 억제
            )
            doc = fitz.open(out_path)
            for i in range(doc.page_count):
                pages.append(doc[i].get_text("text"))
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

    # 1) 페이지 기반 앞부분 컷(원문 기준)
    raw_cut = page_based_front_cut(pages, source)

    # 파일로 저장
    os.makedirs("./processed", exist_ok=True)
    with open("./processed/ocr.txt", "w", encoding="utf-8") as f:
        f.write(raw_cut)

    # 2) 컷 이후에만 클리닝
    combined = clean_full_text(raw_cut, source)

    with open("./processed/before_spacing.txt", "w", encoding="utf-8") as f:
        f.write(combined)

    # 마지막 공백 제거 후 spacing 적용
    spaced = spacing(combined)
    print(f"[Spacing] 띄어쓰기 적용 후 길이: {len(spaced)}")

    with open("./processed/after_spacing.txt", "w", encoding="utf-8") as f:
        f.write(spaced)

    return spaced


# =========================
# 문장 분리 + 요약(추출/임베딩)
# =========================
def is_noise_line(line: str) -> bool:
    return False


def split_korean_sentences(text: str) -> list[str]:
    protected = re.sub(r'(\d)\.(\d)', r'\g<1><dot>\g<2>', text)
    try:
        raws = kss.split_sentences(protected, use_quotes_brackets_processing=False, ignore_quotes_or_brackets=True)
    except Exception:
        raws = re.split(r'(?<=[\.\!\?…])\s*', protected)
    raws = [r.replace('<dot>', '.') for r in raws]
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
# 생성 요약 유틸
# -------------------------
def _token_len(txt: str) -> int:
    return tokenizer.encode(txt, add_special_tokens=True, return_tensors="pt").shape[1]


def _normalize_periods(txt: str) -> str:
    t = re.sub(r"\s+", " ", txt).strip()
    t = re.sub(r"(?<![\.!?])\s*(다|이다|합니다|했습니다|했다|한다)\s*$", r"\g<1>.", t)
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
    lines = [x.strip() for x in cand.splitlines() if x.strip()]
    items = []
    for ln in lines:
        ln = re.sub(r'^\s*(?:\d+[\.\)]|[-•·∙ㆍ‧])\s*', '', ln).strip()
        if ln:
            items.append(ln)
    items = _dedup_keep_order(items)
    if len(items) >= n:
        return [(s if s.endswith('.') else s + '.') for s in items[:n]]

    protected = re.sub(r"(\d)\.(\d)", r"\g<1><dot>\g<2>", cand)
    parts = [p.strip().replace("<dot>", ".") for p in re.split(r"(?<=\.)\s+", protected) if p.strip()]
    parts = _dedup_keep_order(parts)
    if len(parts) >= n:
        return parts[:n]

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

    prompt = (
        f"아래 요약을 바탕으로 서로 다른 핵심 포인트 {n}가지를 한 문장씩 써라.\n"
        f"출력 형식: '1. ...' 문장형, 줄바꿈, 각 문장은 마침표로 끝낼 것. 중복 금지.\n"
        f"모든 문장은 서로 다른 정보를 담아 반드시 {n}문장을 출력할 것.\n\n"
        f"[요약]\n{cand}"
    )
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    with torch.inference_mode():
        outs = model.generate(ids, **decoding_kwargs)
    retry = tokenizer.decode(outs[0], skip_special_tokens=True)
    parts = _parse_numbered_or_periods(_normalize_periods(retry), n)
    parts = _dedup_keep_order(parts)

    if len(parts) < n:
        for s in outline:
            ss = _normalize_periods(s)
            if ss and ss not in parts:
                parts.append(ss if ss.endswith('.') else ss + '.')
            if len(parts) >= n:
                break
    return parts[:n]


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


def _build_global_outline(sent_list: list[str], max_items: int = 10) -> list[str]:
    if not sent_list:
        return []
    sents = sent_list[:4000]
    proc = [_preprocess_for_embed(s) for s in sents]
    embs = sbert.encode(proc, convert_to_numpy=True, show_progress_bar=False)

    k = min(max_items, max(3, int(len(embs) ** 0.5)))
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

def _format_summary_list(parts: list[str], n: int) -> str:
    """
    생성/파싱된 요약 문장들을 다듬어 번호 목록으로 고정.
    - 문장 앞 불필요한 구두점/접속부사 제거
    - '중 ' 같은 잘린 머리 단어 제거
    - 마침표 보장
    - '1. ...' 형식으로 재번호
    """
    cleaned = []
    for s in parts[:n]:
        s = s.strip()
        # 선행 구두점/글머리 제거
        s = re.sub(r'^[\s,;·•\-–—\.]+', '', s)
        # 자주 나오는 군더더기 제거
        s = re.sub(r'^(?:그리고|또한|하지만|그러나|이는|한편|또|이에)\s*[,，]?\s*', '', s)
        s = re.sub(r'^중\s+', '', s)
        # 기존 숫자 글머리 제거(다시 번호 붙일 거라)
        s = re.sub(r'^\d+\s*[\.\)]\s*', '', s)
        # 마침표 보장
        s = re.sub(r'\s*$', '', s)
        if not s.endswith('。') and not s.endswith('.'):
            s = s.rstrip('…') + '.'
        cleaned.append(s)

    # 중복/공백 정리
    uniq = []
    seen = set()
    for s in cleaned:
        k = re.sub(r'\s+', ' ', s)
        if k and k not in seen:
            seen.add(k); uniq.append(k)
        if len(uniq) >= n:
            break

    return "\n".join(f"{i+1}. {uniq[i]}" for i in range(len(uniq)))

def _summarize_stuff_with_outline(base_text: str, outline: list[str], lines: int, decoding_kwargs: dict) -> str:
    outline_text = "\n".join(f"- {s}" for s in outline[:12])
    prompt = (
        f"다음 텍스트를 한국어로 서로 다른 {lines}개의 핵심 문장으로 요약하라.\n"
        f"각 문장은 15~40자, 정보 중복 금지. 전역 윤곽을 우선 반영.\n"
        f"출력 형식: '1. ...' 문장형, 줄바꿈, 마침표로 끝낼 것.\n"
        f"모든 문장은 서로 다른 핵심 정보를 담아야 하며, 반드시 {lines}문장을 모두 작성할 것.\n\n"
        f"[전역 윤곽]\n{outline_text}\n\n[본문]\n{base_text}"
    )
    ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    cand = _decode_generate(ids, decoding_kwargs)
    parts = _ensure_n_distinct_sentences(cand, lines, decoding_kwargs, outline)
    return _format_summary_list(parts, lines)  # ← 여기!


def _summarize_refine_with_outline(full_text: str, outline: list[str], lines: int) -> str:
    total_tokens = _token_len(full_text)
    CHUNK_TOKENS, OVERLAP = _adaptive_chunk_params(total_tokens)
    decoding_kwargs = _decoding_cfg()

    all_tokens = tokenizer.encode(full_text, add_special_tokens=True)
    chunks = list(_chunk_tokens(all_tokens, CHUNK_TOKENS, overlap=OVERLAP))
    total_chunks = len(chunks)

    first_text = tokenizer.decode(chunks[0], skip_special_tokens=True)
    summary = _summarize_stuff_with_outline(first_text, outline, lines, decoding_kwargs)

    outline_text = "\n".join(f"- {s}" for s in outline[:12])
    for i, tok_chunk in enumerate(chunks[1:], start=2):
        sys.stdout.write(f"\r[Refine 진행률] {i}/{total_chunks} 청크 처리 중..."); sys.stdout.flush()
        chunk_text = tokenizer.decode(tok_chunk, skip_special_tokens=True)
        c_ids = tokenizer.encode(chunk_text, add_special_tokens=False)
        if len(c_ids) > 640:
            c_ids = c_ids[:640]
            chunk_text = tokenizer.decode(c_ids, skip_special_tokens=True)

        prompt = (
            f"전역 윤곽을 유지하며 현재 요약을 새로운 컨텍스트로 보강하라.\n"
            f"결과는 서로 다른 {lines}문장, 각 15~40자, 중복 금지.\n"
            f"출력 형식: '1. ...' 문장형, 줄바꿈, 마침표로 끝낼 것.\n\n"
            f"[전역 윤곽]\n{outline_text}\n\n[현재 요약]\n{summary}\n\n[새 컨텍스트]\n{chunk_text}"
        )
        ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
        cand = _decode_generate(ids, decoding_kwargs)
        parts = _ensure_n_distinct_sentences(cand, lines, decoding_kwargs, outline)
        summary = _format_summary_list(parts, lines)  # ← 여기!
    sys.stdout.write("\n"); sys.stdout.flush()
    return summary

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


def generate_summary(sent_list: list[str], lines: int = 3) -> str:
    if not sent_list:
        return ""

    base_text_full = re.sub(r"\s+", " ", " ".join(sent_list)).strip()
    in_len_full = _token_len(base_text_full)

    ULTRA_LONG_EMERGENCY = int(os.getenv("ULTRA_LONG_EMERGENCY_TOKENS", "40000"))
    if in_len_full > ULTRA_LONG_EMERGENCY:
        print(f"[경고] 입력 {in_len_full} 토큰: 비상 축약 수행")
        budget = int(os.getenv("PRECOMPRESS_TO_TOKENS", "8000"))
        reduced = _precompress_ultra_long(sent_list, budget)
        base_text = re.sub(r"\s+", " ", " ".join(reduced)).strip()
    else:
        base_text = base_text_full

    outline = _build_global_outline(sent_list, max_items=int(os.getenv("OUTLINE_ITEMS", "10")))

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
    spaced = perform_ocr_pages(data, source)  # 컷 → 클린 → spacing
    sents = split_korean_sentences(spaced)
    with open("./example.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sents))

    # 운영옵션: EXTRACTIVE_DEV_MODE=1일 때만 추출요약 저장/노출
    DEV_MODE = os.getenv("EXTRACTIVE_DEV_MODE", "0").lower() in ("1", "true", "yes")
    extractive = extractive_summary(sents, extractive_sentences) if DEV_MODE else ""

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
        return {
            "doc_id": doc.id,
            **({"extractive_summary": extractive} if DEV_MODE else {}),
            "generation_started": do_generation
        }
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
