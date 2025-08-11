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
from pytesseract import Output

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from sqlalchemy import create_engine, Column, Integer, Text, DateTime, Enum as SqlEnum
from sqlalchemy.orm import declarative_base, sessionmaker

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import ocrmypdf

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score

import kss
from pykospacing import Spacing


# =========================
# 전역 설정 / 초기화
# =========================
# 모델/토크나이저
tokenizer = AutoTokenizer.from_pretrained("eenzeenee/t5-small-korean-summarization")
model = AutoModelForSeq2SeqLM.from_pretrained("eenzeenee/t5-small-korean-summarization")
# CPU-only 성능: Linear만 dynamic int8 양자화
try:
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
except Exception as e:
    print("[warn] dynamic quantization skipped:", e)
model.eval()

# SBERT 임베딩
sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# 띄어쓰기 복원
spacing = Spacing()

# DB
DB_PATH = os.getenv("DB_PATH", "./ocr.db")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

doc_semaphore = threading.Semaphore(MAX_WORKERS)
app = FastAPI()


# =========================
# Enum / ORM
# =========================
class StatusEnum(str, Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    DONE = "DONE"


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(Text, nullable=False)  # 자유 텍스트로 저장
    ocr_text = Column(Text, nullable=False)
    extractive_summary = Column(Text, nullable=False)
    abstractive_summary = Column(Text, nullable=True)
    status = Column(SqlEnum(StatusEnum), nullable=False, default=StatusEnum.PENDING)


Base.metadata.create_all(bind=engine)


# =========================
# 레이아웃 인지 유틸 (단수 자동 판별 + 재흐름 + 본문/비본문)
# =========================
_END_PUNCT = set(".!?…」』）》〉)]”’\"")
_HYPHEN = "-"

def _estimate_line_tol(word_tuples):
    hs = [ (y1 - y0) for (x0,y0,x1,y1,_) in word_tuples ]
    if not hs:
        return float(os.getenv("LINE_Y_TOL", "4.0"))
    med = float(np.median(hs))
    return max(3.0, min(12.0, float(os.getenv("LINE_Y_TOL_FACTOR", "0.7")) * med))

def _choose_k_by_silhouette(x_centers, page_w, max_k=4):
    X = np.array(x_centers, dtype=np.float32).reshape(-1, 1)
    n = len(X)
    if n < 30:
        return 1, np.zeros(n, dtype=int), [float(X.mean())]

    best_k, best_lbls, best_centers, best_score = 1, np.zeros(n, dtype=int), [float(X.mean())], -1
    min_sep_ratio = float(os.getenv("COLUMN_SEP_RATIO", "0.12"))
    min_sil = float(os.getenv("SILHOUETTE_MIN", "0.15"))
    max_k = int(os.getenv("COLUMN_MAX_K", str(max_k)))
    max_k = max(1, min(max_k, 6))

    idx = np.arange(n)
    if n > 400:
        sample_idx = np.random.RandomState(0).choice(idx, 400, replace=False)
        Xs = X[sample_idx]
    else:
        sample_idx = idx
        Xs = X

    for k in range(1, max_k+1):
        if k == 1:
            km = KMeans(n_clusters=1, n_init=10, random_state=0).fit(Xs)
            centers = sorted(km.cluster_centers_.flatten().tolist())
            labels_full = np.zeros(n, dtype=int)
            score = -1
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
            centers = sorted(km.cluster_centers_.flatten().tolist())
            diffs = np.diff(sorted(centers))
            sep_ratio = float(np.min(diffs)) / max(1.0, float(page_w)) if len(diffs) else 0.0
            s = silhouette_score(Xs, km.labels_) if len(set(km.labels_)) > 1 else -1.0
            if sep_ratio < min_sep_ratio or s < min_sil:
                continue
            score = s
            labels_full = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X).labels_

        if score > best_score:
            best_k, best_lbls, best_centers, best_score = k, labels_full, centers, score

    counts = np.bincount(best_lbls, minlength=best_k)
    if best_k > 1:
        ratios = counts / counts.sum()
        if (ratios < float(os.getenv("MIN_CLUSTER_RATIO", "0.15"))).any():
            # 작은 클러스터를 이웃으로 병합
            centers_arr = np.array(best_centers)
            for sidx, r in enumerate(ratios):
                if r >= float(os.getenv("MIN_CLUSTER_RATIO", "0.15")):
                    continue
                dists = np.abs(centers_arr - centers_arr[sidx])
                dists[sidx] = np.inf
                tgt = int(np.argmin(dists))
                best_lbls[best_lbls==sidx] = tgt
            uniq = sorted(set(best_lbls))
            remap = {u:i for i,u in enumerate(uniq)}
            best_lbls = np.array([remap[v] for v in best_lbls], dtype=int)
            best_k = len(uniq)
            best_centers = [ float(np.mean([x_centers[i] for i in np.where(best_lbls==c)[0]])) for c in range(best_k) ]

    order = np.argsort(best_centers)
    remap = { old:i for i, old in enumerate(order) }
    best_lbls = np.array([ remap[l] for l in best_lbls ], dtype=int)
    best_centers = [ best_centers[o] for o in order ]
    return best_k, best_lbls, best_centers

def _sort_words_reading_order(words):
    return sorted(words, key=lambda w: (round(w[1], 1), w[0]))

def _merge_lines(words, y_tol):
    if not words:
        return ""
    lines, cur, prev_y = [], [], None
    for w in _sort_words_reading_order(words):
        x0,y0,x1,y1,txt = w
        t = str(txt).strip()
        if not t:
            continue
        if prev_y is None or abs(y0 - prev_y) <= y_tol:
            cur.append((x0, t))
        else:
            if cur:
                lines.append(cur)
            cur = [(x0, t)]
        prev_y = y0
    if cur:
        lines.append(cur)

    def join_line(line):
        line = sorted(line, key=lambda t: t[0])
        return " ".join([t for _, t in line])

    raw_lines = [ join_line(ln).strip() for ln in lines if ln ]
    merged = []
    for ln in raw_lines:
        if not merged:
            merged.append(ln); continue
        prev = merged[-1]
        if prev.endswith(_HYPHEN):
            merged[-1] = prev[:-1] + ln.lstrip()
            continue
        if prev and prev[-1] not in _END_PUNCT:
            if re.match(r"^(\d+\.|\*+|-+|•|\([가-힣A-Za-z0-9]\))\s+", ln):
                merged.append(ln)
            else:
                merged[-1] = prev + " " + ln.lstrip()
        else:
            merged.append(ln)
    return "\n".join(merged)

def reflow_adaptive_cols(word_tuples, page_w):
    if not word_tuples:
        return ""
    x_centers = [ (x0+x1)/2.0 for (x0,y0,x1,y1,_) in word_tuples ]
    k, labels, centers = _choose_k_by_silhouette(x_centers, page_w, max_k=4)
    y_tol = _estimate_line_tol(word_tuples)

    col_texts = []
    for c in range(k):
        col_words = [ w for w,l in zip(word_tuples, labels) if l==c ]
        col_words = _sort_words_reading_order(col_words)
        col_texts.append(_merge_lines(col_words, y_tol))
    return "\n".join([t for t in col_texts if t])

def classify_page_simple(word_tuples, page_w, page_h, page_text=""):
    n_words = len(word_tuples)
    if n_words < int(os.getenv("NONBODY_MIN_WORDS", "20")):
        return "nonbody"

    area_sum = 0.0
    for (x0,y0,x1,y1,_) in word_tuples:
        area_sum += max(0.0, (x1-x0)) * max(0.0, (y1-y0))
    coverage = area_sum / max(1.0, page_w*page_h)

    if coverage < float(os.getenv("NONBODY_COVERAGE", "0.01")) and n_words < 80:
        return "nonbody"

    if page_text and re.search(r"(목차|차례|참고문헌|참고자료|부록|Contents|Index|References)", page_text):
        return "nonbody"

    return "body"


# =========================
# OCR + 클린업 + 스페이싱
# =========================
def perform_ocr_pages(file_bytes: bytes):
    """
    - PDF: ocrmypdf → (단어좌표) reflow_adaptive_cols → 텍스트
    - 이미지: tesseract image_to_data → reflow_adaptive_cols → 텍스트
    - 그 뒤 URL/이메일/전화번호/긴 숫자 정리 → 특문 필터 → pykospacing
    """
    pages = []

    if file_bytes.startswith(b"%PDF"):
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
                deskew=True, remove_background=True, jobs=int(os.getenv("OCR_JOBS", "2"))
            )
            doc = fitz.open(out_path)
            for i in range(doc.page_count):
                page = doc[i]
                page_w, page_h = page.rect.width, page.rect.height
                wds = page.get_text("words")  # x0,y0,x1,y1,"word", block, line, word
                if not wds:
                    pages.append(page.get_text("text"))
                    continue
                words_simple = [(x0,y0,x1,y1,w) for (x0,y0,x1,y1,w,*_) in wds if str(w).strip()]
                rough_text = page.get_text("text")
                _page_type = classify_page_simple(words_simple, page_w, page_h, rough_text)
                flowed = reflow_adaptive_cols(words_simple, page_w)
                pages.append(flowed if flowed.strip() else rough_text)
            doc.close()
        finally:
            for p in (in_path, out_path):
                try: os.remove(p)
                except: pass

    else:
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise HTTPException(400, "지원되지 않는 파일 형식")
        print("[3] OCR 처리 중: 이미지 파일")

        df = pytesseract.image_to_data(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            lang="kor",
            config="--psm 1",
            output_type=Output.DATAFRAME
        )
        df = df.dropna(subset=["text"])
        min_conf = float(os.getenv("OCR_MIN_CONF", "0"))
        df = df[df.conf.astype(float) >= min_conf]

        page_w, page_h = img.shape[1], img.shape[0]
        word_tuples = [
            (int(x), int(y), int(x)+int(w), int(y)+int(h), str(t))
            for x,y,w,h,t in zip(df.left, df.top, df.width, df.height, df.text)
            if str(t).strip()
        ]
        flowed = reflow_adaptive_cols(word_tuples, page_w)
        pages.append(flowed)

    # 1) 병합
    combined = "\n".join(pages)

    # 2) URL 제거
    combined = re.sub(r'(?:https?://|www\.)[A-Za-z0-9\-._~:/?#\[\]@!$&\'()*+,;=%]+', ' ', combined)
    # 3) 이메일 제거
    combined = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', combined)
    # 4) 전화번호/숫자 제거
    combined = re.sub(r'(?i)(?:TEL|FAX)[\s\-\.:]*\+?82?[-.\s]?\d{1,2}(?:[-.\s]?\d{3,4}){2}', ' ', combined)
    combined = re.sub(r'(?<!\d)(?:\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4})(?!\d)', ' ', combined)
    combined = re.sub(r'(?<!\d)\d{9,11}(?!\d)', ' ', combined)
    combined = re.sub(r'\d{12,}', ' ', combined)

    # 5) 공백/특수문자 정리
    combined = re.sub(r'[\r\t]+', ' ', combined)
    combined = re.sub(r'[^0-9A-Za-z가-힣\.\,\!\?\s]', ' ', combined)
    combined = re.sub(r'\s+', ' ', combined).strip()

    os.makedirs("./processed", exist_ok=True)
    with open("./processed/before_spacing.txt", "w", encoding="utf-8") as f:
        f.write(combined)

    # 6) 띄어쓰기 복원
    print("[공백 복원 시작]")
    spaced = spacing(combined)
    print("[공백 복원 완료]")

    with open("./processed/after_spacing.txt", "w", encoding="utf-8") as f:
        f.write(spaced)

    return spaced


# =========================
# 문장 분리(KSS) + 노이즈 필터
# =========================
def is_noise_line(line: str) -> bool:
    t = line.strip()
    if len(t) < 10:
        return True
    if re.fullmatch(r"\d+(?:\s*\d+)*", t):
        return True
    if re.search(r"\b(vol\.?\s*\d+|TEL|FAX|E[- ]?mail|월호|목차)\b", t, flags=re.IGNORECASE):
        return True
    return False

def split_korean_sentences(text: str) -> list[str]:
    MAX_CHARS = 8000
    chunks = [text[i:i+MAX_CHARS] for i in range(0, len(text), MAX_CHARS)] or [""]

    raws = []
    for ch in chunks:
        try:
            parts = kss.split_sentences(
                ch,
                use_quotes_brackets_processing=False,
                ignore_quotes_or_brackets=True
            )
        except Exception:
            parts = re.split(r'(?<=[\.!\?])\s*', ch)
        raws.extend(parts)

    results = []
    for raw in raws:
        sent = raw.strip()
        if not sent:
            continue
        if is_noise_line(sent):
            continue
        results.append(sent)
    return results


# =========================
# 요약 파이프라인
# =========================
def preprocess_text(text: str) -> str:
    txt = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    txt = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', ' ', txt)
    txt = re.sub(r"[\r\n\t]+", ' ', txt)
    txt = re.sub(r"[^0-9A-Za-z가-힣\.\?!\s]", ' ', txt)
    txt = re.sub(r"\s+", ' ', txt).strip()
    tokens = re.findall(r"[0-9A-Za-z가-힣]{2,}", txt)
    return " ".join(tokens)

def extractive_summary(sent_list: list[str], num_sentences: int = 3) -> str:
    if not sent_list:
        return ""
    proc = [preprocess_text(s) for s in sent_list]
    embs = sbert.encode(proc, convert_to_numpy=True, show_progress_bar=False)
    n = min(num_sentences, len(embs))
    clust = AgglomerativeClustering(n_clusters=n, metric="cosine", linkage="average")
    labels = clust.fit_predict(embs)
    summary = []
    for lbl in sorted(set(labels)):
        idxs = np.where(labels == lbl)[0]
        cent = embs[idxs].mean(axis=0)
        sims = cosine_similarity([cent], embs[idxs])[0]
        best = idxs[sims.argmax()]
        summary.append((best, sent_list[best]))
    summary.sort(key=lambda x: x[0])
    return "\n".join(f"[{i+1}] {s}" for i, s in summary)

def _chunk_tokens(tokens: list[int], chunk_size: int, overlap: int = 32):
    if chunk_size <= 0:
        yield tokens; return
    i, n = 0, len(tokens)
    while i < n:
        j = min(i + chunk_size, n)
        yield tokens[i:j]
        if j >= n: break
        i = max(0, j - overlap)

def generate_summary(text: str) -> str:
    clean = re.sub(r'\s+', ' ', text).strip()
    base_prompt = "summarize: "
    DECODING = dict(
        max_new_tokens=int(os.getenv("T5_MAX_NEW_TOKENS", "80")),
        num_beams=1,
        do_sample=False,
        use_cache=True,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    input_ids = tokenizer.encode(base_prompt + clean, return_tensors="pt", add_special_tokens=True)
    if input_ids.shape[1] <= 450:
        with torch.no_grad():
            outs = model.generate(input_ids, **DECODING)
        return tokenizer.decode(outs[0], skip_special_tokens=True)

    carry = ""
    carry_max_tokens = 120
    all_tokens = tokenizer.encode(clean, add_special_tokens=True)
    C_SIZE = 320
    out_text = None

    for tok_chunk in _chunk_tokens(all_tokens, C_SIZE, overlap=32):
        chunk_text = tokenizer.decode(tok_chunk, skip_special_tokens=True)
        chunk_prompt = base_prompt + (carry + " " if carry else "") + chunk_text
        ids = tokenizer.encode(chunk_prompt, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            outs = model.generate(ids, **DECODING)
        out_text = tokenizer.decode(outs[0], skip_special_tokens=True)

        carry_ids = tokenizer.encode(out_text, add_special_tokens=False)
        if len(carry_ids) > carry_max_tokens:
            carry_ids = carry_ids[:carry_max_tokens]
        carry = tokenizer.decode(carry_ids, skip_special_tokens=True)

    return carry if carry else (out_text or "")


# =========================
# 백그라운드 생성 태스크
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
        doc.abstractive_summary = generate_summary(doc.ocr_text)
        doc.status = StatusEnum.DONE
        db.commit()
    finally:
        doc_semaphore.release()
        db.close()


# =========================
# API
# =========================
@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    source: str = Form(...),
    file: UploadFile = File(...),
    extractive_sentences: int = Form(3),
    do_generation: bool = Form(False)
):
    print(f"[1] 요청 수신: source={source}")
    data = await file.read()
    print(f"[2] 파일 수신 완료 ({len(data)} bytes)")
    spaced = perform_ocr_pages(data)
    sents = split_korean_sentences(spaced)
    os.makedirs(".", exist_ok=True)
    with open("example.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(sents))
    print(f"[4] example.txt에 {len(sents)}개 문장 저장됨")

    extractive = extractive_summary(sents, extractive_sentences)
    print("[6] 추출적 요약 완료")

    db = SessionLocal()
    try:
        doc = Document(
            source=source,
            ocr_text="\n".join(sents),
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
        print(f"[10] 응답 전송: id={doc.id}")
        return {
            "doc_id": doc.id,
            "extractive_summary": extractive,
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
        return {
            "doc_id": doc.id,
            "ocr_text": doc.ocr_text,
            "extractive_summary": doc.extractive_summary,
            "abstractive_summary": doc.abstractive_summary,
            "status": doc.status.value
        }
    finally:
        db.close()

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("service:app", host="0.0.0.0", port=8000, reload=True)
