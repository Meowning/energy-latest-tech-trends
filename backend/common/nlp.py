# backend/common/nlp.py
import os
import re
import sys
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

import kss


def _offline() -> bool:
    v = (os.getenv("TRANSFORMERS_OFFLINE", "0"), os.getenv("HF_HUB_OFFLINE", "0"))
    return any(str(x).lower() in ("1", "true", "yes") for x in v)


def _pick_device(env_key: str, default_auto: bool = True) -> str:
    want = os.getenv(env_key, "").strip().lower()
    if want in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return "cuda"
        print(f"[warn] {env_key}=cuda 요청됐지만 CUDA 미사용 -> cpu로 폴백")
        return "cpu"
    if want == "mps":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        print(f"[warn] {env_key}=mps 요청됐지만 MPS 미사용 -> cpu로 폴백")
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
            print("[info] T5 동적 양자화 적용됨 (CPU)")
        except Exception as e:
            print("[warn] T5 동적 양자화 스킵:", e)

    mdl.eval().to(device)
    print(f"[info] T5 로딩 완료: {device.upper()} (repo={repo})")
    return tok, mdl


tokenizer, model = load_t5()

try:
    if next(model.parameters()).device.type == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 2))
        torch.set_num_interop_threads(1)
        print(f"[info] torch 스레드 세팅 : {torch.get_num_threads()} / interop 1")
except Exception:
    pass


def load_sbert():
    repo = os.getenv("SBERT_REMOTE_ID", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    off = _offline()
    device = _pick_device("SBERT_DEVICE")
    model = SentenceTransformer(repo, device=device, cache_folder=None)
    print(f"[info] SBERT 로딩 완료: {device.upper()} (repo={repo})")
    return model


sbert = load_sbert()


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

# 추출요약 (완성도+정확도 이상함 이슈로 실제 파이프라인에선 안쓸거임 ㅠㅠ)
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
    생성/파싱된 요약 문장들을 다듬어 번호 목록으로 고정
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
    return _format_summary_list(parts, lines)


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
        summary = _format_summary_list(parts, lines)
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
