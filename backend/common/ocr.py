# backend/common/ocr.py
import os
import re
import tempfile
import unicodedata
from math import ceil

import numpy as np
import cv2
import fitz  # PyMuPDF
import pytesseract
import ocrmypdf

from pykospacing import Spacing

# 숫자 보존 대상
PRESERVE_UNITS = (
    # 일반
    r"(건|개|명|가구|년|년도|연도|월|일|시|분|초|차|회|분기|반기|상반기|하반기|%)"

    # 온도
    r"|(?:°C|℃|°F|℉|K|degC|degF)"

    # 길이
    r"|(?:nm|μm|um|mm|cm|m|km|Å)"

    # 면적
    r"|(?:mm2|cm2|m2|km2|mm²|cm²|m²|km²|㎡|ha)"

    # 부피
    r"|(?:mm3|cm3|m3|km3|mm³|cm³|m³|km³|㎥|L|mL|μL|uL|kL|ML|GL|ℓ)"

    # 질량
    r"|(?:mg|g|kg|t|kt|Mt|Gt)"

    # 농도/밀도/미세먼지
    r"|(?:ppm|ppb|ppt)"
    r"|(?:μg|ug|mg|ng|g)/(?:L|m3|m\^?3|㎥|kg|m2|m\^?2|㎡)"
    r"|(?:PM(?:10|2\.5))"

    # 에너지/전력/전력량
    r"|(?:J|kJ|MJ|GJ|TJ|PJ)"
    r"|(?:W|kW|MW|GW|TW)"
    r"|(?:Wh|kWh|MWh|GWh|TWh|PWh)"
    r"|(?:kcal|Cal)"
    
    # 연료
    r"|(?:toe|ktoe|Mtoe|Gtoe|boe|Mboe)"
    
    # 배출량/강도
    r"|(?:tCO2e|tCO2eq|CO2e|CO2eq|kgCO2e|ktCO2e|MtCO2e)"
    r"|(?:gCO2(?:/kWh|/MJ)|kgCO2(?:/kWh|/MWh))"
    
    # 방사선
    r"|(?:(?:nSv|μSv|uSv|mSv|Sv)(?:/(?:h|hr|d|day|yr|y|a)|·h(?:\^?-?1|⁻¹)|/시간|/년)?)"
    r"|(?:(?:nGy|μGy|uGy|mGy|Gy)(?:/(?:h|hr|d|day|yr|y|a)|·h(?:\^?-?1|⁻¹)|/시간|/년)?)"
    r"|(?:(?:μrem|urem|mrem|rem)(?:/(?:h|hr|d|day|yr|y|a))?)"
    r"|(?:[kMGT]?Bq(?:/(?:m3|m\^?3|㎥|L|kg|m2|m\^?2|㎡))?)"
    r"|(?:Ci|mCi|μCi|uCi|kCi)"
    r"|(?:cpm|cps|dpm)"
    
    # 압력
    r"|(?:Pa|kPa|MPa|GPa|bar|mbar|hPa|atm|Torr|mmHg)"
    
    # 유량/유속/회전수
    r"|(?:m3/s|m3/h|Nm3/h|Sm3/h|L/s|L/min|L/h|mL/min)"
    r"|(?:m/s|km/h|rpm|rps)"
    
    # 주파수
    r"|(?:Hz|kHz|MHz|GHz|THz|dB)"
    
    # 전기
    r"|(?:V|kV|A|mA|μA|uA)"
    r"|(?:Ω|ohm|kΩ|MΩ)"
    r"|(?:S|mS|μS|uS)"
    r"|(?:F|mF|μF|uF|nF)"
    r"|(?:H|mH|μH|uH)"
    
    # 각도/휘도/탁도
    r"|(?:rad|mrad|sr)"
    r"|(?:lx|lm|cd|nit|nt)"
    r"|(?:NTU)"
    
    # 통화
    r"|(?:원|천원|만원|백만원|천만원|억원|십억원|백억원|천억원|조원|십조원|백조원|천조원)"
)


# 본문 직전 단어? 문장? 암튼 본문 가르는 기준
DEFAULT_CLEAN_KEYWORDS = (
    r"(FAX|Mail|메일|E[- ]?mail|이메일|주소|목차|"
    r"제\s*\d{4}\s*\d{1,2}\s*\d{4}\.(?:0[1-9]|1[0-2])\.(?:0[1-9]|[12]\d|3[01])\.?)"
)

# 소스별로 기준
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
    pdf 소스별 맞춤 정규식 반환
    - 매핑된 정규식이 없거나 빈 문자열이면 DEFAULT_CLEAN_KEYWORDS 사용
    - IGNORECASE + MULTILINE로 컴파일
    """
    key = (source or "").strip()
    pattern = CLEAN_KEYWORDS_BY_SOURCE.get(key) or DEFAULT_CLEAN_KEYWORDS
    if pattern not in _COMPILED_CLEAN_REGEX:
        _COMPILED_CLEAN_REGEX[pattern] = re.compile(
            pattern, flags=re.IGNORECASE | re.MULTILINE
        )
    return _COMPILED_CLEAN_REGEX[pattern]

# 이미지 기반 OCR 대비 아스키/유니코드 정규화
def normalize_units_for_ascii(t: str) -> str:
    """
    이미지 기반 OCR 대비 아스키/유니코드 정규화

    """
    repl = {
        "㎡": "m2","㎥":"m3","㎤":"cm3","㎣":"mm3","㎦":"km3",
        "㎟":"mm2","㎠":"cm2","㎢":"km2",
        "㎜":"mm","㎝":"cm","㎞":"km",
        "㎍":"ug","㎎":"mg","㎏":"kg",
        "ℓ":"L",
        "℃":"degC","℉":"degF",
        "Ω":"ohm",
        "㎾":"kW","㎿":"MW",
        "㎸":"kV","㎹":"MV","㎶":"mV",
        "㎄":"kA","㎃":"mA","㎂":"uA",
        "㎌":"uF",
        "㎩":"kPa","㎫":"MPa","㎬":"GPa","㎭":"rad",
        "㎧":"m/s","㎨":"m/s2",
        "㎖":"mL","㎕":"uL","㎘":"kL",
        "㎐":"Hz","㎑":"kHz","㎒":"MHz","㎓":"GHz",
        "㏈":"dB","㏃":"Bq","㏅":"cd","㏗":"pH","㏄":"mL",
        "％":"%","／":"/","－":"-","–":"-","—":"-",
        "µ":"u","μ":"u","²":"2","³":"3",
    }
    for k, v in repl.items(): t = t.replace(k, v)
    t = re.sub(r"\u00B7\s*h(?:\^-?1|⁻1|⁻¹)", r"/h", t)
    t = re.sub(r"\u00B7\s*s(?:\^-?1|⁻1|⁻¹)", r"/s", t)
    t = re.sub(r"\u00B7\s*(?:yr|y|a)(?:\^-?1|⁻1|⁻¹)", r"/yr", t)
    t = re.sub(r"\u00B7\s*h-?1", r"/h", t)
    t = re.sub(r"\u00B7\s*s-?1", r"/s", t)
    t = re.sub(r"([A-Za-z])\s*2\b", r"\g<1>2", t)
    t = re.sub(r"([A-Za-z])\s*3\b", r"\g<1>3", t)
    return t


# 페이지 기반으로 본문컷
def page_based_front_cut(pages: list[str], source: str) -> str:
    """
    페이지 기반으로 5% 이내에 키워드가 있으면, 마지막 매칭 위치 기준으로 앞부분 컷
    5% 이내에 없으면 맨 처음으로 나오는 키워드 위치에서 앞부분 컷
    이상한 공백들을 다 일반 공백으로 바꾸는 작업
    """
    if not pages: return ""
    head_end = min(max(1, ceil(len(pages) * 0.05)), len(pages))
    ck = get_clean_keywords(source)
    full_text = "".join(pages); head_text = "".join(pages[:head_end])
    def norm(t: str) -> str:
        t = unicodedata.normalize("NFKC", t)
        t = re.sub(r"[\u2000-\u200b\u202f\u205f\u3000]", " ", t)
        return t
    head_norm = norm(head_text); full_norm = norm(full_text)
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
    print(f"[앞부분컷] ... → 매치 없음(스킵)")
    return full_text

# 아까 보호 씌운거 벗기는거
def _remove_digits_outside_protection(t: str) -> str:
    START, END = "\u241E", "\u241F"
    out = []; i = 0; n = len(t)
    while i < n:
        s = t.find(START, i)
        if s == -1:
            out.append(re.sub(r"\d+", " ", t[i:])); break
        out.append(re.sub(r"\d+", " ", t[i:s]))
        e = t.find(END, s + 1)
        if e == -1:
            out.append(re.sub(r"\d+", " ", t[s:])); break
        out.append(t[s:e+1]); i = e + 1
    return "".join(out)

# 전처리
def clean_full_text(text: str, source: str) -> str:
    """
    - 주석, 표, 사진, 그림 등 필요없는 정보 제거
    - 숫자 보존 대상(단위, 년도 등) 보호 후, 이외 숫자 제거
    - 공백 연속으로 여러개 있으면 일반 공백으로 바꿈
    """
    print(f"[전처리] 원본 텍스트 길이: {len(text)}")
    t = text.replace('\r','\n').replace('\u00a0',' ').replace('\u200b',' ')
    t = normalize_units_for_ascii(t)
    t = re.sub(r"(?im)^\s*[\(\[]?\s*(?:표|그림|사진)\s*\d+(?:[.\-–]\d+)*\s*[\)\]]?(?:\s*[.:]\s*)?.*$","", t)
    t = re.sub(r"[\(\[]?\s*(?:표|그림|사진)\s*\d+(?:[.\-–]\d+)*\s*[\)\]]?","", t)
    t = re.sub(r"\b(?:표|그림|사진)\b", "", t, flags=re.IGNORECASE)

    START, END = "\u241E", "\u241F"
    pat_year = re.compile(r"(\d{4})\s*(년|년도|연도)")
    pat_date = re.compile(r"(\d{4})[./-](\d{1,2})(?:[./-](\d{1,2}))?")
    pat_md   = re.compile(r"(\d{1,2})\s*(월|일)")
    def _wrap(m: re.Match) -> str: return f"{START}{m.group(0)}{END}"
    t, n_year = pat_year.subn(_wrap, t)
    t, n_date = pat_date.subn(_wrap, t)
    t, n_md   = pat_md.subn(_wrap, t)
    if any([n_year, n_date, n_md]):
        print(f"[전처리] 날짜/연도 보호: year={n_year}, date={n_date}, md={n_md}")

    num_re = r"\d+(?:[,\.\u00B7]\d+)*"
    preserve_re = re.compile(rf"(?P<full>(?P<num>{num_re})\s*(?P<unit>{PRESERVE_UNITS}))", re.IGNORECASE | re.DOTALL)
    def _mark_keep(m: re.Match) -> str: return f"{START}{m.group('num')}{m.group('unit')}{END}"
    t, n_units = preserve_re.subn(_mark_keep, t)
    if n_units: print(f"[전처리] 숫자+단위 보호: {n_units}건")

    t = _remove_digits_outside_protection(t)
    t = t.replace(START, "").replace(END, "")
    t = re.sub(r"[^0-9A-Za-z가-힣\.\,\!\?…/\-\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace(" ", "")
    print(f"[전처리] 전처리 후 길이: {len(t)}")
    return t

# 재스페이싱 모델
spacing = Spacing()

# OCR 로직
def perform_ocr_pages(file_bytes: bytes, source: str) -> str:
    """
    OCR 로직
    파일 OCR -> 앞부분 뒷부분컷 -> 일부 숫자 보존하는 전처리 후 공백제거
    -> 재스페이싱 딥러닝 모델을 사용하여 다시 공백 들어간 결과 얻음
    """
    pages = []
    if file_bytes.startswith(b"%PDF"):
        print("[OCR] PDF 파일 감지")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(file_bytes); in_path = tf.name
        out_path = in_path.replace(".pdf", "_ocr.pdf")
        try:
            ocrmypdf.ocr(
                in_path, out_path,
                language="kor", skip_text=True, force_ocr=False,
                color_conversion_strategy="RGB", output_type="pdf",
                deskew=True, jobs=int(os.getenv("OCR_JOBS", "2")),
            )
            doc = fitz.open(out_path)
            for i in range(doc.page_count):
                pages.append(doc[i].get_text("text"))
            doc.close()
        finally:
            for p in (in_path, out_path):
                try: os.remove(p)
                except: pass
    else:
        print("[OCR] 이미지 파일 감지")
        arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None or img.size == 0:
            raise ValueError("지원되지 않는 파일 형식")
        ocr_text = pytesseract.image_to_string(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), lang="kor")
        print(f"[OCR] 이미지 OCR 결과 길이: {len(ocr_text)}")
        pages.append(ocr_text)

    raw_cut = page_based_front_cut(pages, source)

    os.makedirs("./processed", exist_ok=True)
    with open("./processed/ocr.txt","w",encoding="utf-8") as f: f.write(raw_cut)

    combined = clean_full_text(raw_cut, source)
    with open("./processed/before_spacing.txt","w",encoding="utf-8") as f: f.write(combined)

    spaced = spacing(combined)
    print(f"[Spacing] 띄어쓰기 적용 후 길이: {len(spaced)}")
    with open("./processed/after_spacing.txt","w",encoding="utf-8") as f: f.write(spaced)
    return spaced
