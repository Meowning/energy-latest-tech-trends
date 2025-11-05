# backend/scripts/crawl_kofons_overseas.py
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.common.models import Base, Publication, PubStatus

# === 1. DB 연결 ===
DB_URL = "sqlite:///data.db"  # 필요 시 postgresql 등으로 교체
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(bind=engine)
session = SessionLocal()

# === 2. KOFONS 해외원자력소식 목록 페이지 ===
LIST_URL = "https://www.kofons.or.kr/web/cop/bbs/NewsList.do?bbsId=BBSMSTR_000000000412"

# === 3. 요청 ===
res = requests.get(LIST_URL)
res.raise_for_status()

soup = BeautifulSoup(res.text, "html.parser")

# === 4. 게시물 목록 파싱 ===
rows = soup.select("table.tbType1.sub4_01 tbody tr")

for row in rows:
    cols = row.select("td")
    if len(cols) < 3:
        continue

    title_tag = cols[1].select_one("a")
    if not title_tag:
        continue

    title = title_tag.get_text(strip=True)
    onclick = title_tag.get("onclick", "")
    # 예: fn_egov_inqire_notice('BBSMSTR_000000000412', '17368');
    match = re.search(r"'BBSMSTR_\d+',\s*'(\d+)'", onclick)
    post_id = match.group(1) if match else None

    date_str = cols[2].get_text(strip=True)
    published_at = None
    try:
        published_at = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        pass

    # 상세 URL 구성
    if post_id:
        detail_url = f"https://www.kofons.or.kr/web/cop/bbs/selectBoardArticle.do?bbsId=BBSMSTR_000000000412&nttId={post_id}"
    else:
        detail_url = None

    # === 5. 중복 확인 ===
    exists = session.query(Publication).filter_by(url=detail_url).first()
    if exists:
        print(f"⚪ Skip (exists): {title}")
        continue

    # === 6. DB 저장 ===
    pub = Publication(
        source="한국원자력안전재단",
        title=title,
        published_at=published_at,
        url=detail_url,
        status=PubStatus.PENDING,
    )
    session.add(pub)
    print(f"🟢 Added: {title}")

# === 7. 커밋 ===
session.commit()
session.close()

print("한국원자력안전재단 해외원자력소식 크롤링 완료")
