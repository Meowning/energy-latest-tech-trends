from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from common.config import MYSQL_URL

# MySQL (MariaDB) 연결
engine = create_engine(
    MYSQL_URL,
    pool_pre_ping=True,      # 죽은 커넥션 자동 감지
    pool_recycle=3600,       # 1시간마다 커넥션 재생성
    pool_size=5,             # 기본 커넥션 풀 사이즈
    max_overflow=0,          # 초과 커넥션 수
    future=True
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()