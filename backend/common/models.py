# backend/common/models.py
from datetime import datetime, date
from sqlalchemy.orm import declarative_base, relationship, Mapped, mapped_column
from sqlalchemy import Integer, String, Text, DateTime, Enum, Boolean, ForeignKey, UniqueConstraint, Index
import enum

Base = declarative_base()

class PubStatus(str, enum.Enum):
    PENDING = "PENDING"
    PARTIAL = "PARTIAL"
    DONE    = "DONE"
    FAILED  = "FAILED"

class Publication(Base):
    __tablename__ = "publications"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    source: Mapped[str] = mapped_column(String(200), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    published_at: Mapped[date | None] = mapped_column(nullable=True)
    url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    content_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), unique=True, nullable=True)

    status: Mapped[PubStatus] = mapped_column(Enum(PubStatus), default=PubStatus.PENDING, nullable=False)

    # 처리 타임스탬프
    ocr_started_at:  Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ocr_ended_at:    Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    nlp_started_at:  Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    nlp_ended_at:    Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # 요약 결과(간단 저장)
    extractive_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstractive_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # 관계
    send_logs: Mapped[list["SendLog"]] = relationship(back_populates="publication")

    __table_args__ = (
        Index("ix_publications_created_at", "created_at"),
    )

class Subscriber(Base):
    __tablename__ = "subscribers"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

class SendLog(Base):
    __tablename__ = "send_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    publication_id: Mapped[int | None] = mapped_column(ForeignKey("publications.id"), nullable=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    publication: Mapped["Publication"] = relationship(back_populates="send_logs")

class DailyCounters(Base):
    __tablename__ = "daily_counters"
    day: Mapped[date] = mapped_column(primary_key=True)
    ingested: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    dedup_skipped: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    emails_sent: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

# 초기화
def init_db(engine):
    Base.metadata.create_all(bind=engine)