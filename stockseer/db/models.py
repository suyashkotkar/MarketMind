"""SQLAlchemy ORM models — the warehouse layer of the pipeline."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Ticker(Base):
    __tablename__ = "tickers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    sector: Mapped[str | None] = mapped_column(String(128))
    industry: Mapped[str | None] = mapped_column(String(128))
    currency: Mapped[str | None] = mapped_column(String(8))
    exchange: Mapped[str | None] = mapped_column(String(32))
    market_cap: Mapped[float | None] = mapped_column(Float)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    last_ingested_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    prices: Mapped[list[PriceBar]] = relationship(
        back_populates="ticker", cascade="all, delete-orphan"
    )


class PriceBar(Base):
    """Cleaned daily OHLCV. One row per (ticker, date)."""

    __tablename__ = "price_bars"
    __table_args__ = (
        UniqueConstraint("ticker_id", "date", name="uq_price_ticker_date"),
        Index("ix_price_ticker_date", "ticker_id", "date"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id", ondelete="CASCADE"))
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    ticker: Mapped[Ticker] = relationship(back_populates="prices")


class NewsItem(Base):
    __tablename__ = "news_items"
    __table_args__ = (
        UniqueConstraint("ticker_id", "url_hash", name="uq_news_ticker_url"),
        Index("ix_news_ticker_pub", "ticker_id", "published_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id", ondelete="CASCADE"))
    published_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), index=True)
    headline: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str | None] = mapped_column(String(128))
    url: Mapped[str | None] = mapped_column(Text)
    url_hash: Mapped[str] = mapped_column(String(64))
    sentiment: Mapped[float | None] = mapped_column(Float)      # -1..1
    sentiment_label: Mapped[str | None] = mapped_column(String(16))


class Prediction(Base):
    """Persisted model output so the dashboard reads instantly and we can score
    predictions against realised outcomes later."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("ticker_id", "as_of", "horizon_days",
                         name="uq_pred_ticker_asof_h"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id", ondelete="CASCADE"))
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    horizon_days: Mapped[int] = mapped_column(Integer)
    prob_up: Mapped[float] = mapped_column(Float)
    direction: Mapped[str] = mapped_column(String(8))           # UP | DOWN | NEUTRAL
    confidence: Mapped[float] = mapped_column(Float)            # 0..1
    expected_move_pct: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(64))
    realized_return: Mapped[float | None] = mapped_column(Float)
    was_correct: Mapped[bool | None] = mapped_column(Boolean)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class RiskSnapshot(Base):
    __tablename__ = "risk_snapshots"
    __table_args__ = (
        UniqueConstraint("ticker_id", "as_of", name="uq_risk_ticker_asof"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id", ondelete="CASCADE"))
    as_of: Mapped[dt.date] = mapped_column(Date, index=True)
    risk_score: Mapped[float] = mapped_column(Float)
    grade: Mapped[str] = mapped_column(String(4))
    annualized_vol: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float | None] = mapped_column(Float)
    var_95: Mapped[float | None] = mapped_column(Float)
    cvar_95: Mapped[float | None] = mapped_column(Float)
    beta: Mapped[float | None] = mapped_column(Float)
    sharpe: Mapped[float | None] = mapped_column(Float)
    sortino: Mapped[float | None] = mapped_column(Float)
    components_json: Mapped[str | None] = mapped_column(Text)


class AnomalyEvent(Base):
    __tablename__ = "anomaly_events"
    __table_args__ = (
        UniqueConstraint("ticker_id", "date", "kind", name="uq_anom_ticker_date_kind"),
    )

    id: Mapped[int] = mapped_column(BigInteger().with_variant(Integer, "sqlite"),
                                    primary_key=True)
    ticker_id: Mapped[int] = mapped_column(ForeignKey("tickers.id", ondelete="CASCADE"))
    date: Mapped[dt.date] = mapped_column(Date, index=True)
    kind: Mapped[str] = mapped_column(String(32))     # PRICE_SPIKE | VOLUME_SURGE | ...
    severity: Mapped[float] = mapped_column(Float)    # 0..1
    score: Mapped[float] = mapped_column(Float)       # raw detector score
    detail: Mapped[str | None] = mapped_column(Text)


class ModelRun(Base):
    __tablename__ = "model_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    version: Mapped[str] = mapped_column(String(64), index=True)
    model_type: Mapped[str] = mapped_column(String(32))
    horizon_days: Mapped[int] = mapped_column(Integer)
    n_rows: Mapped[int] = mapped_column(Integer)
    n_features: Mapped[int] = mapped_column(Integer)
    tickers: Mapped[str | None] = mapped_column(Text)
    metrics_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
