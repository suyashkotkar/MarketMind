"""ETL: collect -> clean -> upsert into the warehouse.

Idempotent by construction — re-running for the same ticker/date range updates
rows in place rather than duplicating them, so it is safe on a cron.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import settings
from ..db.models import NewsItem, PriceBar, Ticker
from ..sentiment.scorer import score_texts
from .cleaning import clean_news, clean_prices
from .sources import get_news_source, get_price_source

log = logging.getLogger(__name__)


@dataclass
class IngestReport:
    symbol: str
    price_rows_in: int = 0
    price_rows_written: int = 0
    news_rows_written: int = 0
    first_date: dt.date | None = None
    last_date: dt.date | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price_rows_in": self.price_rows_in,
            "price_rows_written": self.price_rows_written,
            "news_rows_written": self.news_rows_written,
            "first_date": self.first_date.isoformat() if self.first_date else None,
            "last_date": self.last_date.isoformat() if self.last_date else None,
            "error": self.error,
        }


def get_or_create_ticker(db: Session, symbol: str, profile=None) -> Ticker:
    symbol = symbol.upper().strip()
    t = db.scalar(select(Ticker).where(Ticker.symbol == symbol))
    if t is None:
        t = Ticker(symbol=symbol)
        db.add(t)
        db.flush()
    if profile is not None:
        t.name = profile.name or t.name
        t.sector = profile.sector or t.sector
        t.industry = profile.industry or t.industry
        t.currency = profile.currency or t.currency
        t.exchange = profile.exchange or t.exchange
        t.market_cap = profile.market_cap or t.market_cap
    return t


def _upsert_prices(db: Session, ticker: Ticker, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    existing = {
        r.date: r for r in db.scalars(
            select(PriceBar).where(PriceBar.ticker_id == ticker.id)
        )
    }
    written = 0
    new_rows = []
    for rec in df.to_dict("records"):
        cur = existing.get(rec["date"])
        if cur is None:
            new_rows.append(PriceBar(ticker_id=ticker.id, **rec))
            written += 1
        elif abs(cur.adj_close - rec["adj_close"]) > 1e-9 or abs(
                cur.volume - rec["volume"]) > 1e-6:
            for k, v in rec.items():
                setattr(cur, k, v)
            written += 1
    if new_rows:
        db.bulk_save_objects(new_rows)
    return written


def _url_hash(url: str | None, headline: str) -> str:
    return hashlib.sha256((url or headline).encode("utf-8")).hexdigest()[:64]


def _upsert_news(db: Session, ticker: Ticker, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    texts = (df["headline"].fillna("") + ". " + df["summary"].fillna("")).tolist()
    scored = score_texts(texts)
    df = df.assign(
        sentiment=[s.score for s in scored],
        sentiment_label=[s.label for s in scored],
        url_hash=[_url_hash(u, h) for u, h in zip(df["url"], df["headline"], strict=False)],
    )
    have = set(db.scalars(
        select(NewsItem.url_hash).where(NewsItem.ticker_id == ticker.id)
    ))
    rows = [
        NewsItem(
            ticker_id=ticker.id,
            published_at=r["published_at"].to_pydatetime(),
            headline=r["headline"], summary=r["summary"], source=r["source"],
            url=r["url"], url_hash=r["url_hash"],
            sentiment=r["sentiment"], sentiment_label=r["sentiment_label"],
        )
        for r in df.to_dict("records") if r["url_hash"] not in have
    ]
    if rows:
        db.bulk_save_objects(rows)
    return len(rows)


def ingest_symbol(db: Session, symbol: str, period: str | None = None,
                  with_news: bool = True) -> IngestReport:
    period = period or settings.history_period
    rep = IngestReport(symbol=symbol.upper())
    price_src = get_price_source()

    try:
        raw = price_src.fetch_prices(symbol, period=period)
        rep.price_rows_in = len(raw)
        clean = clean_prices(raw, symbol=symbol)
        if clean.empty:
            rep.error = "no usable price rows returned"
            return rep

        ticker = get_or_create_ticker(db, symbol, price_src.fetch_profile(symbol))
        rep.price_rows_written = _upsert_prices(db, ticker, clean)
        rep.first_date, rep.last_date = clean["date"].iloc[0], clean["date"].iloc[-1]

        if with_news:
            news_src = get_news_source()
            if news_src is not None:
                news = clean_news(news_src.fetch_news(symbol, limit=100))
                rep.news_rows_written = _upsert_news(db, ticker, news)

        ticker.last_ingested_at = dt.datetime.now(dt.UTC)
        db.flush()
    except Exception as exc:  # one bad ticker must not kill the batch
        log.exception("ingest failed for %s", symbol)
        rep.error = f"{type(exc).__name__}: {exc}"
    return rep


def ingest_universe(db: Session, symbols: list[str] | None = None,
                    period: str | None = None,
                    with_news: bool = True) -> list[IngestReport]:
    symbols = symbols or (settings.default_tickers + [settings.benchmark_ticker])
    seen, ordered = set(), []
    for s in symbols:
        s = s.upper()
        if s not in seen:
            seen.add(s)
            ordered.append(s)

    reports = []
    for s in ordered:
        rep = ingest_symbol(db, s, period=period, with_news=with_news)
        db.commit()
        log.info("ingested %s: %s", s, rep.as_dict())
        reports.append(rep)
    return reports


# --------------------------------------------------------------------------- #
# Read helpers used by the feature/model/API layers
# --------------------------------------------------------------------------- #

def load_prices(db: Session, symbol: str, start: dt.date | None = None,
                end: dt.date | None = None) -> pd.DataFrame:
    q = (
        select(PriceBar.date, PriceBar.open, PriceBar.high, PriceBar.low,
               PriceBar.close, PriceBar.adj_close, PriceBar.volume)
        .join(Ticker, Ticker.id == PriceBar.ticker_id)
        .where(Ticker.symbol == symbol.upper())
    )
    if start:
        q = q.where(PriceBar.date >= start)
    if end:
        q = q.where(PriceBar.date <= end)
    rows = db.execute(q.order_by(PriceBar.date)).all()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close",
                                     "adj_close", "volume"])
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_news(db: Session, symbol: str, limit: int = 50) -> pd.DataFrame:
    q = (
        select(NewsItem.published_at, NewsItem.headline, NewsItem.summary,
               NewsItem.source, NewsItem.url, NewsItem.sentiment,
               NewsItem.sentiment_label)
        .join(Ticker, Ticker.id == NewsItem.ticker_id)
        .where(Ticker.symbol == symbol.upper())
        .order_by(NewsItem.published_at.desc())
        .limit(limit)
    )
    rows = db.execute(q).all()
    return pd.DataFrame(rows, columns=["published_at", "headline", "summary",
                                       "source", "url", "sentiment",
                                       "sentiment_label"])


def list_symbols(db: Session) -> list[str]:
    return list(db.scalars(select(Ticker.symbol).order_by(Ticker.symbol)))
