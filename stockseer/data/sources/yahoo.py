"""Yahoo Finance source (via yfinance) — the default production price feed."""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from .base import NEWS_COLUMNS, NewsSource, PriceSource, TickerProfile, empty_news, empty_prices

log = logging.getLogger(__name__)


class YahooPriceSource(PriceSource):
    name = "yahoo"

    def __init__(self, auto_adjust: bool = False):
        self.auto_adjust = auto_adjust

    def fetch_prices(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        import yfinance as yf

        raw = yf.download(
            symbol, period=period, interval="1d", auto_adjust=self.auto_adjust,
            progress=False, threads=False,
        )
        if raw is None or raw.empty:
            log.warning("yahoo: no rows for %s", symbol)
            return empty_prices()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw.rename(columns=str.lower).reset_index()
        raw = raw.rename(columns={"adj close": "adj_close", "index": "date"})
        if "date" not in raw.columns:
            raw = raw.rename(columns={raw.columns[0]: "date"})
        if "adj_close" not in raw.columns:
            raw["adj_close"] = raw["close"]

        raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.date
        cols = ["date", "open", "high", "low", "close", "adj_close", "volume"]
        return raw[cols].sort_values("date").reset_index(drop=True)

    def fetch_profile(self, symbol: str) -> TickerProfile:
        import yfinance as yf

        try:
            info = yf.Ticker(symbol).get_info() or {}
        except Exception as exc:  # network / rate limit — degrade gracefully
            log.warning("yahoo: profile fetch failed for %s (%s)", symbol, exc)
            info = {}
        return TickerProfile(
            symbol=symbol,
            name=info.get("shortName") or info.get("longName"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            currency=info.get("currency"),
            exchange=info.get("exchange") or info.get("fullExchangeName"),
            market_cap=info.get("marketCap"),
        )


class YahooNewsSource(NewsSource):
    """Headlines attached to a Yahoo quote page. Free, no key, best-effort."""

    name = "yahoo"

    def fetch_news(self, symbol: str, since: dt.datetime | None = None,
                   limit: int = 100) -> pd.DataFrame:
        import yfinance as yf

        try:
            items = yf.Ticker(symbol).news or []
        except Exception as exc:
            log.warning("yahoo: news fetch failed for %s (%s)", symbol, exc)
            return empty_news()

        rows = []
        for it in items[:limit]:
            content = it.get("content", it)
            ts = content.get("pubDate") or it.get("providerPublishTime")
            if isinstance(ts, (int, float)):
                published = dt.datetime.fromtimestamp(ts, tz=dt.UTC)
            elif isinstance(ts, str):
                published = pd.to_datetime(ts, utc=True, errors="coerce").to_pydatetime()
            else:
                published = dt.datetime.now(dt.UTC)
            if since and published < since:
                continue
            url = (content.get("canonicalUrl") or {}).get("url") if isinstance(
                content.get("canonicalUrl"), dict) else it.get("link")
            rows.append({
                "published_at": published,
                "headline": content.get("title") or it.get("title") or "",
                "summary": content.get("summary") or content.get("description"),
                "source": (content.get("provider") or {}).get("displayName")
                if isinstance(content.get("provider"), dict) else it.get("publisher"),
                "url": url,
            })
        if not rows:
            return empty_news()
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)
