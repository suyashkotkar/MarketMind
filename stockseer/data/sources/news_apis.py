"""Optional keyed news providers. Used when NEWSAPI_KEY / FINNHUB_KEY are set."""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import requests

from .base import NEWS_COLUMNS, NewsSource, empty_news

log = logging.getLogger(__name__)
TIMEOUT = 15


class NewsAPISource(NewsSource):
    name = "newsapi"
    ENDPOINT = "https://newsapi.org/v2/everything"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_news(self, symbol: str, since: dt.datetime | None = None,
                   limit: int = 100) -> pd.DataFrame:
        since = since or dt.datetime.now(dt.UTC) - dt.timedelta(days=30)
        params = {
            "q": symbol, "language": "en", "sortBy": "publishedAt",
            "pageSize": min(limit, 100), "from": since.date().isoformat(),
            "apiKey": self.api_key,
        }
        try:
            r = requests.get(self.ENDPOINT, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            arts = r.json().get("articles", [])
        except Exception as exc:
            log.warning("newsapi failed for %s: %s", symbol, exc)
            return empty_news()

        rows = [{
            "published_at": pd.to_datetime(a["publishedAt"], utc=True),
            "headline": a.get("title") or "",
            "summary": a.get("description"),
            "source": (a.get("source") or {}).get("name"),
            "url": a.get("url"),
        } for a in arts]
        return pd.DataFrame(rows, columns=NEWS_COLUMNS) if rows else empty_news()


class FinnhubNewsSource(NewsSource):
    name = "finnhub"
    ENDPOINT = "https://finnhub.io/api/v1/company-news"

    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_news(self, symbol: str, since: dt.datetime | None = None,
                   limit: int = 100) -> pd.DataFrame:
        since = since or dt.datetime.now(dt.UTC) - dt.timedelta(days=30)
        params = {
            "symbol": symbol, "from": since.date().isoformat(),
            "to": dt.date.today().isoformat(), "token": self.api_key,
        }
        try:
            r = requests.get(self.ENDPOINT, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            arts = r.json()[:limit]
        except Exception as exc:
            log.warning("finnhub failed for %s: %s", symbol, exc)
            return empty_news()

        rows = [{
            "published_at": dt.datetime.fromtimestamp(a["datetime"], tz=dt.UTC),
            "headline": a.get("headline") or "",
            "summary": a.get("summary"),
            "source": a.get("source"),
            "url": a.get("url"),
        } for a in arts]
        return pd.DataFrame(rows, columns=NEWS_COLUMNS) if rows else empty_news()
