"""Source factory — one switch (`DATA_SOURCE`) selects the whole feed."""
from __future__ import annotations

import logging

from ...config import settings
from .base import NewsSource, PriceSource, TickerProfile  # noqa: F401
from .synthetic import SyntheticNewsSource, SyntheticPriceSource
from .yahoo import YahooNewsSource, YahooPriceSource

log = logging.getLogger(__name__)


def get_price_source(name: str | None = None) -> PriceSource:
    name = (name or settings.data_source).lower()
    if name in ("yahoo", "yfinance"):
        return YahooPriceSource()
    if name in ("synthetic", "sim", "offline"):
        return SyntheticPriceSource()
    raise ValueError(f"Unknown price source '{name}' (yahoo | synthetic)")


def get_news_source(name: str | None = None) -> NewsSource | None:
    name = (name or settings.news_source).lower()
    if name == "none":
        return None
    if name == "auto":
        if settings.data_source.lower() in ("synthetic", "sim", "offline"):
            name = "synthetic"
        elif settings.finnhub_key:
            name = "finnhub"
        elif settings.newsapi_key:
            name = "newsapi"
        else:
            name = "yahoo"
    if name == "synthetic":
        return SyntheticNewsSource()
    if name == "yahoo":
        return YahooNewsSource()
    if name == "newsapi":
        from .news_apis import NewsAPISource
        if not settings.newsapi_key:
            log.warning("NEWSAPI_KEY missing; skipping news ingestion")
            return None
        return NewsAPISource(settings.newsapi_key)
    if name == "finnhub":
        from .news_apis import FinnhubNewsSource
        if not settings.finnhub_key:
            log.warning("FINNHUB_KEY missing; skipping news ingestion")
            return None
        return FinnhubNewsSource(settings.finnhub_key)
    raise ValueError(f"Unknown news source '{name}'")
