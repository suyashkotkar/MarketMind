"""Data-source contract.

Every source returns the same tidy frames so the ETL layer never has to know
where the bytes came from. Swap Yahoo for a paid vendor by adding one class.
"""
from __future__ import annotations

import abc
import datetime as dt
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]
NEWS_COLUMNS = ["published_at", "headline", "summary", "source", "url"]


@dataclass
class TickerProfile:
    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    exchange: str | None = None
    market_cap: float | None = None
    extra: dict = field(default_factory=dict)


class PriceSource(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def fetch_prices(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        """Return a DataFrame with PRICE_COLUMNS, ascending by date."""

    def fetch_many(self, symbols: Iterable[str], period: str = "5y"
                   ) -> dict[str, pd.DataFrame]:
        return {s: self.fetch_prices(s, period=period) for s in symbols}

    def fetch_profile(self, symbol: str) -> TickerProfile:
        return TickerProfile(symbol=symbol)


class NewsSource(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def fetch_news(self, symbol: str, since: dt.datetime | None = None,
                   limit: int = 100) -> pd.DataFrame:
        """Return a DataFrame with NEWS_COLUMNS, newest first."""


def empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=PRICE_COLUMNS)


def empty_news() -> pd.DataFrame:
    return pd.DataFrame(columns=NEWS_COLUMNS)
