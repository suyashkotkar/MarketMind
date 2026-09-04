"""Deterministic synthetic market generator.

Why this exists: CI runners, offline machines and sandboxes often cannot reach
Yahoo Finance. This source produces realistic-looking OHLCV (fat tails, volatility
clustering, volume/|return| coupling, a shared market factor) plus matching
headlines, so the *entire* pipeline — ETL, features, training, risk, anomalies,
API, dashboard — can be exercised without a network.

It is NOT a market simulator for research. Anything trained on it is a smoke test.
"""
from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from .base import NEWS_COLUMNS, NewsSource, PriceSource, TickerProfile

_SECTORS = ["Technology", "Financials", "Energy", "Healthcare", "Consumer"]

_POS = ["beats estimates", "raises guidance", "record revenue", "wins contract",
        "upgraded to buy", "announces buyback", "strong demand", "margin expansion"]
_NEG = ["misses estimates", "cuts guidance", "faces probe", "loses contract",
        "downgraded to sell", "recalls product", "weak demand", "margin pressure"]
_NEU = ["schedules earnings call", "files 10-Q", "names new CFO",
        "presents at conference", "updates investor deck"]


def _seed_for(symbol: str, salt: str = "") -> int:
    h = hashlib.sha256(f"{symbol}|{salt}".encode()).hexdigest()
    return int(h[:8], 16)


def _period_to_days(period: str) -> int:
    period = period.strip().lower()
    if period.endswith("y"):
        return int(float(period[:-1]) * 365)
    if period.endswith("mo"):
        return int(float(period[:-2]) * 30)
    if period.endswith("d"):
        return int(period[:-1])
    return 365 * 5


def _market_factor(n: int) -> np.ndarray:
    """A shared factor so betas and correlations are non-trivial."""
    rng = np.random.default_rng(20240101)
    return rng.standard_normal(n) * 0.008


class SyntheticPriceSource(PriceSource):
    name = "synthetic"

    def __init__(self, end: dt.date | None = None):
        self.end = end or dt.date.today()

    def fetch_prices(self, symbol: str, period: str = "5y") -> pd.DataFrame:
        days = _period_to_days(period)
        dates = pd.bdate_range(end=pd.Timestamp(self.end), periods=int(days * 252 / 365))
        n = len(dates)
        rng = np.random.default_rng(_seed_for(symbol))

        beta = 0.6 + rng.random() * 0.9
        drift = (rng.random() - 0.35) * 0.0006
        base_vol = 0.010 + rng.random() * 0.014

        # GARCH(1,1)-flavoured volatility so risk metrics have something to bite on.
        omega, alpha, beta_g = base_vol**2 * 0.05, 0.09, 0.86
        var = np.empty(n)
        eps = np.empty(n)
        var[0] = base_vol**2
        z = rng.standard_t(df=5, size=n) / np.sqrt(5 / 3)
        mkt = _market_factor(n)
        for t in range(n):
            if t:
                var[t] = omega + alpha * eps[t - 1] ** 2 + beta_g * var[t - 1]
            eps[t] = np.sqrt(var[t]) * z[t]

        idio = eps
        rets = drift + beta * mkt + idio

        # A weak, learnable signal: short-term mean reversion + medium momentum.
        for t in range(21, n):
            rets[t] += -0.045 * rets[t - 1] + 0.020 * rets[t - 21:t - 1].mean()

        # Occasional gap events (earnings-like) every ~63 sessions.
        for t in range(0, n, 63):
            j = min(t + int(rng.integers(0, 63)), n - 1)
            rets[j] += rng.choice([-1, 1]) * (0.03 + rng.random() * 0.05)

        price = 20 + rng.random() * 280
        close = price * np.exp(np.cumsum(rets))
        intraday = np.abs(rng.normal(0, 1, n)) * np.sqrt(var) * 1.4
        open_ = close * np.exp(-rets * 0.5 + rng.normal(0, 0.002, n))
        high = np.maximum(open_, close) * (1 + intraday)
        low = np.minimum(open_, close) * (1 - intraday)
        # Volume rises with the size of the move, but the multiplier is clipped so
        # a fat-tailed day cannot produce an absurd share count.
        shock = np.clip(np.abs(rets) / np.sqrt(var), 0, 4.0)
        base_vol_sh = 3e5 + rng.random() * 4e7
        volume = base_vol_sh * np.exp(0.45 * shock + rng.normal(0, 0.25, n))

        return pd.DataFrame({
            "date": [d.date() for d in dates],
            "open": np.round(open_, 4),
            "high": np.round(high, 4),
            "low": np.round(low, 4),
            "close": np.round(close, 4),
            "adj_close": np.round(close, 4),
            "volume": np.round(volume, 0),
        })

    def fetch_profile(self, symbol: str) -> TickerProfile:
        rng = np.random.default_rng(_seed_for(symbol, "profile"))
        return TickerProfile(
            symbol=symbol,
            name=f"{symbol} Synthetic Corp.",
            sector=_SECTORS[int(rng.integers(0, len(_SECTORS)))],
            industry="Simulated",
            currency="USD",
            exchange="SYNTH",
            market_cap=float(rng.integers(5, 3000)) * 1e9,
        )


class SyntheticNewsSource(NewsSource):
    name = "synthetic"

    def fetch_news(self, symbol: str, since: dt.datetime | None = None,
                   limit: int = 100) -> pd.DataFrame:
        rng = np.random.default_rng(_seed_for(symbol, "news"))
        now = dt.datetime.now(dt.UTC)
        rows = []
        for i in range(limit):
            published = now - dt.timedelta(hours=float(rng.integers(1, 24 * 45)))
            if since and published < since:
                continue
            bucket = rng.random()
            if bucket < 0.38:
                phrase, tone = _POS[int(rng.integers(0, len(_POS)))], "positive"
            elif bucket < 0.70:
                phrase, tone = _NEG[int(rng.integers(0, len(_NEG)))], "negative"
            else:
                phrase, tone = _NEU[int(rng.integers(0, len(_NEU)))], "neutral"
            rows.append({
                "published_at": published,
                "headline": f"{symbol} {phrase}",
                "summary": f"Simulated coverage: {symbol} {phrase}. "
                           f"Analysts weigh the {tone} read-through for the quarter.",
                "source": "SyntheticWire",
                "url": f"https://example.invalid/{symbol.lower()}/{i}",
            })
        return pd.DataFrame(rows, columns=NEWS_COLUMNS).sort_values(
            "published_at", ascending=False).reset_index(drop=True)
