"""Cleaning rules applied between raw fetch and the warehouse.

Everything here is deliberately conservative: bad market data is far more
damaging to a model than missing market data.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def clean_prices(df: pd.DataFrame, symbol: str = "?",
                 max_abs_return: float = 0.75) -> pd.DataFrame:
    """Validate + normalise a raw OHLCV frame.

    Steps: schema check -> type coercion -> drop null/non-positive prices ->
    de-duplicate dates -> fix inverted high/low -> flag implausible jumps ->
    forward-fill *volume only* (never prices).
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=REQUIRED)

    df = df.copy()
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{symbol}: missing columns {missing}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    for c in ["open", "high", "low", "close", "adj_close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n0 = len(df)
    df = df.dropna(subset=["close", "adj_close"])
    df = df[(df[["open", "high", "low", "close", "adj_close"]] > 0).all(axis=1)]

    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    # High must bound the bar; some vendors transpose them on illiquid days.
    lo = df[["open", "close", "low", "high"]].min(axis=1)
    hi = df[["open", "close", "low", "high"]].max(axis=1)
    df["low"], df["high"] = lo, hi

    df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)

    # Drop single-print spikes: a >75% move that fully reverses next session is
    # almost always a bad tick, not a real event.
    ret = df["adj_close"].pct_change()
    suspect = (ret.abs() > max_abs_return) & (ret.shift(-1).abs() > max_abs_return) & \
              (np.sign(ret) != np.sign(ret.shift(-1)))
    if suspect.any():
        log.warning("%s: dropping %d suspected bad ticks", symbol, int(suspect.sum()))
        df = df[~suspect]

    dropped = n0 - len(df)
    if dropped:
        log.info("%s: cleaning removed %d/%d rows", symbol, dropped, n0)
    return df[REQUIRED].reset_index(drop=True)


def clean_news(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["published_at", "headline", "summary",
                                     "source", "url"])
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 5]
    df = df.drop_duplicates(subset=["headline"])
    return df.reset_index(drop=True)


def assert_no_gaps(df: pd.DataFrame, max_gap_days: int = 10) -> list[tuple]:
    """Return (start, end) pairs where the series skips more than max_gap_days."""
    d = pd.to_datetime(pd.Series(df["date"]))
    gaps = d.diff().dt.days
    idx = gaps[gaps > max_gap_days].index
    return [(d[i - 1].date(), d[i].date()) for i in idx]
