from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...data.pipeline import load_prices
from ...db.models import Ticker
from ...db.session import get_db
from ...features.builder import build_price_features
from ..schemas import HistoryOut, OverviewOut, PriceBarOut, TickerOut
from ..services import analytics

router = APIRouter(prefix="/stocks", tags=["stocks"])

INDICATOR_COLUMNS = [
    "sma_20_ratio", "sma_50_ratio", "rsi_14", "macd", "macd_signal", "macd_hist",
    "bb_pct_b", "bb_width", "vol_21d", "atr_pct", "vol_z_20", "drawdown_252",
]


@router.get("", response_model=list[TickerOut])
def list_tickers(db: Session = Depends(get_db)):
    return [
        TickerOut(
            symbol=t.symbol, name=t.name, sector=t.sector, industry=t.industry,
            currency=t.currency, exchange=t.exchange, market_cap=t.market_cap,
            last_ingested_at=t.last_ingested_at,
        )
        for t in db.scalars(select(Ticker).order_by(Ticker.symbol))
    ]


@router.get("/overview", response_model=OverviewOut)
def overview(limit: int = Query(25, ge=1, le=100), db: Session = Depends(get_db)):
    return analytics.overview(db, limit=limit)


@router.get("/{symbol}", response_model=TickerOut)
def get_ticker(symbol: str, db: Session = Depends(get_db)):
    t = db.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
    if not t:
        raise HTTPException(404, f"{symbol.upper()} not in the warehouse")
    return TickerOut(symbol=t.symbol, name=t.name, sector=t.sector,
                     industry=t.industry, currency=t.currency, exchange=t.exchange,
                     market_cap=t.market_cap, last_ingested_at=t.last_ingested_at)


@router.get("/{symbol}/history", response_model=HistoryOut)
def history(symbol: str,
            days: int = Query(365, ge=30, le=5000),
            with_indicators: bool = True,
            db: Session = Depends(get_db)):
    prices = load_prices(db, symbol)
    if prices.empty:
        raise HTTPException(404, f"no price history for {symbol.upper()}")

    indicators: dict[str, list] = {}
    if with_indicators:
        feat = build_price_features(prices)
        tail = feat.tail(days)
        for col in INDICATOR_COLUMNS:
            if col in tail.columns:
                indicators[col] = [None if pd.isna(v) else round(float(v), 6)
                                   for v in tail[col]]

    tail = prices.tail(days)
    bars = [PriceBarOut(date=pd.to_datetime(r["date"]).date(), open=r["open"],
                        high=r["high"], low=r["low"], close=r["close"],
                        adj_close=r["adj_close"], volume=r["volume"])
            for r in tail.to_dict("records")]
    return HistoryOut(symbol=symbol.upper(), bars=bars, indicators=indicators)


@router.get("/{symbol}/stats")
def stats(symbol: str, db: Session = Depends(get_db)):
    prices = load_prices(db, symbol)
    if prices.empty:
        raise HTTPException(404, f"no price history for {symbol.upper()}")
    c = prices["adj_close"]
    ret = np.log(c).diff().dropna()
    return {
        "symbol": symbol.upper(),
        "first_date": str(pd.to_datetime(prices["date"].iloc[0]).date()),
        "last_date": str(pd.to_datetime(prices["date"].iloc[-1]).date()),
        "n_bars": int(len(prices)),
        "latest_close": round(float(c.iloc[-1]), 4),
        "change_1d": analytics._pct_change(c, 1),
        "change_1w": analytics._pct_change(c, 5),
        "change_1m": analytics._pct_change(c, 21),
        "change_1y": analytics._pct_change(c, 252),
        "high_52w": round(float(c.tail(252).max()), 4),
        "low_52w": round(float(c.tail(252).min()), 4),
        "avg_volume_30d": round(float(prices["volume"].tail(30).mean()), 0),
        "annualized_vol": round(float(ret.std(ddof=0) * np.sqrt(252)), 4),
    }
