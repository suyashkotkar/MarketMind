"""Pydantic response models — these are the API contract the React app codes against."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class HealthOut(BaseModel):
    status: str
    version: str
    env: str
    database: str
    data_source: str
    model_version: str | None = None
    tickers_loaded: int = 0
    latest_price_date: dt.date | None = None


class TickerOut(BaseModel):
    symbol: str
    name: str | None = None
    sector: str | None = None
    industry: str | None = None
    currency: str | None = None
    exchange: str | None = None
    market_cap: float | None = None
    last_ingested_at: dt.datetime | None = None


class PriceBarOut(BaseModel):
    date: dt.date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: float


class HistoryOut(BaseModel):
    symbol: str
    bars: list[PriceBarOut]
    indicators: dict[str, list[float | None]] = Field(default_factory=dict)


class PredictionOut(BaseModel):
    symbol: str
    as_of: dt.date
    horizon_days: int
    prob_up: float
    direction: str
    confidence: float
    expected_move_pct: float | None = None
    latest_close: float | None = None
    model_version: str | None = None
    model_metrics: dict[str, Any] = Field(default_factory=dict)
    top_features: dict[str, float] = Field(default_factory=dict)
    disclaimer: str


class RiskComponentOut(BaseModel):
    name: str
    raw: float
    scaled: float
    weight: float
    contribution: float


class RiskOut(BaseModel):
    symbol: str
    as_of: str
    risk_score: float
    grade: str
    annualized_vol: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float | None = None
    sharpe: float
    sortino: float
    downside_deviation: float
    median_dollar_volume: float
    components: list[RiskComponentOut]
    narrative: str


class AnomalyOut(BaseModel):
    date: str
    kind: str
    severity: float
    score: float
    detail: str
    close: float
    volume: float
    return_pct: float


class AnomalyResponse(BaseModel):
    symbol: str
    events: list[AnomalyOut]
    summary: dict[str, Any]


class NewsOut(BaseModel):
    published_at: dt.datetime
    headline: str
    summary: str | None = None
    source: str | None = None
    url: str | None = None
    sentiment: float | None = None
    sentiment_label: str | None = None


class SentimentOut(BaseModel):
    symbol: str
    articles: list[NewsOut]
    mean_sentiment: float | None = None
    label: str | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    daily: list[dict[str, Any]] = Field(default_factory=list)


class CompareRow(BaseModel):
    symbol: str
    latest_close: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_1y: float | None = None
    risk_score: float | None = None
    grade: str | None = None
    annualized_vol: float | None = None
    max_drawdown: float | None = None
    sharpe: float | None = None
    beta: float | None = None
    prob_up: float | None = None
    direction: str | None = None
    error: str | None = None


class CompareOut(BaseModel):
    rows: list[CompareRow]
    series: dict[str, dict[str, list]] = Field(default_factory=dict)
    correlation: dict[str, dict[str, float]] = Field(default_factory=dict)


class AlertOut(BaseModel):
    symbol: str
    kind: str          # SIGNAL | RISK | ANOMALY
    level: str         # info | warning | critical
    message: str
    date: str
    meta: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    symbols: list[str] | None = None
    period: str | None = None
    with_news: bool = True


class TrainRequest(BaseModel):
    symbols: list[str] | None = None
    horizon_days: int | None = None
    model_type: str | None = None


class OverviewOut(BaseModel):
    as_of: dt.date
    model_version: str | None = None
    universe_size: int
    cards: list[dict[str, Any]]
