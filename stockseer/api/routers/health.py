from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ... import __version__
from ...config import settings
from ...db.models import PriceBar, Ticker
from ...db.session import get_db
from ...models import registry
from ..schemas import HealthOut

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
def health(db: Session = Depends(get_db)):
    try:
        n_tickers = db.scalar(select(func.count(Ticker.id))) or 0
        latest = db.scalar(select(func.max(PriceBar.date)))
        db_status = "ok"
    except Exception as exc:
        n_tickers, latest, db_status = 0, None, f"error: {exc}"

    return HealthOut(
        status="ok" if db_status == "ok" else "degraded",
        version=__version__, env=settings.env, database=db_status,
        data_source=settings.data_source,
        model_version=registry.latest_version(),
        tickers_loaded=int(n_tickers), latest_price_date=latest,
    )


@router.get("/config")
def public_config():
    """Non-secret settings the frontend needs to render correctly."""
    return {
        "app_name": settings.app_name,
        "horizon_days": settings.horizon_days,
        "long_threshold": settings.long_threshold,
        "short_threshold": settings.short_threshold,
        "benchmark": settings.benchmark_ticker,
        "data_source": settings.data_source,
        "default_tickers": settings.default_tickers,
    }
