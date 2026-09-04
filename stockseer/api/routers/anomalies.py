from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...db.session import get_db
from ..schemas import AnomalyResponse
from ..services import analytics

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("/recent")
def recent(days: int = Query(30, ge=1, le=365),
           min_severity: float = Query(0.3, ge=0.0, le=1.0),
           db: Session = Depends(get_db)):
    """Cross-universe feed for the alerts panel."""
    return analytics.recent_anomalies(db, days=days, min_severity=min_severity)


@router.get("/{symbol}", response_model=AnomalyResponse)
def anomalies(symbol: str, lookback_days: int = Query(180, ge=10, le=1000),
              db: Session = Depends(get_db)):
    try:
        return analytics.anomalies_for(db, symbol, lookback_days=lookback_days)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
