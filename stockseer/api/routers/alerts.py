"""Alerts = the layer that turns raw model output into something worth reading.

An alert fires when the system has an opinion strong enough to interrupt someone:
a confident directional signal, a risk grade in the danger band, or a fresh
anomaly. The work happens in the service layer so it can be cached and reused.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from ...db.session import get_db
from ..schemas import AlertOut
from ..services import analytics

router = APIRouter(prefix="/alerts", tags=["alerts"])


@router.get("", response_model=list[AlertOut])
def alerts(days: int = Query(7, ge=1, le=90),
           min_confidence: float = Query(0.2, ge=0.0, le=1.0),
           db: Session = Depends(get_db)):
    return analytics.build_alerts(db, days=days, min_confidence=min_confidence)
