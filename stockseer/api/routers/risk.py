from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...db.session import get_db
from ..schemas import RiskOut
from ..services import analytics

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/ranking")
def ranking(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """Whole universe sorted from safest to riskiest."""
    return analytics.risk_ranking(db, limit=limit)


@router.get("/{symbol}", response_model=RiskOut)
def risk(symbol: str, db: Session = Depends(get_db)):
    try:
        return analytics.risk_for(db, symbol)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
