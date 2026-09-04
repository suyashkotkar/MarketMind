from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...db.session import get_db
from ..schemas import CompareOut
from ..services import analytics

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("", response_model=CompareOut)
def compare(symbols: str = Query(..., description="comma-separated, 2-10 symbols"),
            db: Session = Depends(get_db)):
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if not 2 <= len(syms) <= 10:
        raise HTTPException(400, "provide between 2 and 10 symbols")
    return analytics.compare(db, syms)
