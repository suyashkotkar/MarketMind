from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from ...db.session import get_db
from ...models import registry
from ..schemas import PredictionOut
from ..services import analytics

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("/model")
def model_info():
    _, meta = registry.load()
    if meta is None:
        raise HTTPException(404, "no trained model yet — POST /admin/train")
    return {
        "version": meta.version, "model_type": meta.model_type,
        "horizon_days": meta.horizon_days, "n_rows": meta.n_rows,
        "n_features": len(meta.features), "tickers": meta.tickers,
        "trained_at": meta.trained_at, "metrics": meta.metrics,
        "feature_importance": meta.feature_importance, "notes": meta.notes,
    }


@router.get("/model/versions")
def model_versions():
    return registry.list_versions()


@router.get("/track-record")
def track_record(symbol: str | None = None, db: Session = Depends(get_db)):
    """Hit rate of predictions this system already made, scored against reality."""
    analytics.score_past_predictions(db)
    return analytics.prediction_track_record(db, symbol)


@router.get("/{symbol}", response_model=PredictionOut)
def predict(symbol: str, db: Session = Depends(get_db)):
    try:
        return analytics.predict_symbol(db, symbol)
    except FileNotFoundError as exc:
        raise HTTPException(409, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("")
def predict_many(symbols: str | None = Query(
        None, description="comma-separated; defaults to the whole universe"),
        db: Session = Depends(get_db)):
    syms = [s.strip() for s in symbols.split(",")] if symbols else None
    return analytics.predict_universe(db, syms)
