"""Operational endpoints: ingest, train, score, cache. Long jobs run in the
background so the HTTP call returns immediately."""
from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from ...config import settings
from ...data.pipeline import ingest_universe
from ...db.session import get_db, session_scope
from .. import cache
from ..schemas import IngestRequest, TrainRequest
from ..services import analytics

log = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

_jobs: dict[str, dict] = {}


def _run_ingest(symbols, period, with_news):
    _jobs["ingest"] = {"status": "running"}
    try:
        with session_scope() as db:
            reports = ingest_universe(db, symbols, period, with_news)
        cache.clear()
        _jobs["ingest"] = {"status": "done",
                           "reports": [r.as_dict() for r in reports]}
    except Exception as exc:
        log.exception("ingest job failed")
        _jobs["ingest"] = {"status": "error", "error": str(exc)}


def _run_train(symbols, horizon, model_type):
    _jobs["train"] = {"status": "running"}
    try:
        with session_scope() as db:
            result = analytics.train(db, symbols, horizon, model_type)
        cache.clear()
        _jobs["train"] = {"status": "done", "result": result}
    except Exception as exc:
        log.exception("train job failed")
        _jobs["train"] = {"status": "error", "error": str(exc)}


@router.post("/ingest")
def ingest(req: IngestRequest, bg: BackgroundTasks, sync: bool = False):
    if sync:
        _run_ingest(req.symbols, req.period, req.with_news)
        return _jobs["ingest"]
    bg.add_task(_run_ingest, req.symbols, req.period, req.with_news)
    return {"status": "accepted",
            "symbols": req.symbols or settings.default_tickers,
            "poll": "/api/v1/admin/jobs/ingest"}


@router.post("/train")
def train(req: TrainRequest, bg: BackgroundTasks, sync: bool = False):
    if sync:
        _run_train(req.symbols, req.horizon_days, req.model_type)
        job = _jobs["train"]
        if job["status"] == "error":
            raise HTTPException(400, job["error"])
        return job
    bg.add_task(_run_train, req.symbols, req.horizon_days, req.model_type)
    return {"status": "accepted", "poll": "/api/v1/admin/jobs/train"}


@router.get("/jobs/{name}")
def job_status(name: str):
    if name not in _jobs:
        raise HTTPException(404, f"no job named '{name}' has run in this process")
    return _jobs[name]


@router.post("/score-predictions")
def score_predictions(db: Session = Depends(get_db)):
    return analytics.score_past_predictions(db)


@router.post("/cache/clear")
def clear_cache(prefix: str | None = None):
    return {"cleared": cache.clear(prefix)}
