"""FastAPI application factory."""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .. import __version__
from ..config import settings
from ..db.session import init_db
from .routers import (
    admin,
    alerts,
    anomalies,
    compare,
    health,
    predictions,
    risk,
    sentiment,
    stocks,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
)
log = logging.getLogger("stockseer.api")

DESCRIPTION = """
**StockSeer** turns raw market data into calibrated probabilities, an explainable
risk score, and anomaly alerts — rather than a single made-up price for tomorrow.

Pipeline: `market + news data -> ETL -> features -> prediction & risk models -> this API -> dashboard`

* `/stocks` — universe, OHLCV history, indicators
* `/predictions` — P(up) over the configured horizon, plus the model's own scorecard
* `/risk` — 0-100 risk score with its component breakdown
* `/anomalies` — unusual price/volume behaviour
* `/sentiment` — finance-lexicon news sentiment
* `/compare` — side-by-side metrics and return correlation
* `/alerts` — the signals worth interrupting someone for
* `/admin` — ingest and training jobs

Outputs are statistical estimates, not investment advice.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("StockSeer %s up (env=%s, source=%s)", __version__, settings.env,
             settings.data_source)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name, version=__version__, description=DESCRIPTION,
        lifespan=lifespan, docs_url="/docs", redoc_url="/redoc",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins,
        allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
        return response

    @app.exception_handler(ValueError)
    async def value_error_handler(_request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    p = settings.api_prefix
    app.include_router(health.router, prefix=p)
    app.include_router(stocks.router, prefix=p)
    app.include_router(predictions.router, prefix=p)
    app.include_router(risk.router, prefix=p)
    app.include_router(anomalies.router, prefix=p)
    app.include_router(sentiment.router, prefix=p)
    app.include_router(compare.router, prefix=p)
    app.include_router(alerts.router, prefix=p)
    app.include_router(admin.router, prefix=p)

    @app.get("/", include_in_schema=False)
    def root():
        return {"name": settings.app_name, "version": __version__,
                "docs": "/docs", "api": p}

    return app


app = create_app()
