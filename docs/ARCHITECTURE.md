# Architecture

## The pipeline

```
                    ┌──────────────────────────────────────┐
                    │  Stock / news data                   │
                    │  Yahoo Finance · NewsAPI · Finnhub   │
                    │  (or the synthetic generator)        │
                    └───────────────┬──────────────────────┘
                                    │  stockseer/data/sources/
                    ┌───────────────▼──────────────────────┐
                    │  Data collection                     │
                    │  one PriceSource / NewsSource        │
                    │  interface, swappable by env var     │
                    └───────────────┬──────────────────────┘
                                    │  stockseer/data/cleaning.py
                    ┌───────────────▼──────────────────────┐
                    │  Cleaning + ETL                      │
                    │  schema check · null & non-positive  │
                    │  drop · de-dupe · high/low repair ·  │
                    │  bad-tick removal · idempotent upsert│
                    └───────────────┬──────────────────────┘
                                    │  PostgreSQL (SQLAlchemy)
                    ┌───────────────▼──────────────────────┐
                    │  Warehouse                           │
                    │  tickers · price_bars · news_items    │
                    │  predictions · risk_snapshots ·      │
                    │  anomaly_events · model_runs         │
                    └───────────────┬──────────────────────┘
                                    │  stockseer/features/
                    ┌───────────────▼──────────────────────┐
                    │  Feature engineering                 │
                    │  ~60 columns: trend, momentum,       │
                    │  volatility, volume, structure,      │
                    │  market context, lagged sentiment    │
                    └───────────────┬──────────────────────┘
                                    │  stockseer/models/
              ┌─────────────────────┼──────────────────────┐
              │                     │                      │
   ┌──────────▼────────┐ ┌──────────▼────────┐ ┌───────────▼───────┐
   │ Prediction model  │ │ Risk model        │ │ Anomaly detector  │
   │ LightGBM/XGBoost  │ │ 6 weighted        │ │ z-score rules +   │
   │ purged walk-fwd   │ │ components →      │ │ IsolationForest   │
   │ calibrated P(up)  │ │ 0–100 + grade     │ │ typed events      │
   └──────────┬────────┘ └──────────┬────────┘ └───────────┬───────┘
              └─────────────────────┼──────────────────────┘
                                    │  stockseer/api/services/analytics.py
                    ┌───────────────▼──────────────────────┐
                    │  API / Backend  (FastAPI)            │
                    │  thin routers · TTL cache · OpenAPI  │
                    └───────────────┬──────────────────────┘
                                    │  JSON over HTTP
                    ┌───────────────▼──────────────────────┐
                    │  Interactive dashboard (React+Plotly)│
                    └───────────────┬──────────────────────┘
                                    │
                    ┌───────────────▼──────────────────────┐
                    │  User: insights and alerts           │
                    └──────────────────────────────────────┘
```

## Module map

```
stockseer/
  config.py                 pydantic-settings; one env-driven Settings object
  db/
    models.py               7 tables, unique constraints that make ETL idempotent
    session.py              engine, SessionLocal, FastAPI dependency
  data/
    sources/                base.py (interfaces) · yahoo.py · synthetic.py · news_apis.py
    cleaning.py             validation rules applied between fetch and warehouse
    pipeline.py             collect → clean → upsert; plus the read helpers
  features/
    technical.py            indicators in pure pandas/numpy, each unit-testable
    builder.py              assembles the model matrix; owns the no-look-ahead rules
  models/
    validation.py           PurgedWalkForward
    direction.py            training, calibration, metrics, inference
    risk.py                 composite score + narrative
    anomaly.py              rules + IsolationForest
    registry.py             versioned artifacts, metadata, LATEST pointer, memoised load
  sentiment/
    lexicon.py              finance-domain word weights, negators, intensifiers
    scorer.py               scoring with negation/intensifier handling
  api/
    main.py                 app factory, CORS, timing middleware
    schemas.py              the contract the frontend codes against
    cache.py                TTL cache + @cached decorator
    routers/                thin HTTP layer, one module per resource
    services/analytics.py   all the logic the CLI and the API share
  cli.py                    init-db · ingest · train · predict · risk · anomalies ·
                            compare · pipeline · serve
```

## Decisions worth defending

**Why a source interface instead of calling yfinance directly.** Vendors go
down, change shape, and rate-limit. `PriceSource`/`NewsSource` means swapping in
a paid feed is one new class, and it is what makes the offline synthetic source —
and therefore hermetic CI — possible.

**Why the warehouse, rather than fetching on request.** Indicators need years of
history; recomputing from the vendor on every page load would be slow, rate-limited
and non-reproducible. The warehouse also lets predictions be *scored against
reality later*, which is the only honest measure of the system.

**Why a pooled cross-sectional model.** ~1,200 usable rows per ticker over five
years is thin for a boosted tree. Pooling gives ~12k and forces cross-sectional
patterns rather than one stock's idiosyncrasies. `symbol` is deliberately not a
feature, so the model cannot memorise "NVDA goes up".

**Why purged walk-forward instead of KFold.** Two leaks: future rows training a
model that scores past rows, and the h-day forward target of the last training
rows overlapping the first test rows. The embargo closes the second. Random CV
on this data reports impressive, meaningless numbers.

**Why the risk score is a weighted blend and not a learned model.** A user told
"risk 78/100" needs to know why. Every component's raw value, scaled value,
weight and contribution comes back with the score, and the score equals their
sum — a property the test suite asserts. A learned risk score would be more
flexible and completely unauditable.

**Why two anomaly detectors.** Rules give the event a *type* a person can act on
("volume surge", "gap"); the isolation forest catches joint patterns that look
unremarkable one dimension at a time. Neither alone is enough.

**Why a lexicon instead of a transformer for sentiment.** Deterministic, no model
download, no GPU, fast enough to score at ingest time — and general-purpose
sentiment models mislabel financial text ("liability", "cost", "tax" are not
negative in a filing). A FinBERT scorer would slot in behind `score_texts`.

**Why the service layer.** Routers stay thin and the CLI and the API call exactly
the same functions, so a scheduled job and an HTTP request can never drift apart.
It is also the single place caching is applied.

## Data model

| Table | Grain | Notes |
|---|---|---|
| `tickers` | one per symbol | profile + last ingest timestamp |
| `price_bars` | (ticker, date) | unique constraint = idempotent upsert |
| `news_items` | (ticker, url hash) | sentiment scored at ingest |
| `predictions` | (ticker, as_of, horizon) | `realized_return` / `was_correct` backfilled later |
| `risk_snapshots` | (ticker, as_of) | components stored as JSON for auditability |
| `anomaly_events` | (ticker, date, kind) | de-duplicated by construction |
| `model_runs` | one per training run | metrics JSON, lineage for the model registry |

## Request path

`GET /api/v1/predictions/AAPL`

1. Router → `analytics.predict_symbol`
2. `registry.load()` — memoised, so the bundle is unpickled once per process
3. `load_prices` + `load_news` from Postgres
4. `build_dataset` — indicators, market context, lagged sentiment
5. Take the last row with enough non-null features
6. `predict_proba` through the calibrator (falling back to the raw estimator)
7. Classify against the long/short thresholds, size the expected move from
   21-day realized volatility
8. Upsert into `predictions` so the track record can be scored later
9. Serialise through the pydantic schema

## Scaling notes

The current shape comfortably handles a universe in the low hundreds on one
box. Beyond that:

- Move the TTL cache to Redis (`api/cache.py` is a two-function interface).
- Move ingest and training to a worker (Celery/RQ/Arq) — `admin.py` already
  runs them as background jobs with a poll endpoint.
- Partition `price_bars` by year, or move to TimescaleDB.
- Precompute features into a feature table on the daily job rather than
  recomputing per request.
- Add Alembic once the schema starts evolving; `create_all` is only right while
  the schema is still young.
