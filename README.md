# StockSeer

An end-to-end stock analysis system that answers **"how likely is this to go up, and how much can it hurt me?"** instead of pretending to know tomorrow's closing price.

```
Stock / news data → collection → cleaning + ETL → feature engineering
      → prediction model + risk model → FastAPI → React dashboard → you
```

Every layer is real code you can run today: a warehouse, a walk-forward-validated
classifier, an explainable risk score, an anomaly detector, a finance-tuned
sentiment scorer, an OpenAPI backend and a dark-mode dashboard.

---

## What it actually does

| Capability | Where it lives | What you get |
|---|---|---|
| Automatic historical collection | `stockseer/data/` | Daily OHLCV + headlines, idempotent upserts, safe to cron |
| Technical indicators | `stockseer/features/technical.py` | RSI, MACD, SMA/EMA, Bollinger, ATR, ADX, stochastic, OBV, MFI, realized & Parkinson volatility |
| Direction prediction | `stockseer/models/direction.py` | Calibrated **P(up)** over a 5-session horizon, pooled across the universe |
| Risk scoring | `stockseer/models/risk.py` | 0–100 score + letter grade, broken into six named, weighted components |
| Stock comparison | `/api/v1/compare` | Indexed performance, risk side by side, return correlation |
| News sentiment | `stockseer/sentiment/` | Loughran–McDonald-style finance lexicon with negation and intensifier handling |
| Anomaly detection | `stockseer/models/anomaly.py` | Z-score rules (typed events) + IsolationForest (joint outliers) |
| Dashboard | `frontend/` | Candles, indicators, probability, risk breakdown, sentiment, anomalies, alerts |

---

## Quick start

### The 60-second version (no network, no API keys, no database server)

```bash
pip install -r requirements.txt
make demo            # DATA_SOURCE=synthetic: ingest → train → predict
make api             # http://localhost:8000/docs
make web             # http://localhost:5173
```

`DATA_SOURCE=synthetic` generates deterministic pseudo-market data with fat
tails, volatility clustering and a shared market factor, so the whole pipeline —
including CI — runs anywhere. It is a smoke test, **not** a research dataset.

### With real market data

```bash
cp .env.example .env       # DATA_SOURCE=yahoo is the default
python -m stockseer.cli pipeline
```

`pipeline` = ingest the universe → train with walk-forward validation → write
predictions. It prints the model's honest scorecard when it finishes.

### With Docker + Postgres

```bash
docker compose up --build -d          # Postgres, API :8000, dashboard :8080
docker compose --profile jobs run --rm etl    # first load + training
```

---

## Command line

```bash
python -m stockseer.cli init-db
python -m stockseer.cli ingest --symbols AAPL,MSFT,NVDA --period 5y
python -m stockseer.cli train --model-type lightgbm --horizon 5
python -m stockseer.cli predict AAPL
python -m stockseer.cli risk NVDA
python -m stockseer.cli anomalies TSLA --lookback 90
python -m stockseer.cli compare AAPL,MSFT,GOOGL
python -m stockseer.cli serve --reload
```

## API

Interactive docs at `/docs`. Everything is under `/api/v1`.

| Endpoint | Purpose |
|---|---|
| `GET /health`, `GET /config` | Liveness, and the non-secret settings the UI needs |
| `GET /stocks`, `/stocks/{s}`, `/stocks/{s}/history`, `/stocks/{s}/stats` | Universe, OHLCV, indicator series |
| `GET /stocks/overview` | One card per ticker: price, signal, risk |
| `GET /predictions/{s}` | P(up), direction, confidence, expected move, model metrics |
| `GET /predictions/model`, `/predictions/track-record` | Model card; live hit rate of past predictions |
| `GET /risk/{s}`, `/risk/ranking` | Score + component breakdown; universe ranked safest-first |
| `GET /anomalies/{s}`, `/anomalies/recent` | Typed unusual-activity events |
| `GET /sentiment/{s}`, `POST /sentiment/score` | Scored headlines; ad-hoc text scoring |
| `GET /compare?symbols=A,B,C` | Indexed series, metric table, correlation matrix |
| `GET /alerts` | Signals, risk grades and anomalies worth surfacing |
| `POST /admin/ingest`, `POST /admin/train` | Background jobs; poll `/admin/jobs/{name}` |

---

## How the prediction is kept honest

Financial ML is unusually easy to get wrong in ways that look like success. The
guards here are deliberate:

- **Purged walk-forward CV.** Folds only ever move forward in time, and an
  embargo of at least the prediction horizon is cut between train and test — so
  the overlapping forward returns of the last training rows cannot leak into the
  first test rows. A random `KFold` on this data will happily report AUC 0.7.
- **Truncation test.** `tests/test_features.py` removes the last 40 rows,
  recomputes every feature, and asserts nothing changed on the rows that remain.
  Centred windows, `shift(-n)` and full-sample scaling all fail this.
- **Pooled, not per-ticker.** One model over the whole universe (`symbol` is not
  a feature) — ~1,200 rows per ticker is too thin for a boosted tree.
- **Probability calibration, chosen on held-out data.** Raw GBDT scores are not
  probabilities, and a probability is the product. Sigmoid and isotonic are both
  fitted on a slice the model never saw and scored on a further slice; isotonic
  is penalised when it collapses onto a handful of steps (which makes every
  stock show the identical number).
- **Metrics reported, not hidden.** Five-day equity direction is near a coin
  flip. Expect out-of-fold AUC in the **0.50–0.56** band. The dashboard's Model
  page shows AUC, Brier, per-fold results and a non-overlapping signal-vs-hold
  comparison. **An AUC of 0.8 here means a bug, not an edge** — the test suite
  asserts the value stays inside a plausible band for exactly that reason.

---

## How the risk score works

Not a black box: a weighted blend of six components, each mapped from its raw
value onto 0–100 through published bands. The API returns every component's raw
value, scaled value, weight and contribution, and the score equals their sum —
a property the tests assert.

| Component | Weight | Raw measure |
|---|---|---|
| Volatility | 0.28 | Annualised 63-day realized volatility |
| Drawdown | 0.20 | Worst peak-to-trough fall in the lookback |
| Tail risk | 0.18 | Historical 95% CVaR (expected shortfall) |
| Beta | 0.12 | Rolling sensitivity to the benchmark |
| Downside deviation | 0.12 | Standard deviation of losing days only |
| Liquidity | 0.10 | Median dollar volume (more is safer) |

Grades: **A** < 20, **B** < 35, **C** < 50, **D** < 65, **E** < 80, **F** ≥ 80.

---

## Configuration

Everything is environment-driven — see `.env.example`. The switches that matter
most:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | SQLite in `artifacts/` | Postgres in Docker/production |
| `DATA_SOURCE` | `yahoo` | `synthetic` for offline/CI |
| `NEWS_SOURCE` | `auto` | `yahoo` \| `newsapi` \| `finnhub` \| `synthetic` \| `none` |
| `DEFAULT_TICKERS` | 10 large caps | Comma-separated |
| `HORIZON_DAYS` | `5` | Embargo tracks this |
| `MODEL_TYPE` | `lightgbm` | `xgboost`, or `gbdt` (sklearn, no extra deps) |
| `LONG_THRESHOLD` / `SHORT_THRESHOLD` | `0.55` / `0.45` | Where NEUTRAL ends |
| `ARTIFACTS_DIR` | `./artifacts` | Mount as a volume so models survive restarts |

---

## Testing

```bash
make test     # 40 tests: indicators, leakage, ETL, risk, anomalies, training, API
make lint
```

The suite runs entirely on the synthetic source, so it needs no network, no API
keys and no database server.

---

## Operating it

- **Daily**: `stockseer ingest --period 6mo` after the close. Upserts make
  re-runs harmless.
- **Weekly**: `stockseer train`. A five-day-horizon model does not meaningfully
  change from one session's data.
- **Scoring yourself**: `POST /admin/score-predictions` backfills realised
  outcomes so `/predictions/track-record` shows the live hit rate — the number
  that matters more than any backtest.

`.github/workflows/refresh.yml` and the `cron` service in `render.yaml` do
exactly this on a schedule.

---

## Known limits

- **News history is short.** Free news APIs return ~30 days, so sentiment
  features are near-empty for most of a 5-year training set; they earn their
  place only once daily ingestion has been accumulating headlines for a while.
  They are useful in the dashboard from day one.
- **Daily bars only.** No intraday, no order book, no fundamentals.
- **The backtest is a sanity check.** Non-overlapping periods, equal weight, no
  costs, no slippage, no borrow. It answers "is the signal better than nothing?",
  not "would this have made money".
- **Cold cross-universe endpoints take a few seconds** (`/alerts` fits an
  isolation forest per ticker); results are cached for `CACHE_TTL_SECONDS`.

## Not investment advice

StockSeer produces statistical estimates from historical data. Markets are
adversarial and largely efficient; a model that beats a coin flip by a few
percent is a good model, not a money printer. Do not trade on this without
understanding everything above.

MIT licensed.
