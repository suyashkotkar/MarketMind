"""Service layer: the only place that knows how DB rows become insights.

Routers stay thin; the CLI and the API share this code so a scheduled job and an
HTTP request can never drift apart.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import settings
from ...data.pipeline import list_symbols, load_news, load_prices
from ...db.models import AnomalyEvent, ModelRun, Prediction, RiskSnapshot, Ticker
from ...features.builder import build_dataset
from ...models import anomaly as anomaly_mod
from ...models import registry
from ...models.direction import classify, predict_proba, train_direction_model
from ...models.risk import compute_risk
from ..cache import cached

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #

@cached("bench")
def _benchmark(db: Session) -> pd.DataFrame:
    """Benchmark prices are read once per symbol per cross-universe sweep;
    without this the ranking and alert endpoints re-query them N times."""
    return load_prices(db, settings.benchmark_ticker)


def build_symbol_frame(db: Session, symbol: str, horizon: int | None = None,
                       benchmark: pd.DataFrame | None = None,
                       with_target: bool = True) -> pd.DataFrame:
    horizon = horizon or settings.horizon_days
    prices = load_prices(db, symbol)
    if prices.empty:
        return pd.DataFrame()
    bench = benchmark if benchmark is not None else _benchmark(db)
    news = load_news(db, symbol, limit=500)
    return build_dataset(prices, symbol, benchmark=bench, news=news,
                         horizon=horizon, with_target=with_target)


def build_panel(db: Session, symbols: list[str] | None = None,
                horizon: int | None = None) -> pd.DataFrame:
    symbols = symbols or [s for s in list_symbols(db)
                          if s != settings.benchmark_ticker.upper()]
    bench = _benchmark(db)
    frames = []
    for s in symbols:
        f = build_symbol_frame(db, s, horizon=horizon, benchmark=bench)
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["date", "symbol"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def train(db: Session, symbols: list[str] | None = None,
          horizon: int | None = None, model_type: str | None = None) -> dict:
    panel = build_panel(db, symbols, horizon)
    if panel.empty:
        raise ValueError("no data — run ingestion first")
    res = train_direction_model(panel, horizon=horizon, model_type=model_type)
    db.add(ModelRun(
        version=res.version, model_type=model_type or settings.model_type,
        horizon_days=horizon or settings.horizon_days, n_rows=res.n_rows,
        n_features=len(res.features), tickers=json.dumps(res.tickers),
        metrics_json=json.dumps(res.metrics, default=str),
    ))
    db.commit()
    return {
        "version": res.version, "n_rows": res.n_rows,
        "n_features": len(res.features), "tickers": res.tickers,
        "metrics": res.metrics, "feature_importance": res.feature_importance,
    }


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #

def _expected_move(row: pd.Series, horizon: int, prob_up: float) -> float | None:
    vol = row.get("vol_21d")
    if vol is None or not np.isfinite(vol):
        return None
    sigma_h = float(vol) / np.sqrt(252) * np.sqrt(horizon)
    # Centre the expected move on the model's edge over a coin flip.
    return float((prob_up - 0.5) * 2 * sigma_h * 100)


def predict_symbol(db: Session, symbol: str, persist: bool = True,
                   version: str | None = None) -> dict:
    bundle, meta = registry.load(version)
    if bundle is None:
        raise FileNotFoundError("no trained model found — POST /admin/train first")

    horizon = meta.horizon_days
    frame = build_symbol_frame(db, symbol, horizon=horizon, with_target=False)
    if frame.empty:
        raise ValueError(f"{symbol}: no price history in the warehouse")

    feats = meta.features
    X = frame.reindex(columns=feats)
    valid = X.notna().sum(axis=1) >= max(1, int(len(feats) * 0.6))
    if not valid.any():
        raise ValueError(f"{symbol}: not enough history to compute features")
    last_idx = frame.index[valid][-1]
    p = float(predict_proba(bundle, X.loc[[last_idx]])[0])
    direction, confidence = classify(p)
    row = frame.loc[last_idx]
    as_of = pd.to_datetime(row["date"]).date()

    result = {
        "symbol": symbol.upper(),
        "as_of": as_of.isoformat(),
        "horizon_days": horizon,
        "prob_up": round(p, 4),
        "direction": direction,
        "confidence": round(confidence, 4),
        "expected_move_pct": _expected_move(row, horizon, p),
        "model_version": meta.version,
        "model_metrics": meta.metrics.get("out_of_fold", {}),
        "top_features": dict(list(meta.feature_importance.items())[:8]),
        "latest_close": round(float(row["adj_close"]), 4),
        "disclaimer": "Probabilistic estimate from historical patterns. "
                      "Not investment advice.",
    }

    if persist:
        t = db.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
        if t:
            existing = db.scalar(select(Prediction).where(
                Prediction.ticker_id == t.id, Prediction.as_of == as_of,
                Prediction.horizon_days == horizon))
            payload = {"prob_up": p, "direction": direction,
                       "confidence": confidence,
                       "expected_move_pct": result["expected_move_pct"],
                       "model_version": meta.version}
            if existing:
                for k, v in payload.items():
                    setattr(existing, k, v)
            else:
                db.add(Prediction(ticker_id=t.id, as_of=as_of,
                                  horizon_days=horizon, **payload))
            db.commit()
    return result


def predict_universe(db: Session, symbols: list[str] | None = None) -> list[dict]:
    symbols = symbols or [s for s in list_symbols(db)
                          if s != settings.benchmark_ticker.upper()]
    out = []
    for s in symbols:
        try:
            out.append(predict_symbol(db, s))
        except Exception as exc:
            log.warning("prediction failed for %s: %s", s, exc)
            out.append({"symbol": s, "error": str(exc)})
    return out


def score_past_predictions(db: Session) -> dict:
    """Backfill realised outcomes so the dashboard can show live hit rate."""
    scored = 0
    for pred in db.scalars(select(Prediction).where(Prediction.realized_return.is_(None))):
        t = db.get(Ticker, pred.ticker_id)
        prices = load_prices(db, t.symbol)
        if prices.empty:
            continue
        prices = prices.set_index(pd.to_datetime(prices["date"]).dt.date)
        if pred.as_of not in prices.index:
            continue
        pos = prices.index.get_loc(pred.as_of)
        tgt = pos + pred.horizon_days
        if tgt >= len(prices):
            continue
        r = float(np.log(prices["adj_close"].iloc[tgt] /
                         prices["adj_close"].iloc[pos]))
        pred.realized_return = r
        pred.was_correct = bool((r > 0) == (pred.prob_up > 0.5))
        scored += 1
    db.commit()
    return {"scored": scored}


def prediction_track_record(db: Session, symbol: str | None = None) -> dict:
    q = select(Prediction).where(Prediction.was_correct.isnot(None))
    if symbol:
        t = db.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
        if not t:
            return {"n": 0}
        q = q.where(Prediction.ticker_id == t.id)
    rows = list(db.scalars(q))
    if not rows:
        return {"n": 0, "hit_rate": None, "mean_realized_return": None}
    hits = sum(1 for r in rows if r.was_correct)
    return {
        "n": len(rows),
        "hit_rate": round(hits / len(rows), 4),
        "mean_realized_return": round(
            float(np.mean([r.realized_return for r in rows])), 5),
    }


# --------------------------------------------------------------------------- #
# Risk + anomalies
# --------------------------------------------------------------------------- #

@cached("risk")
def risk_for(db: Session, symbol: str, persist: bool = True) -> dict:
    prices = load_prices(db, symbol)
    if prices.empty:
        raise ValueError(f"{symbol}: no price history")
    assessment = compute_risk(prices, symbol, benchmark=_benchmark(db))
    if persist:
        t = db.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
        if t:
            as_of = dt.date.fromisoformat(assessment.as_of)
            snap = db.scalar(select(RiskSnapshot).where(
                RiskSnapshot.ticker_id == t.id, RiskSnapshot.as_of == as_of))
            payload = {
                "risk_score": assessment.risk_score, "grade": assessment.grade,
                "annualized_vol": assessment.annualized_vol,
                "max_drawdown": assessment.max_drawdown,
                "var_95": assessment.var_95, "cvar_95": assessment.cvar_95,
                "beta": assessment.beta, "sharpe": assessment.sharpe,
                "sortino": assessment.sortino,
                "components_json": assessment.components_json(),
            }
            if snap:
                for k, v in payload.items():
                    setattr(snap, k, v)
            else:
                db.add(RiskSnapshot(ticker_id=t.id, as_of=as_of, **payload))
            db.commit()
    return assessment.as_dict()


@cached("anomalies")
def anomalies_for(db: Session, symbol: str, lookback_days: int = 180,
                  persist: bool = True) -> dict:
    prices = load_prices(db, symbol)
    if prices.empty:
        raise ValueError(f"{symbol}: no price history")
    events = anomaly_mod.detect_anomalies(prices, lookback_days=lookback_days)
    if persist and events:
        t = db.scalar(select(Ticker).where(Ticker.symbol == symbol.upper()))
        if t:
            have = {(e.date, e.kind) for e in db.scalars(
                select(AnomalyEvent).where(AnomalyEvent.ticker_id == t.id))}
            for a in events:
                key = (dt.date.fromisoformat(a.date), a.kind)
                if key in have:
                    continue
                db.add(AnomalyEvent(ticker_id=t.id, date=key[0], kind=a.kind,
                                    severity=a.severity, score=a.score,
                                    detail=a.detail))
            db.commit()
    return {"symbol": symbol.upper(),
            "events": [a.as_dict() for a in events],
            "summary": anomaly_mod.summarize(events)}


# --------------------------------------------------------------------------- #
# Comparison + overview
# --------------------------------------------------------------------------- #

def compare(db: Session, symbols: list[str]) -> dict:
    bench = _benchmark(db)
    rows, series = [], {}
    for s in symbols:
        prices = load_prices(db, s)
        if prices.empty:
            rows.append({"symbol": s.upper(), "error": "no data"})
            continue
        try:
            risk = compute_risk(prices, s, benchmark=bench)
        except ValueError as exc:
            rows.append({"symbol": s.upper(), "error": str(exc)})
            continue
        try:
            pred = predict_symbol(db, s, persist=False)
        except Exception:
            pred = {}
        c = prices["adj_close"]
        norm = (c / c.iloc[0] * 100).round(3)
        series[s.upper()] = {
            "dates": [d.strftime("%Y-%m-%d") for d in prices["date"]],
            "normalized": norm.tolist(),
        }
        rows.append({
            "symbol": s.upper(),
            "latest_close": round(float(c.iloc[-1]), 4),
            "return_1m": _pct_change(c, 21), "return_3m": _pct_change(c, 63),
            "return_6m": _pct_change(c, 126), "return_1y": _pct_change(c, 252),
            "risk_score": risk.risk_score, "grade": risk.grade,
            "annualized_vol": risk.annualized_vol,
            "max_drawdown": risk.max_drawdown, "sharpe": risk.sharpe,
            "beta": risk.beta,
            "prob_up": pred.get("prob_up"), "direction": pred.get("direction"),
        })

    # Correlation of daily returns across the compared names.
    closes = {}
    for s in symbols:
        p = load_prices(db, s)
        if not p.empty:
            closes[s.upper()] = p.set_index("date")["adj_close"]
    corr = {}
    if len(closes) > 1:
        cdf = pd.DataFrame(closes).sort_index()
        rets = np.log(cdf).diff().dropna(how="all")
        corr = rets.corr().round(3).fillna(0).to_dict()

    return {"rows": rows, "series": series, "correlation": corr}


def _pct_change(c: pd.Series, periods: int) -> float | None:
    if len(c) <= periods:
        return None
    return round(float(c.iloc[-1] / c.iloc[-1 - periods] - 1.0), 4)


def _universe(db: Session) -> list[str]:
    return [s for s in list_symbols(db) if s != settings.benchmark_ticker.upper()]


@cached("risk_ranking")
def risk_ranking(db: Session, limit: int = 50) -> list[dict]:
    rows = []
    for s in _universe(db):
        try:
            r = risk_for(db, s, persist=False)
        except ValueError:
            continue
        rows.append({k: r[k] for k in ("symbol", "risk_score", "grade",
                                       "annualized_vol", "max_drawdown", "sharpe",
                                       "beta", "var_95")})
    return sorted(rows, key=lambda r: r["risk_score"])[:limit]


@cached("recent_anomalies")
def recent_anomalies(db: Session, days: int = 30,
                     min_severity: float = 0.3) -> list[dict]:
    out = []
    for s in _universe(db):
        try:
            res = anomalies_for(db, s, lookback_days=days, persist=False)
        except ValueError:
            continue
        out += [{"symbol": s, **e} for e in res["events"]
                if e["severity"] >= min_severity]
    return sorted(out, key=lambda e: (e["date"], e["severity"]), reverse=True)


@cached("alerts")
def build_alerts(db: Session, days: int = 7,
                 min_confidence: float = 0.2) -> list[dict]:
    """Turn model output into things worth interrupting someone for."""
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    out: list[dict] = []

    for s in _universe(db):
        try:
            p = predict_symbol(db, s, persist=False)
            if p["direction"] != "NEUTRAL" and p["confidence"] >= min_confidence:
                out.append({
                    "symbol": s, "kind": "SIGNAL",
                    "level": "warning" if p["confidence"] > 0.4 else "info",
                    "message": (f"{s}: model leans {p['direction']} over the next "
                                f"{p['horizon_days']} sessions "
                                f"(p(up)={p['prob_up']:.2f})"),
                    "date": p["as_of"],
                    "meta": {"prob_up": p["prob_up"], "confidence": p["confidence"],
                             "expected_move_pct": p["expected_move_pct"]},
                })
        except Exception:
            pass

        try:
            r = risk_for(db, s, persist=False)
            if r["grade"] in ("E", "F"):
                out.append({
                    "symbol": s, "kind": "RISK", "level": "critical",
                    "message": (f"{s}: risk grade {r['grade']} "
                                f"({r['risk_score']:.0f}/100) — "
                                f"{r['annualized_vol'] * 100:.0f}% annualised vol, "
                                f"{r['max_drawdown'] * 100:.0f}% max drawdown"),
                    "date": r["as_of"],
                    "meta": {"risk_score": r["risk_score"], "grade": r["grade"]},
                })
        except Exception:
            pass

        try:
            a = anomalies_for(db, s, lookback_days=days, persist=False)
            for e in a["events"][:3]:
                if e["date"] >= cutoff and e["severity"] >= 0.5:
                    out.append({
                        "symbol": s, "kind": "ANOMALY",
                        "level": "critical" if e["severity"] > 0.75 else "warning",
                        "message": (f"{s}: {e['kind'].replace('_', ' ').lower()} — "
                                    f"{e['detail']}"),
                        "date": e["date"],
                        "meta": {"severity": e["severity"], "kind": e["kind"]},
                    })
        except Exception:
            pass

    order = {"critical": 0, "warning": 1, "info": 2}
    return sorted(out, key=lambda a: (order[a["level"]], a["date"]))


@cached("overview")
def overview(db: Session, limit: int = 25) -> dict:
    symbols = _universe(db)
    _, meta = registry.load()
    bench = _benchmark(db)
    cards = []
    for s in symbols[:limit]:
        prices = load_prices(db, s)
        if prices.empty:
            continue
        c = prices["adj_close"]
        card: dict[str, Any] = {
            "symbol": s,
            "latest_close": round(float(c.iloc[-1]), 4),
            "change_1d": _pct_change(c, 1),
            "change_1m": _pct_change(c, 21),
        }
        try:
            risk = compute_risk(prices, s, benchmark=bench)
            card["risk_score"] = risk.risk_score
            card["grade"] = risk.grade
        except ValueError:
            pass
        if meta is not None:
            try:
                p = predict_symbol(db, s, persist=False)
                card["prob_up"] = p["prob_up"]
                card["direction"] = p["direction"]
            except Exception:
                pass
        cards.append(card)
    return {
        "as_of": dt.date.today().isoformat(),
        "model_version": meta.version if meta else None,
        "universe_size": len(symbols),
        "cards": cards,
    }
