"""Risk model — a transparent, auditable 0–100 score.

Deliberately *not* a black box. A user who is told "risk 78/100" needs to know
why, so the score is a weighted blend of six named components, each mapped from
its raw value onto 0–100 through published bands. Every component is returned
alongside the total.

Higher score = more risk.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..config import settings
from ..features.technical import TRADING_DAYS, max_drawdown

# component -> (weight, band_low, band_high, higher_is_riskier)
# band_low maps to 0, band_high maps to 100, linear in between, clipped.
COMPONENTS = {
    "volatility":   (0.28, 0.12, 0.75, True),    # annualised 63d vol
    "drawdown":     (0.20, 0.05, 0.60, True),    # |max drawdown| over lookback
    "tail_risk":    (0.18, 0.015, 0.075, True),  # |CVaR 95%| daily
    "beta":         (0.12, 0.4, 2.0, True),      # market sensitivity
    "downside_dev": (0.12, 0.008, 0.045, True),  # daily downside deviation
    "liquidity":    (0.10, 1e5, 5e8, False),     # median $ volume (more = safer)
}

GRADE_BANDS = [(20, "A"), (35, "B"), (50, "C"), (65, "D"), (80, "E"), (101, "F")]


@dataclass
class RiskComponent:
    name: str
    raw: float
    scaled: float      # 0..100
    weight: float
    contribution: float


@dataclass
class RiskAssessment:
    symbol: str
    as_of: str
    risk_score: float
    grade: str
    annualized_vol: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float | None
    sharpe: float
    sortino: float
    downside_deviation: float
    median_dollar_volume: float
    components: list[RiskComponent] = field(default_factory=list)
    narrative: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["components"] = [asdict(c) for c in self.components]
        return d

    def components_json(self) -> str:
        return json.dumps([asdict(c) for c in self.components])


def _scale(value: float, lo: float, hi: float, higher_is_riskier: bool,
           log_scale: bool = False) -> float:
    if value is None or not np.isfinite(value):
        return 50.0
    if log_scale:
        value, lo, hi = np.log10(max(value, 1e-9)), np.log10(lo), np.log10(hi)
    pct = (value - lo) / (hi - lo) if hi != lo else 0.5
    pct = float(np.clip(pct, 0.0, 1.0))
    return 100 * pct if higher_is_riskier else 100 * (1 - pct)


def _grade(score: float) -> str:
    for threshold, letter in GRADE_BANDS:
        if score < threshold:
            return letter
    return "F"


def historical_var(returns: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Historical (non-parametric) VaR and CVaR as positive loss magnitudes."""
    r = returns[np.isfinite(returns)]
    if len(r) < 30:
        return float("nan"), float("nan")
    q = np.quantile(r, 1 - confidence)
    tail = r[r <= q]
    return float(-q), float(-tail.mean()) if len(tail) else float(-q)


def compute_risk(prices: pd.DataFrame, symbol: str,
                 benchmark: pd.DataFrame | None = None,
                 lookback: int | None = None,
                 risk_free_annual: float = 0.04) -> RiskAssessment:
    lookback = lookback or settings.risk_lookback
    df = prices.sort_values("date").tail(max(lookback, 60)).copy()
    if len(df) < 30:
        raise ValueError(f"{symbol}: need >= 30 price rows for a risk assessment")

    c = df["adj_close"]
    ret = np.log(c).diff().dropna()
    r = ret.values

    ann_vol = float(np.std(r, ddof=0) * np.sqrt(TRADING_DAYS))
    mdd = abs(max_drawdown(c))
    var95, cvar95 = historical_var(r, settings.var_confidence)

    downside = r[r < 0]
    dd_dev = float(np.std(downside, ddof=0)) if len(downside) > 5 else float("nan")
    mean_daily = float(np.mean(r))
    rf_daily = risk_free_annual / TRADING_DAYS
    sharpe = float((mean_daily - rf_daily) / np.std(r, ddof=0)
                   * np.sqrt(TRADING_DAYS)) if np.std(r) > 0 else float("nan")
    sortino = float((mean_daily - rf_daily) / dd_dev * np.sqrt(TRADING_DAYS)) \
        if dd_dev and np.isfinite(dd_dev) and dd_dev > 0 else float("nan")

    beta = None
    if benchmark is not None and not benchmark.empty:
        b = benchmark.sort_values("date").copy()
        merged = df[["date", "adj_close"]].merge(
            b[["date", "adj_close"]], on="date", suffixes=("", "_b")).dropna()
        if len(merged) > 40:
            ra = np.log(merged["adj_close"]).diff().dropna().values
            rb = np.log(merged["adj_close_b"]).diff().dropna().values
            n = min(len(ra), len(rb))
            if n > 40 and np.var(rb[-n:]) > 0:
                beta = float(np.cov(ra[-n:], rb[-n:])[0, 1] / np.var(rb[-n:]))

    med_dollar_vol = float((df["adj_close"] * df["volume"]).median())

    raws = {
        "volatility": ann_vol,
        "drawdown": mdd,
        "tail_risk": cvar95,
        "beta": abs(beta) if beta is not None else 1.0,
        "downside_dev": dd_dev,
        "liquidity": med_dollar_vol,
    }

    comps: list[RiskComponent] = []
    total, weight_used = 0.0, 0.0
    for name, (w, lo, hi, higher) in COMPONENTS.items():
        scaled = _scale(raws[name], lo, hi, higher, log_scale=(name == "liquidity"))
        comps.append(RiskComponent(name, float(raws[name]) if np.isfinite(
            raws[name]) else float("nan"), round(scaled, 2), w, round(w * scaled, 2)))
        total += w * scaled
        weight_used += w
    score = round(total / weight_used, 2)

    assessment = RiskAssessment(
        symbol=symbol.upper(),
        as_of=str(pd.to_datetime(df["date"].iloc[-1]).date()),
        risk_score=score, grade=_grade(score),
        annualized_vol=round(ann_vol, 4), max_drawdown=round(mdd, 4),
        var_95=round(var95, 4) if np.isfinite(var95) else float("nan"),
        cvar_95=round(cvar95, 4) if np.isfinite(cvar95) else float("nan"),
        beta=round(beta, 3) if beta is not None else None,
        sharpe=round(sharpe, 3) if np.isfinite(sharpe) else float("nan"),
        sortino=round(sortino, 3) if np.isfinite(sortino) else float("nan"),
        downside_deviation=round(dd_dev, 5) if np.isfinite(dd_dev) else float("nan"),
        median_dollar_volume=round(med_dollar_vol, 0),
        components=comps,
    )
    assessment.narrative = _narrative(assessment)
    return assessment


def _narrative(a: RiskAssessment) -> str:
    top = sorted(a.components, key=lambda c: -c.contribution)[:2]
    drivers = " and ".join(c.name.replace("_", " ") for c in top)
    tone = {"A": "low", "B": "low-to-moderate", "C": "moderate",
            "D": "elevated", "E": "high", "F": "very high"}[a.grade]
    var_txt = (f"a typical bad day (95th percentile) loses about "
               f"{a.var_95 * 100:.1f}%" if np.isfinite(a.var_95) else "")
    return (f"{a.symbol} scores {a.risk_score:.0f}/100 ({a.grade}) — {tone} risk. "
            f"Annualised volatility is {a.annualized_vol * 100:.0f}% and the worst "
            f"peak-to-trough fall in the window was {a.max_drawdown * 100:.0f}%. "
            f"{var_txt}. The score is driven mainly by {drivers}.").replace(" .", ".")


def rank_by_risk(assessments: list[RiskAssessment]) -> list[dict]:
    rows = [{"symbol": a.symbol, "risk_score": a.risk_score, "grade": a.grade,
             "annualized_vol": a.annualized_vol, "max_drawdown": a.max_drawdown,
             "sharpe": a.sharpe, "beta": a.beta} for a in assessments]
    return sorted(rows, key=lambda r: r["risk_score"])
