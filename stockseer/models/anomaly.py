"""Unusual price / volume movement detection.

Two detectors, combined:
  * **Rule-based z-scores** — interpretable, and they give the event a *type*
    ("volume surge", "gap") that a user can act on.
  * **IsolationForest** on the joint feature vector — catches combinations that
    look unremarkable one dimension at a time (a modest move on modest volume
    but with an abnormal intraday range and a gap).

Both are fit on a trailing window so "unusual" means unusual *for this stock,
recently* rather than unusual versus the whole market.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ..config import settings

KINDS = ("PRICE_SPIKE", "PRICE_DROP", "VOLUME_SURGE", "GAP", "VOLATILITY_SHIFT",
         "MULTIVARIATE")


@dataclass
class Anomaly:
    date: str
    kind: str
    severity: float      # 0..1
    score: float         # raw detector output (z or -iforest score)
    detail: str
    close: float
    volume: float
    return_pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def _z(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=max(10, window // 3)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 3)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def _severity_from_z(z: float, thr: float) -> float:
    return float(np.clip((abs(z) - thr) / (3 * thr) + 0.34, 0.0, 1.0))


def detect_anomalies(prices: pd.DataFrame, window: int = 60,
                     z_threshold: float | None = None,
                     contamination: float | None = None,
                     lookback_days: int = 180) -> list[Anomaly]:
    z_threshold = z_threshold or settings.anomaly_z_threshold
    contamination = contamination or settings.anomaly_contamination

    df = prices.sort_values("date").copy()
    if len(df) < window + 20:
        return []

    c, v, h, lo, o = (df["adj_close"], df["volume"], df["high"],
                      df["low"], df["open"])
    ret = np.log(c).diff()
    df["ret"] = ret
    df["ret_z"] = _z(ret, window)
    df["logvol"] = np.log1p(v)
    df["vol_z"] = _z(df["logvol"], window)
    df["gap"] = o / c.shift(1) - 1.0
    df["gap_z"] = _z(df["gap"], window)
    df["range_pct"] = (h - lo) / c
    df["range_z"] = _z(df["range_pct"], window)
    df["vol21"] = ret.rolling(21, min_periods=21).std(ddof=0)
    df["vol_regime_z"] = _z(df["vol21"], window)

    recent = df.tail(lookback_days).copy()
    out: list[Anomaly] = []

    def _add(row, kind, z, detail):
        out.append(Anomaly(
            date=str(pd.to_datetime(row["date"]).date()), kind=kind,
            severity=round(_severity_from_z(z, z_threshold), 3),
            score=round(float(z), 3), detail=detail,
            close=round(float(row["adj_close"]), 4),
            volume=float(row["volume"]),
            return_pct=round(float(row["ret"] * 100), 3)
            if np.isfinite(row["ret"]) else 0.0,
        ))

    for _, r in recent.iterrows():
        if np.isfinite(r["ret_z"]) and abs(r["ret_z"]) >= z_threshold:
            kind = "PRICE_SPIKE" if r["ret"] > 0 else "PRICE_DROP"
            _add(r, kind, r["ret_z"],
                 f"{r['ret'] * 100:+.2f}% move, {abs(r['ret_z']):.1f}σ vs its own "
                 f"{window}-day distribution")
        if np.isfinite(r["vol_z"]) and r["vol_z"] >= z_threshold:
            _add(r, "VOLUME_SURGE", r["vol_z"],
                 f"volume {r['volume']:,.0f}, {r['vol_z']:.1f}σ above its "
                 f"{window}-day norm")
        if np.isfinite(r["gap_z"]) and abs(r["gap_z"]) >= z_threshold:
            _add(r, "GAP", r["gap_z"],
                 f"opened {r['gap'] * 100:+.2f}% away from the prior close")
        if np.isfinite(r["vol_regime_z"]) and r["vol_regime_z"] >= z_threshold:
            _add(r, "VOLATILITY_SHIFT", r["vol_regime_z"],
                 f"21-day realised volatility jumped {r['vol_regime_z']:.1f}σ — "
                 f"regime change")

    # --- multivariate ---
    cols = ["ret_z", "vol_z", "gap_z", "range_z"]
    fit = df[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(fit) >= 100:
        iso = IsolationForest(n_estimators=200, contamination=contamination,
                              random_state=42, n_jobs=-1)
        iso.fit(fit)
        recent_fit = recent[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(recent_fit):
            scores = -iso.score_samples(recent_fit)   # higher = more anomalous
            flags = iso.predict(recent_fit) == -1
            s_min, s_max = float(scores.min()), float(scores.max())
            already = {(a.date) for a in out}
            for idx, sc, fl in zip(recent_fit.index, scores, flags, strict=False):
                if not fl:
                    continue
                row = df.loc[idx]
                d = str(pd.to_datetime(row["date"]).date())
                if d in already:
                    continue
                sev = (sc - s_min) / (s_max - s_min) if s_max > s_min else 0.5
                out.append(Anomaly(
                    date=d, kind="MULTIVARIATE", severity=round(float(sev), 3),
                    score=round(float(sc), 4),
                    detail="joint price/volume/gap/range pattern is an outlier "
                           "even though no single measure is extreme",
                    close=round(float(row["adj_close"]), 4),
                    volume=float(row["volume"]),
                    return_pct=round(float(row["ret"] * 100), 3),
                ))

    out.sort(key=lambda a: (a.date, -a.severity), reverse=True)
    return out


def summarize(anomalies: list[Anomaly]) -> dict:
    if not anomalies:
        return {"count": 0, "by_kind": {}, "max_severity": 0.0, "latest": None}
    by_kind: dict[str, int] = {}
    for a in anomalies:
        by_kind[a.kind] = by_kind.get(a.kind, 0) + 1
    return {
        "count": len(anomalies),
        "by_kind": by_kind,
        "max_severity": max(a.severity for a in anomalies),
        "latest": anomalies[0].as_dict(),
    }
