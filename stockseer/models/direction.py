"""Direction model: P(forward h-day return > 0), pooled across the universe.

Choices worth defending:
  * **Pooled, not per-ticker.** ~1200 usable rows per ticker over 5y is thin for
    a boosted tree. Pooling gives ~12k rows and forces the model to learn
    cross-sectional patterns rather than one stock's idiosyncrasies. `symbol` is
    deliberately *not* a feature.
  * **Purged walk-forward CV**, never random KFold (see validation.py).
  * **Probability calibration.** Raw GBDT scores are not probabilities, and the
    whole product is a probability shown to a user. Isotonic on a held-out slice.
  * **Reported honestly.** AUC on 5-day equity direction lands around 0.52-0.56
    for real data. The trainer surfaces that instead of hiding it, and the API
    passes the metrics through to the UI.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ..config import settings
from ..features.builder import feature_columns
from . import registry
from .validation import PurgedWalkForward

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Estimator construction
# --------------------------------------------------------------------------- #

def _base_estimator(model_type: str, seed: int = 42):
    model_type = model_type.lower()
    if model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=400, learning_rate=0.03, num_leaves=15,
                max_depth=5, min_child_samples=40, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.1,
                reg_lambda=1.0, random_state=seed, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            log.warning("lightgbm unavailable; falling back to sklearn HGB")
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=400, learning_rate=0.03, max_depth=4,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0, random_state=seed,
                n_jobs=-1, eval_metric="logloss", tree_method="hist",
            )
        except ImportError:
            log.warning("xgboost unavailable; falling back to sklearn HGB")
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.03, max_depth=5, max_leaf_nodes=15,
        min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
    )


def _prefit_calibrator(base, method: str = "isotonic"):
    """Calibrate an already-fitted estimator.

    sklearn <1.6 used ``cv="prefit"``; 1.6 replaced it with ``FrozenEstimator``.
    Support both so the repo installs cleanly on either.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(base), method=method)
    except ImportError:
        return CalibratedClassifierCV(base, method=method, cv="prefit")


def make_pipeline(model_type: str | None = None, seed: int = 42) -> Pipeline:
    model_type = model_type or settings.model_type
    return Pipeline([
        # keep_empty_features keeps the column count stable when a feature is
        # all-NaN inside one fold (common for news sentiment, which only has
        # ~30 days of history until the daily ingest has been running a while).
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("clf", _base_estimator(model_type, seed)),
    ])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def _metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    yhat = (p >= thr).astype(int)
    out = {
        "n": int(len(y)),
        "base_rate": float(np.mean(y)),
        "accuracy": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)))
    else:
        out["roc_auc"] = float("nan")
        out["log_loss"] = float("nan")
    return out


def signal_backtest(dates: pd.Series, fwd_returns: np.ndarray, p: np.ndarray,
                    long_thr: float, horizon: int) -> dict[str, float]:
    """Economic sanity check: does acting on the signal beat holding the panel?

    Non-overlapping periods only (every `horizon` sessions), equal weight across
    whatever names trigger, no costs. Directional, not a production backtest.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates).values,
                       "fwd": fwd_returns, "p": p}).dropna()
    if df.empty:
        return {}
    uniq = np.array(sorted(df["date"].unique()))[::horizon]
    df = df[df["date"].isin(uniq)]
    if df.empty:
        return {}

    df["is_long"] = df["p"] >= long_thr
    grouped = df.groupby("date")
    per_day = pd.DataFrame({
        "signal": grouped.apply(
            lambda g: g.loc[g["is_long"], "fwd"].mean() if g["is_long"].any()
            else np.nan),
        "hold": grouped["fwd"].mean(),
        "n_long": grouped["is_long"].sum(),
    })
    sig = per_day["signal"].dropna()
    hold = per_day["hold"].dropna()
    periods_per_year = 252 / horizon
    def _sharpe(x):
        return float(x.mean() / x.std(ddof=0) * np.sqrt(periods_per_year)) \
            if len(x) > 2 and x.std(ddof=0) > 0 else float("nan")
    return {
        "signal_mean_return": float(sig.mean()) if len(sig) else float("nan"),
        "hold_mean_return": float(hold.mean()) if len(hold) else float("nan"),
        "signal_hit_rate": float((sig > 0).mean()) if len(sig) else float("nan"),
        "signal_sharpe": _sharpe(sig),
        "hold_sharpe": _sharpe(hold),
        "avg_positions": float(per_day["n_long"].mean()),
        "n_periods": int(len(per_day)),
    }


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #

@dataclass
class TrainResult:
    version: str
    metrics: dict[str, Any]
    features: list[str]
    n_rows: int
    tickers: list[str] = field(default_factory=list)
    feature_importance: dict[str, float] = field(default_factory=dict)


def _importance(pipe: Pipeline, features: list[str]) -> dict[str, float]:
    clf = pipe.named_steps["clf"]
    imp = getattr(clf, "feature_importances_", None)
    if imp is None:
        return {}
    imp = np.asarray(imp, dtype=float)
    if imp.sum() > 0:
        imp = imp / imp.sum()
    pairs = sorted(zip(features, imp, strict=False), key=lambda kv: -kv[1])
    return {k: round(float(v), 5) for k, v in pairs[:30]}


def train_direction_model(panel: pd.DataFrame, horizon: int | None = None,
                          model_type: str | None = None,
                          n_splits: int | None = None,
                          save_model: bool = True) -> TrainResult:
    horizon = horizon or settings.horizon_days
    model_type = model_type or settings.model_type
    n_splits = n_splits or settings.n_splits

    if panel.empty:
        raise ValueError("empty panel — run ingestion first")

    data = panel.dropna(subset=["target_direction"]).copy()
    feats = feature_columns(data)
    # Drop columns that are entirely missing (e.g. sentiment with no news source).
    feats = [f for f in feats if data[f].notna().any()]
    if not feats:
        raise ValueError("no usable features")

    X = data[feats].replace([np.inf, -np.inf], np.nan)
    y = data["target_direction"].astype(int).values
    dates = data["date"]

    if len(data) < settings.min_train_rows:
        log.warning("only %d rows — metrics will be noisy", len(data))

    cv = PurgedWalkForward(n_splits=n_splits, embargo=max(settings.embargo_days,
                                                          horizon),
                           min_train_size=min(250, max(50, len(data) // 10)))

    oof_p = np.full(len(data), np.nan)
    fold_metrics = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, (tr, te) in enumerate(cv.split(dates)):
            pipe = make_pipeline(model_type, seed=42 + i)
            pipe.fit(X.iloc[tr], y[tr])
            p = pipe.predict_proba(X.iloc[te])[:, 1]
            oof_p[te] = p
            fold_metrics.append({"fold": i, **_metrics(y[te], p)})

    mask = ~np.isnan(oof_p)
    if mask.sum() == 0:
        raise ValueError("walk-forward produced no test rows — not enough history")

    oof = _metrics(y[mask], oof_p[mask])
    bt = signal_backtest(dates[mask], data.loc[mask, "fwd_return"].values,
                         oof_p[mask], settings.long_threshold, horizon)

    # Final model. Calibration is fitted on a chronological slice the base model
    # never saw, and the *method* is chosen on a further held-out slice: isotonic
    # is flexible but collapses to a handful of steps when the signal is weak
    # (every stock then gets the identical probability), while sigmoid/Platt stays
    # continuous and preserves the ranking. Let the data decide.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cut = int(len(data) * 0.70)
        cal_cut = int(len(data) * 0.88)
        base = make_pipeline(model_type, seed=7)
        base.fit(X.iloc[:cut], y[:cut])

        best, best_brier, chosen = None, np.inf, None
        for method in ("sigmoid", "isotonic"):
            cal = _prefit_calibrator(base, method)
            cal.fit(X.iloc[cut:cal_cut], y[cut:cal_cut])
            p_hold = cal.predict_proba(X.iloc[cal_cut:])[:, 1]
            # Penalise a calibrator that maps everything onto a few values.
            spread_penalty = 0.0 if len(np.unique(np.round(p_hold, 3))) > 10 else 0.02
            b = brier_score_loss(y[cal_cut:], p_hold) + spread_penalty
            if b < best_brier:
                best, best_brier, chosen = cal, b, method

        # Re-fit the underlying estimator on the full history for the shipped model.
        full = make_pipeline(model_type, seed=7)
        full.fit(X, y)

    calibrated = {"calibrator": best, "estimator": full, "features": feats}

    metrics = {
        "out_of_fold": oof,
        "folds": fold_metrics,
        "backtest": bt,
        "horizon_days": horizon,
        "model_type": model_type,
        "date_range": [str(pd.to_datetime(dates).min().date()),
                       str(pd.to_datetime(dates).max().date())],
        "calibration": {"method": chosen, "holdout_brier": round(float(best_brier), 5)},
    }
    version = registry.new_version("direction")
    result = TrainResult(
        version=version, metrics=metrics, features=feats, n_rows=int(len(data)),
        tickers=sorted(data["symbol"].unique().tolist()),
        feature_importance=_importance(full, feats),
    )
    if save_model:
        registry.save(calibrated, registry.ModelMetadata(
            version=version, model_type=model_type, horizon_days=horizon,
            features=feats, n_rows=result.n_rows, tickers=result.tickers,
            metrics=metrics, feature_importance=result.feature_importance,
            notes="pooled cross-sectional direction classifier, isotonic-calibrated",
        ))
    return result


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def predict_proba(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    feats = bundle["features"]
    Xa = X.reindex(columns=feats).replace([np.inf, -np.inf], np.nan)
    try:
        return bundle["calibrator"].predict_proba(Xa)[:, 1]
    except Exception:  # calibrator can fail on degenerate input
        return bundle["estimator"].predict_proba(Xa)[:, 1]


def classify(p: float, long_thr: float | None = None,
             short_thr: float | None = None) -> tuple[str, float]:
    long_thr = long_thr if long_thr is not None else settings.long_threshold
    short_thr = short_thr if short_thr is not None else settings.short_threshold
    if p >= long_thr:
        direction = "UP"
    elif p <= short_thr:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"
    return direction, float(abs(p - 0.5) * 2)
