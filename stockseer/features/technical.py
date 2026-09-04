"""Technical indicators, implemented in pandas/numpy only.

No TA-Lib (a C build dependency that breaks slim Docker images) and no
`ta`/`pandas-ta` (both have had API churn). Each function takes and returns
plain Series/DataFrames so they are unit-testable in isolation.

Convention: every indicator uses only information available *at or before* the
bar it is assigned to. No centred windows, no shift(-n) anywhere in this file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #

def sma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window, min_periods=window).mean()


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False, min_periods=span).mean()


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
         ) -> pd.DataFrame:
    line = ema(s, fast) - ema(s, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": line - sig})


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
        ) -> pd.Series:
    up = high.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    plus_di = 100 * pd.Series(plus_dm, index=high.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr_
    minus_di = 100 * pd.Series(minus_dm, index=high.index).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean() / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


# --------------------------------------------------------------------------- #
# Momentum
# --------------------------------------------------------------------------- #

def rsi(s: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI (exponential smoothing, not simple mean)."""
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(100.0).where(avg_loss.notna())


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k: int = 14, d: int = 3) -> pd.DataFrame:
    ll = low.rolling(k, min_periods=k).min()
    hh = high.rolling(k, min_periods=k).max()
    pct_k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return pd.DataFrame({"stoch_k": pct_k,
                         "stoch_d": pct_k.rolling(d, min_periods=d).mean()})


def roc(s: pd.Series, window: int = 10) -> pd.Series:
    return s.pct_change(window) * 100


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               window: int = 14) -> pd.Series:
    hh = high.rolling(window, min_periods=window).max()
    ll = low.rolling(window, min_periods=window).min()
    return -100 * (hh - close) / (hh - ll).replace(0, np.nan)


# --------------------------------------------------------------------------- #
# Volatility
# --------------------------------------------------------------------------- #

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()],
                     axis=1).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14
        ) -> pd.Series:
    return true_range(high, low, close).ewm(
        alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger(s: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    mid = sma(s, window)
    sd = s.rolling(window, min_periods=window).std(ddof=0)
    upper, lower = mid + n_std * sd, mid - n_std * sd
    width = (upper - lower) / mid.replace(0, np.nan)
    pct_b = (s - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower,
                         "bb_width": width, "bb_pct_b": pct_b})


def realized_vol(returns: pd.Series, window: int = 21,
                 annualize: bool = True) -> pd.Series:
    v = returns.rolling(window, min_periods=window).std(ddof=0)
    return v * np.sqrt(TRADING_DAYS) if annualize else v


def parkinson_vol(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    """Range-based estimator — uses intraday info, ~5x more efficient than close-to-close."""
    hl = np.log(high / low) ** 2
    return np.sqrt(hl.rolling(window, min_periods=window).mean()
                   / (4 * np.log(2)) * TRADING_DAYS)


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    lv = np.log1p(volume)
    mu = lv.rolling(window, min_periods=window).mean()
    sd = lv.rolling(window, min_periods=window).std(ddof=0)
    return (lv - mu) / sd.replace(0, np.nan)


def money_flow_index(high, low, close, volume, window: int = 14) -> pd.Series:
    tp = (high + low + close) / 3
    mf = tp * volume
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(window, min_periods=window).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(window, min_periods=window).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


# --------------------------------------------------------------------------- #
# Drawdown
# --------------------------------------------------------------------------- #

def rolling_drawdown(close: pd.Series, window: int = TRADING_DAYS) -> pd.Series:
    peak = close.rolling(window, min_periods=20).max()
    return close / peak - 1.0


def max_drawdown(close: pd.Series) -> float:
    if close.empty:
        return float("nan")
    dd = close / close.cummax() - 1.0
    return float(dd.min())
