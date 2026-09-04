"""Assemble the model matrix.

Design rules that keep this honest:
  * Every feature at row t uses data up to and including t. The only forward-looking
    column is the target, and it is named `target_*` so it can never be mistaken.
  * Sentiment is aggregated to the *daily* level and shifted by one day, because
    a headline published during session t is only reliably tradable at t+1.
  * The benchmark is joined on date and its own features are computed first,
    so a missing benchmark degrades to NaN rather than silently misaligning.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import technical as ta

FEATURE_PREFIX_DROP = ("target_", "date", "symbol", "open", "high", "low",
                       "close", "adj_close", "volume", "fwd_")


def _safe_div(a, b):
    return a / pd.Series(b).replace(0, np.nan).values


def build_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: date, open, high, low, close, adj_close, volume (ascending)."""
    if df.empty:
        return df.copy()

    out = df.copy().sort_values("date").reset_index(drop=True)
    c = out["adj_close"]
    h, lo, o, v = out["high"], out["low"], out["open"], out["volume"]
    # Scale intraday levels onto the adjusted-close basis so ratios stay valid
    # across splits and dividends.
    adj_factor = (c / out["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    h, lo, o = h * adj_factor, lo * adj_factor, o * adj_factor

    ret = np.log(c).diff()
    out["ret_1d"] = ret

    for w in (2, 3, 5, 10, 21, 63):
        out[f"ret_{w}d"] = np.log(c).diff(w)

    # --- trend ---
    for w in (10, 20, 50, 200):
        m = ta.sma(c, w)
        out[f"sma_{w}_ratio"] = c / m - 1.0
    out["sma_10_50_cross"] = ta.sma(c, 10) / ta.sma(c, 50) - 1.0
    out["sma_50_200_cross"] = ta.sma(c, 50) / ta.sma(c, 200) - 1.0
    for w in (12, 26):
        out[f"ema_{w}_ratio"] = c / ta.ema(c, w) - 1.0
    out = out.join(ta.macd(c))
    out["macd_norm"] = out["macd"] / c
    out["macd_hist_norm"] = out["macd_hist"] / c
    out["adx_14"] = ta.adx(h, lo, c, 14)

    # --- momentum ---
    out["rsi_14"] = ta.rsi(c, 14)
    out["rsi_7"] = ta.rsi(c, 7)
    out = out.join(ta.stochastic(h, lo, c))
    out["roc_10"] = ta.roc(c, 10)
    out["williams_r"] = ta.williams_r(h, lo, c, 14)

    # --- volatility ---
    for w in (10, 21, 63):
        out[f"vol_{w}d"] = ta.realized_vol(ret, w)
    out["vol_ratio_10_63"] = out["vol_10d"] / out["vol_63d"].replace(0, np.nan)
    out["parkinson_21"] = ta.parkinson_vol(h, lo, 21)
    out["atr_14"] = ta.atr(h, lo, c, 14)
    out["atr_pct"] = out["atr_14"] / c
    out = out.join(ta.bollinger(c, 20, 2.0)[["bb_width", "bb_pct_b"]])

    # --- volume ---
    out["vol_z_20"] = ta.volume_zscore(v, 20)
    out["dollar_vol"] = c * v
    out["dollar_vol_log"] = np.log1p(out["dollar_vol"])
    out["obv_slope_10"] = ta.obv(c, v).diff(10) / v.rolling(
        20, min_periods=20).mean().replace(0, np.nan)
    out["mfi_14"] = ta.money_flow_index(h, lo, c, v, 14)

    # --- shape / structure ---
    out["gap_pct"] = o / c.shift(1) - 1.0
    out["intraday_range"] = (h - lo) / c
    out["close_position"] = (c - lo) / (h - lo).replace(0, np.nan)
    out["drawdown_252"] = ta.rolling_drawdown(c, 252)
    out["dist_52w_high"] = c / c.rolling(252, min_periods=60).max() - 1.0
    out["dist_52w_low"] = c / c.rolling(252, min_periods=60).min() - 1.0

    # --- higher moments (regime hints) ---
    out["skew_63"] = ret.rolling(63, min_periods=63).skew()
    out["kurt_63"] = ret.rolling(63, min_periods=63).kurt()

    # --- calendar ---
    d = pd.to_datetime(out["date"])
    out["dow"] = d.dt.dayofweek
    out["month"] = d.dt.month
    out["is_month_end"] = (d.dt.is_month_end).astype(int)
    out["is_quarter_end"] = (d.dt.is_quarter_end).astype(int)

    return out


def add_market_features(df: pd.DataFrame, bench: pd.DataFrame,
                        beta_window: int = 60) -> pd.DataFrame:
    """Relative strength and rolling beta against the benchmark."""
    if bench is None or bench.empty:
        for col in ("bench_ret_1d", "bench_ret_5d", "rel_strength_21d",
                    "beta_60d", "corr_60d", "bench_vol_21d"):
            df[col] = np.nan
        return df

    b = bench.sort_values("date").copy()
    b["bench_ret_1d"] = np.log(b["adj_close"]).diff()
    b["bench_ret_5d"] = np.log(b["adj_close"]).diff(5)
    b["bench_ret_21d"] = np.log(b["adj_close"]).diff(21)
    b["bench_vol_21d"] = ta.realized_vol(b["bench_ret_1d"], 21)
    b = b[["date", "bench_ret_1d", "bench_ret_5d", "bench_ret_21d", "bench_vol_21d"]]

    out = df.merge(b, on="date", how="left")
    out["rel_strength_21d"] = out["ret_21d"] - out["bench_ret_21d"]

    cov = out["ret_1d"].rolling(beta_window, min_periods=beta_window).cov(
        out["bench_ret_1d"])
    var = out["bench_ret_1d"].rolling(beta_window, min_periods=beta_window).var()
    out["beta_60d"] = cov / var.replace(0, np.nan)
    out["corr_60d"] = out["ret_1d"].rolling(
        beta_window, min_periods=beta_window).corr(out["bench_ret_1d"])
    return out.drop(columns=["bench_ret_21d"])


def add_sentiment_features(df: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """Daily news sentiment, lagged one session to avoid look-ahead."""
    cols = ["sent_mean_1d", "sent_mean_3d", "sent_mean_7d",
            "news_count_1d", "news_count_7d", "sent_dispersion_7d"]
    if news is None or news.empty or "sentiment" not in news.columns:
        for c in cols:
            df[c] = 0.0 if c.startswith("news_count") else np.nan
        return df

    n = news.copy()
    n["date"] = pd.to_datetime(n["published_at"], utc=True).dt.tz_convert(
        None).dt.normalize()
    daily = n.groupby("date")["sentiment"].agg(["mean", "count", "std"])
    daily.columns = ["sent_mean_1d", "news_count_1d", "sent_std_1d"]

    idx = pd.to_datetime(df["date"]).dt.normalize()
    daily = daily.reindex(pd.DatetimeIndex(idx.unique()).sort_values())
    daily["news_count_1d"] = daily["news_count_1d"].fillna(0.0)
    daily["sent_mean_3d"] = daily["sent_mean_1d"].rolling(3, min_periods=1).mean()
    daily["sent_mean_7d"] = daily["sent_mean_1d"].rolling(7, min_periods=1).mean()
    daily["news_count_7d"] = daily["news_count_1d"].rolling(7, min_periods=1).sum()
    daily["sent_dispersion_7d"] = daily["sent_mean_1d"].rolling(7, min_periods=2).std()

    # Shift by one session: a headline is only actionable on the next open.
    daily = daily.shift(1)

    out = df.copy()
    joined = daily.reindex(idx.values)
    for c in cols:
        out[c] = joined[c].values
    out["news_count_1d"] = out["news_count_1d"].fillna(0.0)
    out["news_count_7d"] = out["news_count_7d"].fillna(0.0)
    return out


def add_target(df: pd.DataFrame, horizon: int = 5,
               threshold: float = 0.0) -> pd.DataFrame:
    """Forward log-return over `horizon` sessions and its sign."""
    out = df.copy()
    fwd = np.log(out["adj_close"]).shift(-horizon) - np.log(out["adj_close"])
    out["fwd_return"] = fwd
    out["target_direction"] = (fwd > threshold).astype("float")
    out.loc[fwd.isna(), "target_direction"] = np.nan
    # Volatility-normalised move — used for the "expected move" readout.
    out["fwd_return_vol_adj"] = fwd / (out["vol_21d"] / np.sqrt(252) *
                                       np.sqrt(horizon)).replace(0, np.nan)
    return out


def build_dataset(prices: pd.DataFrame, symbol: str,
                  benchmark: pd.DataFrame | None = None,
                  news: pd.DataFrame | None = None,
                  horizon: int = 5,
                  with_target: bool = True) -> pd.DataFrame:
    """Full per-symbol feature frame."""
    df = build_price_features(prices)
    if df.empty:
        return df
    df = add_market_features(df, benchmark)
    df = add_sentiment_features(df, news)
    if with_target:
        df = add_target(df, horizon=horizon)
    df.insert(1, "symbol", symbol.upper())
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model inputs = everything that is not an identifier, a raw price, or a target."""
    drop_exact = {"date", "symbol", "open", "high", "low", "close", "adj_close",
                  "volume", "dollar_vol", "bb_mid", "bb_upper", "bb_lower",
                  "macd", "macd_signal", "macd_hist", "atr_14"}
    return [
        c for c in df.columns
        if c not in drop_exact
        and not c.startswith("target_")
        and not c.startswith("fwd_")
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def stack_datasets(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Pool per-symbol frames into one cross-sectional panel, sorted by date."""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True)
    return panel.sort_values(["date", "symbol"]).reset_index(drop=True)
