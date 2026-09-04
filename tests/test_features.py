"""Indicator correctness and — more importantly — absence of look-ahead."""
import numpy as np
import pandas as pd
import pytest

from stockseer.features import technical as ta
from stockseer.features.builder import (
    add_target,
    build_price_features,
    feature_columns,
)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 80, dtype=float))
    r = ta.rsi(up, 14).dropna()
    assert (r >= 0).all() and (r <= 100).all()
    assert r.iloc[-1] > 95, "a monotonically rising series must be extremely overbought"

    down = pd.Series(np.arange(80, 1, -1, dtype=float))
    assert ta.rsi(down, 14).dropna().iloc[-1] < 5


def test_sma_matches_manual_mean():
    s = pd.Series(np.arange(1, 51, dtype=float))
    assert ta.sma(s, 10).iloc[-1] == pytest.approx(np.mean(np.arange(41, 51)))


def test_macd_histogram_is_line_minus_signal():
    s = pd.Series(np.cumsum(np.random.default_rng(0).standard_normal(300)) + 100)
    m = ta.macd(s)
    valid = m.dropna()
    assert np.allclose(valid["macd_hist"], valid["macd"] - valid["macd_signal"])


def test_bollinger_pct_b_within_band():
    s = pd.Series(np.cumsum(np.random.default_rng(1).standard_normal(300)) + 100)
    b = ta.bollinger(s, 20, 2.0).dropna()
    assert (b["bb_upper"] >= b["bb_mid"]).all()
    assert (b["bb_lower"] <= b["bb_mid"]).all()


def test_atr_is_non_negative(prices):
    a = ta.atr(prices["high"], prices["low"], prices["close"]).dropna()
    assert (a >= 0).all()


def test_max_drawdown_sign_and_bounds(prices):
    mdd = ta.max_drawdown(prices["adj_close"])
    assert -1.0 <= mdd <= 0.0


def test_features_use_no_future_information(prices):
    """Truncating the series must not change any feature on the rows that remain.

    This is the property that a look-ahead bug violates, and it catches
    centred windows, shift(-n), and full-sample scaling in one assertion.
    """
    full = build_price_features(prices)
    truncated = build_price_features(prices.iloc[:-40].copy())
    cols = [c for c in feature_columns(full) if c in truncated.columns]

    a = full.iloc[:len(truncated)][cols].reset_index(drop=True)
    b = truncated[cols].reset_index(drop=True)
    both = a.notna() & b.notna()
    diff = (a[both] - b[both]).abs().max().max()
    assert diff < 1e-9, "a feature changed when future rows were removed"


def test_target_is_forward_looking_by_exactly_the_horizon(prices):
    df = add_target(build_price_features(prices), horizon=5)
    c = df["adj_close"]
    expected = np.log(c.shift(-5)) - np.log(c)
    assert np.allclose(df["fwd_return"].dropna(), expected.dropna())
    assert df["target_direction"].iloc[-5:].isna().all(), \
        "the last `horizon` rows cannot have a known outcome"


def test_feature_columns_exclude_targets_and_raw_prices(prices):
    df = add_target(build_price_features(prices), horizon=5)
    feats = feature_columns(df)
    assert not any(f.startswith(("target_", "fwd_")) for f in feats)
    for banned in ("close", "adj_close", "open", "high", "low", "volume"):
        assert banned not in feats
    assert len(feats) > 30
