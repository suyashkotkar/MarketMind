import numpy as np
import pandas as pd
import pytest

from stockseer.data.cleaning import clean_prices
from stockseer.data.pipeline import ingest_universe, load_prices
from stockseer.models.anomaly import detect_anomalies
from stockseer.models.risk import compute_risk, historical_var
from stockseer.models.validation import PurgedWalkForward


# --------------------------------------------------------------------- ETL --
def test_cleaning_drops_bad_rows_and_dedupes():
    raw = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
        "open": [10, 10, 11, np.nan], "high": [9, 9, 12, 12],
        "low": [11, 11, 10, 10], "close": [10.5, 10.6, 11.5, 11.0],
        "adj_close": [10.5, 10.6, 11.5, np.nan], "volume": [100, 100, 200, 300],
    })
    out = clean_prices(raw, "TEST")
    assert len(out) == 2, "duplicate date collapsed, null adj_close dropped"
    assert (out["high"] >= out["low"]).all(), "inverted high/low repaired"
    assert out["adj_close"].iloc[0] == 10.6, "last write wins on a duplicate date"


def test_ingest_is_idempotent(seeded_db):
    before = len(load_prices(seeded_db, "AAA"))
    ingest_universe(seeded_db, ["AAA"], period="4y", with_news=False)
    assert len(load_prices(seeded_db, "AAA")) == before


def test_prices_are_sorted_and_unique(prices):
    d = pd.to_datetime(prices["date"])
    assert d.is_monotonic_increasing
    assert d.is_unique


# -------------------------------------------------------------- validation --
def test_walk_forward_never_leaks_future_into_training():
    dates = pd.Series(pd.bdate_range("2020-01-01", periods=1000))
    cv = PurgedWalkForward(n_splits=5, embargo=5, min_train_size=50)
    splits = list(cv.split(dates))
    assert splits, "expected at least one usable split"
    for tr, te in splits:
        assert dates.iloc[tr].max() < dates.iloc[te].min(), "train must precede test"
        gap = (dates.iloc[te].min() - dates.iloc[tr].max()).days
        assert gap >= 5, "the embargo gap was not honoured"
        assert not set(tr) & set(te)


def test_walk_forward_rejects_too_short_a_series():
    with pytest.raises(ValueError):
        list(PurgedWalkForward(n_splits=5).split(pd.Series(pd.bdate_range(
            "2020-01-01", periods=4))))


# -------------------------------------------------------------------- risk --
def test_var_is_a_positive_loss_magnitude():
    r = np.random.default_rng(3).standard_normal(1000) * 0.01
    var, cvar = historical_var(r, 0.95)
    assert var > 0 and cvar >= var, "CVaR sits in the tail beyond VaR"


def test_risk_score_is_bounded_and_explained(prices, seeded_db):
    bench = load_prices(seeded_db, "SPY")
    a = compute_risk(prices, "AAA", benchmark=bench)
    assert 0 <= a.risk_score <= 100
    assert a.grade in list("ABCDEF")
    assert len(a.components) == 6
    total = sum(c.contribution for c in a.components)
    assert total == pytest.approx(a.risk_score, abs=0.05), \
        "the score must equal the sum of its published parts"
    assert a.symbol in a.narrative


def test_a_more_volatile_series_scores_riskier(seeded_db):
    base = load_prices(seeded_db, "AAA").copy()
    rng = np.random.default_rng(11)
    shocked = base.copy()
    noise = np.exp(np.cumsum(rng.standard_normal(len(base)) * 0.05))
    for col in ("open", "high", "low", "close", "adj_close"):
        shocked[col] = base[col] * noise
    calm = compute_risk(base, "CALM")
    wild = compute_risk(shocked, "WILD")
    assert wild.annualized_vol > calm.annualized_vol
    assert wild.risk_score > calm.risk_score


def test_risk_needs_enough_history(prices):
    with pytest.raises(ValueError):
        compute_risk(prices.head(10), "AAA")


# ----------------------------------------------------------------- anomaly --
def test_injected_shock_is_detected(prices):
    df = prices.copy().reset_index(drop=True)
    i = len(df) - 5
    for col in ("close", "adj_close", "high"):
        df.loc[i, col] = df.loc[i, col] * 1.35
    df.loc[i, "volume"] = df["volume"].median() * 40

    events = detect_anomalies(df, lookback_days=30)
    kinds = {e.kind for e in events if e.date == str(pd.to_datetime(df.loc[i, "date"]).date())}
    assert kinds & {"PRICE_SPIKE", "VOLUME_SURGE", "MULTIVARIATE"}, \
        f"a 35% jump on 40x volume went unnoticed (found {kinds})"


def test_quiet_series_produces_few_anomalies():
    n = 400
    dates = pd.bdate_range("2022-01-03", periods=n)
    c = 100 + np.linspace(0, 10, n)
    df = pd.DataFrame({
        "date": dates, "open": c, "high": c * 1.001, "low": c * 0.999,
        "close": c, "adj_close": c, "volume": np.full(n, 1e6),
    })
    events = detect_anomalies(df, lookback_days=90)
    assert len(events) <= 3, "an almost-straight line should not be full of surprises"


# ---------------------------------------------------------------- training --
def test_training_reports_honest_metrics(trained):
    oof = trained["metrics"]["out_of_fold"]
    assert trained["n_rows"] > 1000
    assert 0.0 <= oof["accuracy"] <= 1.0
    assert 0.3 <= oof["roc_auc"] <= 0.75, \
        "an AUC outside this band on daily equity direction means leakage or a bug"
    assert 0.0 <= oof["brier"] <= 0.5
    assert trained["metrics"]["calibration"]["method"] in ("sigmoid", "isotonic")
    assert len(trained["feature_importance"]) > 0


def test_predictions_are_probabilities_and_differ_across_tickers(seeded_db, trained):
    from stockseer.api.services import analytics
    probs = {}
    for s in ("AAA", "BBB", "CCC"):
        p = analytics.predict_symbol(seeded_db, s, persist=False)
        assert 0.0 <= p["prob_up"] <= 1.0
        assert p["direction"] in ("UP", "DOWN", "NEUTRAL")
        probs[s] = p["prob_up"]
    assert len({round(v, 4) for v in probs.values()}) > 1, \
        "identical probabilities for every ticker means the calibrator collapsed"
