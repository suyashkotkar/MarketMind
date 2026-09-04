import pytest

P = "/api/v1"


def test_health(client):
    r = client.get(f"{P}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["tickers_loaded"] >= 3
    assert body["model_version"]


def test_config_exposes_no_secrets(client):
    body = client.get(f"{P}/config").json()
    assert "horizon_days" in body
    for leaky in ("newsapi_key", "finnhub_key", "database_url"):
        assert leaky not in body


def test_list_and_get_ticker(client):
    symbols = [t["symbol"] for t in client.get(f"{P}/stocks").json()]
    assert "AAA" in symbols
    assert client.get(f"{P}/stocks/AAA").json()["symbol"] == "AAA"
    assert client.get(f"{P}/stocks/NOPE").status_code == 404


def test_history_returns_bars_and_indicators(client):
    body = client.get(f"{P}/stocks/AAA/history?days=200").json()
    assert len(body["bars"]) <= 200 and len(body["bars"]) > 100
    assert "rsi_14" in body["indicators"]
    assert len(body["indicators"]["rsi_14"]) == len(body["bars"])
    bar = body["bars"][0]
    assert bar["high"] >= bar["low"]


def test_overview_and_ranking(client):
    ov = client.get(f"{P}/stocks/overview?limit=5").json()
    assert ov["universe_size"] >= 3
    assert ov["cards"]
    ranking = client.get(f"{P}/risk/ranking?limit=10").json()
    scores = [r["risk_score"] for r in ranking]
    assert scores == sorted(scores), "ranking must run safest-first"


def test_prediction_contract(client):
    body = client.get(f"{P}/predictions/AAA").json()
    assert 0 <= body["prob_up"] <= 1
    assert body["direction"] in ("UP", "DOWN", "NEUTRAL")
    assert body["horizon_days"] >= 1
    assert "not investment advice" in body["disclaimer"].lower()
    assert client.get(f"{P}/predictions/NOPE").status_code == 404


def test_model_card(client):
    body = client.get(f"{P}/predictions/model").json()
    assert body["metrics"]["out_of_fold"]["n"] > 0
    assert body["n_features"] > 10


def test_risk_contract(client):
    body = client.get(f"{P}/risk/AAA").json()
    assert 0 <= body["risk_score"] <= 100
    assert body["grade"] in list("ABCDEF")
    assert len(body["components"]) == 6
    assert body["narrative"]


def test_anomalies(client):
    body = client.get(f"{P}/anomalies/AAA?lookback_days=180").json()
    assert body["symbol"] == "AAA"
    assert "summary" in body
    for e in body["events"]:
        assert 0 <= e["severity"] <= 1


def test_sentiment_and_ad_hoc_scoring(client):
    body = client.get(f"{P}/sentiment/AAA?limit=10").json()
    assert body["articles"]
    assert -1 <= body["mean_sentiment"] <= 1

    scored = client.post(f"{P}/sentiment/score", json={"texts": [
        "Company beats estimates and raises guidance",
        "Company misses estimates, faces fraud probe",
        "Company files its quarterly report",
    ]}).json()
    assert scored[0]["score"] > 0.1
    assert scored[1]["score"] < -0.1
    assert abs(scored[2]["score"]) < 0.4


def test_compare(client):
    body = client.get(f"{P}/compare?symbols=AAA,BBB").json()
    assert len(body["rows"]) == 2
    assert set(body["series"]) == {"AAA", "BBB"}
    assert body["series"]["AAA"]["normalized"][0] == pytest.approx(100.0)
    assert body["correlation"]["AAA"]["AAA"] == pytest.approx(1.0, abs=1e-6)
    assert client.get(f"{P}/compare?symbols=AAA").status_code == 400


def test_alerts_shape(client):
    alerts = client.get(f"{P}/alerts?days=60&min_confidence=0.0").json()
    for a in alerts:
        assert a["kind"] in ("SIGNAL", "RISK", "ANOMALY")
        assert a["level"] in ("info", "warning", "critical")


def test_openapi_is_served(client):
    schema = client.get("/openapi.json").json()
    assert f"{P}/predictions/{{symbol}}" in schema["paths"]
