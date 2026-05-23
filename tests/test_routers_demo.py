import pytest
from fastapi.testclient import TestClient
from models import Stock, DailySnapshot


@pytest.fixture
def auth_header(client):
    client.post("/auth/register", json={"username": "demouser", "password": "pass1234"})
    r = client.post("/auth/login", json={"username": "demouser", "password": "pass1234"})
    token = r.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def seeded_db():
    from tests.conftest import TestingSessionLocal
    db = TestingSessionLocal()
    db.add(Stock(ticker="AAPL", company_name="Apple Inc.", sector="Technology"))
    for i, (d, c, v) in enumerate([
        ("2026-04-25", 173.0, 50_000_000),
        ("2026-04-24", 170.0, 55_000_000),
        ("2026-04-23", 168.0, 52_000_000),
    ]):
        db.add(DailySnapshot(
            ticker="AAPL", date=d,
            open=c - 2, high=c + 3, low=c - 3, close=c, volume=v,
            rsi=58.0, ma7=171.0, ma30=165.0, upper_bb=180.0, lower_bb=160.0
        ))
    db.commit()
    db.close()


def test_stocks_requires_auth(client):
    r = client.get("/demo/stocks")
    assert r.status_code in (401, 403)


def test_stocks_returns_list(client, auth_header, seeded_db):
    r = client.get("/demo/stocks", headers=auth_header)
    assert r.status_code == 200
    body = r.json()
    assert "stocks" in body
    assert isinstance(body["stocks"], list)


def test_history_requires_auth(client):
    r = client.get("/demo/history/AAPL")
    assert r.status_code in (401, 403)


def test_history_returns_list(client, auth_header, seeded_db):
    r = client.get("/demo/history/AAPL", headers=auth_header)
    assert r.status_code == 200
    assert isinstance(r.json()["predictions"], list)


def test_snapshot_requires_auth(client):
    r = client.get("/demo/snapshot/AAPL")
    assert r.status_code in (401, 403)


def test_snapshot_returns_cached(client, auth_header, seeded_db):
    r = client.get("/demo/snapshot/AAPL", headers=auth_header)
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert "close" in body
    assert body["source"] == "cache"


def test_trend_clock_requires_auth(client):
    r = client.get("/demo/trend-clock/AAPL")
    assert r.status_code in (401, 403)


def test_trend_clock_returns_trend(client, auth_header, seeded_db):
    r = client.get("/demo/trend-clock/AAPL", headers=auth_header)
    assert r.status_code == 200
    body = r.json()
    assert "trend" in body
    assert body["trend"]["direction"] in ("up", "down")
    assert "prob_reversal_7d" in body
