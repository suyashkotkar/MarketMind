import os
import tempfile

import pytest

# Point the whole test session at a throwaway SQLite file and the offline
# data source *before* stockseer.config is first imported.
_TMP = tempfile.mkdtemp(prefix="stockseer-tests-")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP}/test.db")
os.environ.setdefault("ARTIFACTS_DIR", _TMP)
os.environ.setdefault("DATA_SOURCE", "synthetic")
os.environ.setdefault("NEWS_SOURCE", "synthetic")
os.environ.setdefault("ENV", "test")

from stockseer.data.pipeline import ingest_universe  # noqa: E402
from stockseer.db.session import SessionLocal, init_db  # noqa: E402

TEST_SYMBOLS = ["AAA", "BBB", "CCC", "SPY"]


@pytest.fixture(scope="session")
def seeded_db():
    init_db()
    db = SessionLocal()
    ingest_universe(db, TEST_SYMBOLS, period="4y", with_news=True)
    yield db
    db.close()


@pytest.fixture(scope="session")
def prices(seeded_db):
    from stockseer.data.pipeline import load_prices
    return load_prices(seeded_db, "AAA")


@pytest.fixture(scope="session")
def trained(seeded_db):
    from stockseer.api.services import analytics
    return analytics.train(seeded_db, ["AAA", "BBB", "CCC"])


@pytest.fixture(scope="session")
def client(seeded_db, trained):
    from fastapi.testclient import TestClient

    from stockseer.api.main import app
    with TestClient(app) as c:
        yield c
