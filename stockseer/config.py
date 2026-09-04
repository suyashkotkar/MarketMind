"""Central configuration. All values overridable via environment / .env."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = ROOT_DIR / "artifacts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- app ---
    app_name: str = "StockSeer"
    env: str = Field("development", description="development | production | test")
    log_level: str = "INFO"
    # Where trained models and the SQLite dev database live. Point this at a
    # mounted volume in production so models survive a container restart.
    artifacts_dir: str = str(ARTIFACT_DIR)

    # --- database ---
    # Postgres in docker/production; sqlite fallback keeps local dev & CI zero-setup.
    database_url: str = "sqlite:///./artifacts/stockseer.db"
    db_echo: bool = False

    # --- data ---
    # "yahoo" hits Yahoo Finance via yfinance. "synthetic" generates deterministic
    # pseudo-market data so the pipeline is testable offline / in CI.
    data_source: str = "yahoo"
    news_source: str = "auto"  # auto | yahoo | newsapi | finnhub | synthetic | none
    newsapi_key: str | None = None
    finnhub_key: str | None = None

    default_tickers: list[str] = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "JPM", "XOM", "JNJ",
    ]
    benchmark_ticker: str = "SPY"
    history_period: str = "5y"

    # --- modelling ---
    horizon_days: int = 5          # forward window we predict direction over
    embargo_days: int = 5          # purge gap in walk-forward CV (>= horizon)
    n_splits: int = 5
    min_train_rows: int = 500
    model_type: str = "lightgbm"   # lightgbm | xgboost | gbdt(sklearn)
    long_threshold: float = 0.55   # p(up) above this = actionable long signal
    short_threshold: float = 0.45

    # --- risk ---
    var_confidence: float = 0.95
    risk_lookback: int = 252

    # --- anomaly ---
    anomaly_contamination: float = 0.02
    anomaly_z_threshold: float = 3.0

    # --- api ---
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = [
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000",
        "http://localhost:4173", "http://127.0.0.1:4173",
    ]
    cache_ttl_seconds: int = 300

    @field_validator("default_tickers", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

    @property
    def artifact_dir(self) -> Path:
        d = Path(self.artifacts_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def model_dir(self) -> Path:
        d = self.artifact_dir / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
