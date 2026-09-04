"""Filesystem model registry: versioned artifacts + metadata, latest pointer."""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

from ..config import settings

log = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    version: str
    model_type: str
    horizon_days: int
    features: list[str]
    n_rows: int
    tickers: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    trained_at: str = field(default_factory=lambda: dt.datetime.now(
        dt.UTC).isoformat())
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def new_version(prefix: str = "direction") -> str:
    return f"{prefix}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%S}"


def _root() -> Path:
    return settings.model_dir


_LOADED: dict[str, tuple[Any, ModelMetadata]] = {}


def save(estimator: Any, meta: ModelMetadata) -> Path:
    d = _root() / meta.version
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, d / "model.joblib")
    (d / "metadata.json").write_text(meta.to_json())
    (_root() / "LATEST").write_text(meta.version)
    _LOADED.clear()
    log.info("saved model %s -> %s", meta.version, d)
    return d


def latest_version() -> str | None:
    p = _root() / "LATEST"
    if p.exists():
        v = p.read_text().strip()
        if (_root() / v / "model.joblib").exists():
            return v
    versions = sorted(x.name for x in _root().iterdir()
                      if x.is_dir() and (x / "model.joblib").exists())
    return versions[-1] if versions else None


def load(version: str | None = None) -> tuple[Any, ModelMetadata] | tuple[None, None]:
    """Load a model bundle. Deserialisation is memoised — the dashboard asks for
    a prediction once per ticker per page and unpickling each time dominates."""
    version = version or latest_version()
    if not version:
        return None, None
    if version in _LOADED:
        return _LOADED[version]
    d = _root() / version
    if not (d / "model.joblib").exists():
        return None, None
    est = joblib.load(d / "model.joblib")
    meta = ModelMetadata(**json.loads((d / "metadata.json").read_text()))
    _LOADED[version] = (est, meta)
    return est, meta


def list_versions() -> list[dict]:
    out = []
    for d in sorted(_root().iterdir()):
        f = d / "metadata.json"
        if d.is_dir() and f.exists():
            out.append(json.loads(f.read_text()))
    return out
