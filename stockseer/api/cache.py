"""Tiny TTL cache. Feature/risk computation is ~50-200ms per symbol; the
dashboard polls a lot. Swap for Redis by reimplementing get/set."""
from __future__ import annotations

import functools
import threading
import time
from collections.abc import Callable
from typing import Any

from ..config import settings

_store: dict[str, tuple[float, Any]] = {}
_lock = threading.Lock()


def get(key: str) -> Any | None:
    with _lock:
        item = _store.get(key)
        if not item:
            return None
        expires, value = item
        if expires < time.time():
            _store.pop(key, None)
            return None
        return value


def set(key: str, value: Any, ttl: int | None = None) -> None:  # noqa: A001
    ttl = ttl if ttl is not None else settings.cache_ttl_seconds
    with _lock:
        _store[key] = (time.time() + ttl, value)


def clear(prefix: str | None = None) -> int:
    with _lock:
        keys = [k for k in _store if prefix is None or k.startswith(prefix)]
        for k in keys:
            _store.pop(k, None)
        return len(keys)


def cached(prefix: str, ttl: int | None = None) -> Callable:
    """Decorator keyed on the non-Session positional args."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key_parts = [prefix, fn.__name__]
            key_parts += [str(a) for a in args if not hasattr(a, "execute")]
            key_parts += [f"{k}={v}" for k, v in sorted(kwargs.items())]
            key = "|".join(key_parts)
            hit = get(key)
            if hit is not None:
                return hit
            val = fn(*args, **kwargs)
            set(key, val, ttl)
            return val
        return wrapper
    return deco
