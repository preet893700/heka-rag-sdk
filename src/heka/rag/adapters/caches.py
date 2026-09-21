"""Caches. `disk` (SQLite) lets repeated dev and eval runs reuse earlier LLM and embedding calls."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Any

from heka.rag.adapters._deps import Settings
from heka.rag.text import stable_hash


def cache_key(namespace: str, *parts: object) -> str:
    return f"{namespace}:{stable_hash(*parts)}"


class DiskCacheSettings(Settings):
    path: str = ".heka-rag/cache.sqlite"
    default_ttl_s: float | None = None  # None = entries never expire


class DiskCache:
    settings_model = DiskCacheSettings

    def __init__(self, settings: DiskCacheSettings | None = None) -> None:
        self.settings = settings or DiskCacheSettings()
        self._path = Path(self.settings.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT, expires REAL)"
            )
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self._path, timeout=30)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def _get(self, key: str) -> Any | None:
        with closing(self._connect()) as db:
            row = db.execute("SELECT value, expires FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None or (row[1] is not None and row[1] < time.time()):
            return None
        return json.loads(row[0])

    def _set(self, key: str, value: Any, ttl_s: float | None) -> None:
        ttl = ttl_s if ttl_s is not None else self.settings.default_ttl_s
        expires = time.time() + ttl if ttl is not None else None
        with closing(self._connect()) as db:
            db.execute(
                "INSERT OR REPLACE INTO kv (key, value, expires) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires),
            )
            db.commit()

    async def get(self, key: str) -> Any | None:
        return await asyncio.to_thread(self._get, key)

    async def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        await asyncio.to_thread(self._set, key, value, ttl_s)


class MemoryCache:
    """Process-local cache. Handy in tests; nothing survives the process."""

    def __init__(self) -> None:
        self._data: dict[str, tuple[Any, float | None]] = {}

    async def get(self, key: str) -> Any | None:
        entry = self._data.get(key)
        if entry is None or (entry[1] is not None and entry[1] < time.time()):
            return None
        return entry[0]

    async def set(self, key: str, value: Any, ttl_s: float | None = None) -> None:
        self._data[key] = (value, time.time() + ttl_s if ttl_s is not None else None)
