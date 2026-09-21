"""Qdrant vector store: embedded (in-memory or on-disk, no server) or a Qdrant server / cloud.

The SDK's metadata filter dialect is translated to Qdrant's, and the semantics are held identical to the
built-in store's (a differential test runs the same filters through both), because access control is
built on these filters: list-valued fields match on any element, `$in` on a list is an overlap, and
a missing field satisfies `$ne` / `$nin` but never `$eq` / `$in`.

Keyword search is not native here, so hybrid retrieval uses the SDK's own BM25 index over `all_chunks()`.
"""

from __future__ import annotations

import asyncio
import uuid
import weakref
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from pydantic import model_validator

from heka.rag.adapters._deps import Settings, api_key_from_env, require
from heka.rag.adapters.stores import IndexMismatchError
from heka.rag.errors import ConfigError
from heka.rag.filters import validate_filter
from heka.rag.types import Chunk, Filter, ScoredChunk

_NAMESPACE = uuid.UUID("6f0f1c1e-7a52-4f0b-9c1e-3a3f5c0d9b11")
_CHUNK_KEY = "_kb_chunk"
_MAX_INDEXED_TEXT = (
    512  # longer strings stay only inside the stored chunk JSON, not as filter fields
)


def point_id(chunk_id: str) -> str:
    """Qdrant ids must be UUIDs or ints; derive a stable UUID from the chunk id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class QdrantSettings(Settings):
    location: str | None = None  # ":memory:" or a folder for embedded on-disk storage
    url: str | None = None  # a Qdrant server or cloud cluster
    api_key_env: str | None = None  # environment variable holding the server API key
    collection: str | None = None  # default: derived from the agent name
    batch_size: int = 128
    timeout_s: float | None = 30.0

    @model_validator(mode="after")
    def _one_place(self) -> QdrantSettings:
        if self.location and self.url:
            raise ValueError("set either `location` (embedded) or `url` (server), not both")
        return self


# -- filter translation -------------------------------------------------------------------------


def _scalar_ok(value: Any) -> bool:
    return isinstance(value, str | int | bool) and not isinstance(value, float)


def _range(models: Any, comparisons: dict[str, Any], field: str) -> Any:
    sample = next(iter(comparisons.values()))
    if isinstance(sample, str):
        try:
            values = {op: datetime.fromisoformat(v) for op, v in comparisons.items()}
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"Qdrant range filters on {field!r} need numbers or ISO dates, got {sample!r}"
            ) from exc
        return models.FieldCondition(key=field, range=models.DatetimeRange(**values))
    if not all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in comparisons.values()
    ):
        raise ConfigError(f"Qdrant range filters on {field!r} need numbers or ISO dates")
    return models.FieldCondition(key=field, range=models.Range(**comparisons))


def to_qdrant_filter(models: Any, flt: Filter | None) -> Any:
    """Translate the SDK filter dialect to a `qdrant_client.models.Filter` (None for no filter)."""
    if not flt:
        return None
    validate_filter(flt)
    must: list[Any] = []
    should: list[Any] = []
    must_not: list[Any] = []
    for key, cond in flt.items():
        if key == "$and":
            must.extend(to_qdrant_filter(models, sub) for sub in cond)
        elif key == "$or":
            should_group = [to_qdrant_filter(models, sub) or models.Filter() for sub in cond]
            must.append(models.Filter(should=should_group))
        elif isinstance(cond, dict) and cond and all(str(k).startswith("$") for k in cond):
            comparisons: dict[str, Any] = {}
            for op, operand in cond.items():
                if op == "$eq":
                    must.append(_match(models, key, operand))
                elif op == "$ne":
                    must_not.append(_match(models, key, operand))
                elif op == "$in":
                    must.append(_match_any(models, key, operand))
                elif op == "$nin":
                    must_not.append(_match_any(models, key, operand))
                else:
                    comparisons[op[1:]] = operand  # gt / gte / lt / lte
            if comparisons:
                must.append(_range(models, comparisons, key))
        else:
            must.append(_match(models, key, cond))
    return models.Filter(must=must or None, should=should or None, must_not=must_not or None)


def _match(models: Any, key: str, value: Any) -> Any:
    if not _scalar_ok(value):
        raise ConfigError(
            f"Qdrant equality filters support strings, integers and booleans ({key!r})"
        )
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(models: Any, key: str, values: Any) -> Any:
    items = list(values)
    if not items or not all(_scalar_ok(v) and not isinstance(v, bool) for v in items):
        raise ConfigError(
            f"Qdrant $in / $nin on {key!r} needs a non-empty list of strings or integers"
        )
    return models.FieldCondition(key=key, match=models.MatchAny(any=items))


def _payload(chunk: Chunk) -> dict[str, Any]:
    """Filterable metadata at the top level, plus the whole chunk as JSON for exact round-trips."""
    payload: dict[str, Any] = {}
    for key, value in chunk.metadata.items():
        if key == _CHUNK_KEY:
            continue
        if isinstance(value, str) and len(value) > _MAX_INDEXED_TEXT:
            continue
        if isinstance(value, str | int | float | bool) or (
            isinstance(value, list) and all(isinstance(v, str | int | float | bool) for v in value)
        ):
            payload[key] = value
    payload[_CHUNK_KEY] = chunk.model_dump_json()
    return payload


def _chunk(payload: dict[str, Any]) -> Chunk:
    return Chunk.model_validate_json(payload[_CHUNK_KEY])


class QdrantStore:
    settings_model = QdrantSettings
    supports_keyword = False

    def __init__(self, settings: QdrantSettings | None = None) -> None:
        self.settings = settings or QdrantSettings()
        module = require("qdrant_client", stage="store", provider="qdrant", extra="qdrant")
        self._module = module
        self._models = module.models
        self.collection = self.settings.collection or "heka-rag"
        self.revision = 0
        self._shared: Any = None  # embedded mode: one client (a folder can only be opened once)
        self._per_loop: weakref.WeakKeyDictionary[Any, Any] = weakref.WeakKeyDictionary()
        self._server = bool(self.settings.url)

    # -- client handling ----------------------------------------------------------------------

    def _new_client(self) -> Any:
        if self.settings.url:
            key = (
                api_key_from_env(self.settings.api_key_env, "qdrant")
                if self.settings.api_key_env
                else None
            )
            return self._module.AsyncQdrantClient(
                url=self.settings.url, api_key=key, timeout=self.settings.timeout_s
            )
        location = self.settings.location or ":memory:"
        if location == ":memory:":
            return self._module.AsyncQdrantClient(location=":memory:")
        return self._module.AsyncQdrantClient(path=location)  # embedded, persisted in a folder

    def _client(self) -> Any:
        if not self._server:
            if self._shared is None:
                self._shared = self._new_client()
            return self._shared
        loop = asyncio.get_running_loop()  # network clients are bound to the loop that made them
        client = self._per_loop.get(loop)
        if client is None:
            client = self._per_loop[loop] = self._new_client()
        return client

    # -- helpers ------------------------------------------------------------------------------

    async def _exists(self) -> bool:
        return bool(await self._client().collection_exists(self.collection))

    async def _ensure(self, size: int) -> None:
        client = self._client()
        if not await self._exists():
            await client.create_collection(
                self.collection,
                vectors_config=self._models.VectorParams(
                    size=size, distance=self._models.Distance.COSINE
                ),
            )
            return
        info = await client.get_collection(self.collection)
        existing = info.config.params.vectors.size
        if existing != size:
            raise IndexMismatchError(
                f"Embedding size changed ({existing} -> {size}): the embedder differs from the one "
                "that built this collection. Re-ingest with a full rebuild."
            )

    def _filter(self, flt: Filter | None) -> Any:
        return to_qdrant_filter(self._models, flt)

    async def _matching_ids(self, flt: Filter | None) -> list[str]:
        ids: list[str] = []
        offset = None
        while True:
            points, offset = await self._client().scroll(
                self.collection,
                scroll_filter=self._filter(flt),
                limit=512,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            ids.extend(str(p.id) for p in points)
            if offset is None:
                return ids

    # -- VectorStore --------------------------------------------------------------------------

    async def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        await self._ensure(len(embeddings[0]))
        step = self.settings.batch_size
        for start in range(0, len(chunks), step):
            points = [
                self._models.PointStruct(
                    id=point_id(chunk.id), vector=list(map(float, vector)), payload=_payload(chunk)
                )
                for chunk, vector in zip(
                    chunks[start : start + step], embeddings[start : start + step], strict=True
                )
            ]
            await self._client().upsert(self.collection, points=points, wait=True)
        self.revision += 1

    async def delete(
        self, *, ids: Sequence[str] | None = None, filter: Filter | None = None
    ) -> int:
        if ids is None and not filter:
            raise ValueError("delete() needs ids or a filter; use clear() to remove everything")
        if not await self._exists():
            return 0
        validate_filter(filter)
        if ids is not None:
            wanted = [point_id(i) for i in ids]
            existing = {str(p.id) for p in await self._client().retrieve(self.collection, wanted)}
            doomed = [w for w in wanted if w in existing]
            if filter and doomed:
                allowed = set(await self._matching_ids(filter))
                doomed = [d for d in doomed if d in allowed]
        else:
            doomed = await self._matching_ids(filter)
        if not doomed:
            return 0
        await self._client().delete(
            self.collection, points_selector=self._models.PointIdsList(points=doomed), wait=True
        )
        self.revision += 1
        return len(doomed)

    async def clear(self) -> None:
        if await self._exists():
            await self._client().delete_collection(self.collection)
        self.revision += 1

    async def search(
        self, embedding: Sequence[float], *, k: int, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        if k <= 0 or not await self._exists():
            return []
        info = await self._client().get_collection(self.collection)
        if info.config.params.vectors.size != len(embedding):
            raise IndexMismatchError(
                f"Query embedding size {len(embedding)} does not match the collection "
                f"({info.config.params.vectors.size}): the embedder differs from the one used to ingest."
            )
        response = await self._client().query_points(
            self.collection,
            query=list(map(float, embedding)),
            limit=k,
            query_filter=self._filter(filter),
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=_chunk(p.payload), score=float(p.score), retriever="dense")
            for p in response.points
            if p.payload
        ]

    async def keyword_search(
        self, query: str, *, k: int, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        raise NotImplementedError(
            "Qdrant store has no native keyword search; the SDK's BM25 is used"
        )

    async def get(self, ids: Sequence[str]) -> list[Chunk]:
        if not await self._exists():
            return []
        points = await self._client().retrieve(
            self.collection, [point_id(i) for i in ids], with_payload=True
        )
        return [_chunk(p.payload) for p in points if p.payload]

    async def count(self) -> int:
        if not await self._exists():
            return 0
        return int((await self._client().count(self.collection, exact=True)).count)

    async def all_chunks(self) -> list[Chunk]:
        if not await self._exists():
            return []
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = await self._client().scroll(
                self.collection, limit=512, offset=offset, with_payload=True, with_vectors=False
            )
            chunks.extend(_chunk(p.payload) for p in points if p.payload)
            if offset is None:
                return sorted(chunks, key=lambda c: (c.doc_id, c.index))
