"""The built-in local vector store: exact cosine search over a NumPy matrix, persisted to disk.

Zero extra dependencies, deterministic, and fast enough for the intended scale (tens of thousands of
chunks). Larger deployments swap in pgvector / Qdrant / a managed store through the same interface.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import BinaryIO

import numpy as np

from heka.rag.adapters._deps import Settings
from heka.rag.errors import HekaRagError
from heka.rag.filters import matches, validate_filter
from heka.rag.types import Chunk, Filter, ScoredChunk

_FORMAT_VERSION = 1


class LocalStoreSettings(Settings):
    path: str | None = None  # directory for the index; None = in-memory only
    autosave: bool = True


class IndexMismatchError(HekaRagError):
    """The stored index does not fit the data being added (usually: the embedder changed)."""


class LocalVectorStore:
    settings_model = LocalStoreSettings
    supports_keyword = False

    def __init__(self, settings: LocalStoreSettings | None = None) -> None:
        self.settings = settings or LocalStoreSettings()
        self._path = Path(self.settings.path) if self.settings.path else None
        self._chunks: dict[str, Chunk] = {}
        self._ids: list[str] = []
        self._index: dict[str, int] = {}
        self._matrix: np.ndarray | None = None
        # Bumped on every change, so dependants (the BM25 index) know when to rebuild.
        self.revision = 0
        if self._path and (self._path / "meta.json").exists():
            self._load()

    # -- persistence --------------------------------------------------------------------------

    def _load(self) -> None:
        assert self._path is not None
        meta = json.loads((self._path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("format_version") != _FORMAT_VERSION:
            raise IndexMismatchError(
                f"Unsupported index format in {self._path}; re-ingest to rebuild."
            )
        lines = (self._path / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        matrix = np.load(self._path / "vectors.npy") if chunks else None
        if len(chunks) != meta["count"] or (matrix is not None and len(matrix) != len(chunks)):
            raise IndexMismatchError(
                f"Index in {self._path} is inconsistent; re-ingest to rebuild."
            )
        self._ids = [c.id for c in chunks]
        self._chunks = {c.id: c for c in chunks}
        self._index = {cid: i for i, cid in enumerate(self._ids)}
        self._matrix = matrix

    def _save(self) -> None:
        self.revision += 1
        if not self._path or not self.settings.autosave:
            return
        self._path.mkdir(parents=True, exist_ok=True)
        count = len(self._ids)
        vectors = self._matrix if self._matrix is not None else np.zeros((0, 0), dtype=np.float32)
        writes: dict[str, Callable[[BinaryIO], object]] = {
            "vectors.npy": lambda f: np.save(f, vectors),
            "chunks.jsonl": lambda f: f.write(
                "\n".join(self._chunks[i].model_dump_json() for i in self._ids).encode("utf-8")
            ),
            "meta.json": lambda f: f.write(
                json.dumps(
                    {
                        "format_version": _FORMAT_VERSION,
                        "count": count,
                        "dimensions": vectors.shape[-1],
                    }
                ).encode("utf-8")
            ),
        }
        for name, write in writes.items():  # meta.json last: it is what makes the index loadable
            temp = self._path / f"{name}.tmp"
            with temp.open("wb") as handle:
                write(handle)
            os.replace(temp, self._path / name)

    # -- VectorStore --------------------------------------------------------------------------

    async def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        raw = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        vectors = (raw / np.where(norms == 0, 1.0, norms)).astype(np.float32)
        if self._matrix is not None and self._matrix.shape[1] != vectors.shape[1]:
            raise IndexMismatchError(
                f"Embedding size changed ({self._matrix.shape[1]} -> {vectors.shape[1]}): the embedder "
                "differs from the one that built this index. Re-ingest with a full rebuild."
            )
        fresh: list[np.ndarray] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            if chunk.id in self._index and self._matrix is not None:
                self._matrix[self._index[chunk.id]] = vector
            else:
                self._index[chunk.id] = len(self._ids) + len(fresh)
                self._ids.append(chunk.id)
                fresh.append(vector)
            self._chunks[chunk.id] = chunk
        if fresh:
            stacked = np.vstack(fresh)
            self._matrix = stacked if self._matrix is None else np.vstack([self._matrix, stacked])
        self._save()

    async def delete(
        self, *, ids: Sequence[str] | None = None, filter: Filter | None = None
    ) -> int:
        if ids is None and not filter:
            raise ValueError("delete() needs ids or a filter; use clear() to remove everything")
        validate_filter(filter)
        candidates = set(ids) if ids is not None else set(self._ids)
        doomed = {
            cid
            for cid in candidates
            if cid in self._chunks and matches(filter, self._chunks[cid].metadata)
        }
        if not doomed:
            return 0
        keep = [i for i, cid in enumerate(self._ids) if cid not in doomed]
        self._matrix = self._matrix[keep] if self._matrix is not None and keep else None
        self._ids = [self._ids[i] for i in keep]
        self._chunks = {cid: self._chunks[cid] for cid in self._ids}
        self._index = {cid: i for i, cid in enumerate(self._ids)}
        self._save()
        return len(doomed)

    async def clear(self) -> None:
        self._chunks, self._ids, self._index, self._matrix = {}, [], {}, None
        self._save()

    async def search(
        self, embedding: Sequence[float], *, k: int, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        if self._matrix is None or k <= 0:
            return []
        validate_filter(filter)
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape[0] != self._matrix.shape[1]:
            raise IndexMismatchError(
                f"Query embedding size {query.shape[0]} does not match the index "
                f"({self._matrix.shape[1]}): the embedder differs from the one used to ingest."
            )
        norm = float(np.linalg.norm(query))
        scores = self._matrix @ (query / norm if norm else query)
        if filter:
            allowed = np.fromiter(
                (matches(filter, self._chunks[cid].metadata) for cid in self._ids),
                dtype=bool,
                count=len(self._ids),
            )
            scores = np.where(allowed, scores, -np.inf)
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [
            ScoredChunk(chunk=self._chunks[self._ids[i]], score=float(scores[i]), retriever="dense")
            for i in top
            if np.isfinite(scores[i])
        ]

    async def keyword_search(
        self, query: str, *, k: int, filter: Filter | None = None
    ) -> list[ScoredChunk]:
        raise NotImplementedError(
            "The local store has no native keyword search (arrives in Phase 2)"
        )

    async def get(self, ids: Sequence[str]) -> list[Chunk]:
        return [self._chunks[cid] for cid in ids if cid in self._chunks]

    async def all_chunks(self) -> list[Chunk]:
        return [self._chunks[cid] for cid in self._ids]

    async def count(self) -> int:
        return len(self._ids)
