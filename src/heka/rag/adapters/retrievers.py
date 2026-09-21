"""Retrievers: dense (embeddings), sparse (BM25) and hybrid (both, fused), plus rank fusion helpers."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from heka.rag.adapters.sparse import SharedBm25
from heka.rag.interfaces import Embedder, Retriever, VectorStore
from heka.rag.types import Filter, RequestContext, ScoredChunk


def fuse_rrf(
    ranked_lists: Sequence[Sequence[ScoredChunk]],
    *,
    k: int,
    rrf_k: int = 60,
    weights: Sequence[float] | None = None,
    label: str = "hybrid",
) -> list[ScoredChunk]:
    """Reciprocal Rank Fusion: score = sum over lists of weight / (rrf_k + rank).

    Uses ranks only, so lists with incomparable scores (cosine similarity vs BM25) combine fairly.
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, ScoredChunk] = {}
    signals: dict[str, dict[str, float]] = {}
    order: list[str] = []
    for list_index, results in enumerate(ranked_lists):
        weight = weights[list_index] if weights else 1.0
        for rank, item in enumerate(results, start=1):
            chunk_id = item.chunk.id
            if chunk_id not in first_seen:
                first_seen[chunk_id] = item
                signals[chunk_id] = {}
                order.append(chunk_id)
            signals[chunk_id].update(item.signals)
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (rrf_k + rank)
    ranked = sorted(order, key=lambda cid: (-fused[cid], order.index(cid)))
    return [
        ScoredChunk(
            chunk=first_seen[cid].chunk, score=fused[cid], retriever=label, signals=signals[cid]
        )
        for cid in ranked[:k]
    ]


def _minmax(results: Sequence[ScoredChunk]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r.score for r in results]
    low, high = min(scores), max(scores)
    span = high - low
    return {r.chunk.id: (1.0 if span == 0 else (r.score - low) / span) for r in results}


def fuse_weighted(
    dense: Sequence[ScoredChunk], sparse: Sequence[ScoredChunk], *, k: int, dense_weight: float
) -> list[ScoredChunk]:
    """Blend min-max normalised scores: `dense_weight * dense + (1 - dense_weight) * sparse`."""
    dense_scores, sparse_scores = _minmax(dense), _minmax(sparse)
    by_id: dict[str, ScoredChunk] = {}
    signals: dict[str, dict[str, float]] = {}
    for item in [*dense, *sparse]:
        by_id.setdefault(item.chunk.id, item)
        signals.setdefault(item.chunk.id, {}).update(item.signals)
    blended = {
        cid: dense_weight * dense_scores.get(cid, 0.0)
        + (1 - dense_weight) * sparse_scores.get(cid, 0.0)
        for cid in by_id
    }
    ranked = sorted(blended, key=lambda cid: -blended[cid])
    return [
        ScoredChunk(
            chunk=by_id[cid].chunk, score=blended[cid], retriever="hybrid", signals=signals[cid]
        )
        for cid in ranked[:k]
    ]


class DenseRetriever:
    """Embedding similarity search."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    async def retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: Filter | None = None,
        context: RequestContext | None = None,
    ) -> list[ScoredChunk]:
        embedding = await self.embedder.embed_query(query)
        results = await self.store.search(embedding, k=k, filter=filter)
        return [r.model_copy(update={"signals": {**r.signals, "dense": r.score}}) for r in results]


class SparseRetriever:
    """Keyword search: the store's native one if it has one, otherwise the SDK's BM25 index."""

    def __init__(self, store: VectorStore) -> None:
        self.store = store
        self._bm25 = SharedBm25()

    async def _index_key(self) -> object:
        return (getattr(self.store, "revision", None), await self.store.count())

    async def retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: Filter | None = None,
        context: RequestContext | None = None,
    ) -> list[ScoredChunk]:
        if self.store.supports_keyword:
            results = await self.store.keyword_search(query, k=k, filter=filter)
            return [
                r.model_copy(update={"signals": {**r.signals, "sparse": r.score}}) for r in results
            ]
        key = await self._index_key()
        if key != self._bm25.key:
            chunks = await self.store.all_chunks()
            index = await asyncio.to_thread(self._bm25.ensure, key, chunks)
        else:
            index = self._bm25.ensure(key, [])
        return index.search(query, k=k, filter=filter)


class HybridRetriever:
    """Runs dense and sparse retrieval and fuses the two rankings."""

    def __init__(
        self,
        dense: Retriever,
        sparse: Retriever,
        *,
        fusion: str = "rrf",
        rrf_k: int = 60,
        dense_weight: float = 0.5,
    ) -> None:
        self.dense, self.sparse = dense, sparse
        self.fusion, self.rrf_k, self.dense_weight = fusion, rrf_k, dense_weight

    async def retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: Filter | None = None,
        context: RequestContext | None = None,
    ) -> list[ScoredChunk]:
        depth = max(2 * k, 20)  # each side looks deeper than k so fusion has real choices
        dense, sparse = await asyncio.gather(
            self.dense.retrieve(query, k=depth, filter=filter, context=context),
            self.sparse.retrieve(query, k=depth, filter=filter, context=context),
        )
        if self.fusion == "weighted":
            return fuse_weighted(dense, sparse, k=k, dense_weight=self.dense_weight)
        return fuse_rrf(
            [dense, sparse],
            k=k,
            rrf_k=self.rrf_k,
            weights=[self.dense_weight * 2, (1 - self.dense_weight) * 2],
        )
