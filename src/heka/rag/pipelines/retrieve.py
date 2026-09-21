"""The retrieval stage: everything between a question and the chunks the model will read.

    condense follow-up -> transform (rewrite / multi-query / HyDE) -> retrieve each query
    -> fuse (RRF) -> rerank -> expand to parent sections -> top `final_k`

Every optional step is off unless configured, and every step leaves a `TraceEvent`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from heka.rag.access import AccessPolicy
from heka.rag.adapters.retrievers import fuse_rrf
from heka.rag.adapters.transforms import Condenser
from heka.rag.config import RAGConfig
from heka.rag.filters import combine
from heka.rag.interfaces import QueryTransformer, Reranker, Retriever
from heka.rag.types import Filter, Message, RequestContext, ScoredChunk, TraceEvent


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk]
    queries: list[str]
    trace: list[TraceEvent] = field(default_factory=list)
    embedding_calls: int = 0


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def expand_parents(candidates: Sequence[ScoredChunk], limit: int) -> list[ScoredChunk]:
    """Swap small "child" chunks for the larger section they came from (parent-child chunking).

    Search precision comes from small chunks; answer quality from their surrounding context. Several
    children of one parent collapse into a single result at the best child's rank.
    """
    seen: set[str] = set()
    expanded: list[ScoredChunk] = []
    for item in candidates:
        metadata = item.chunk.metadata
        parent_text = metadata.get("parent_text")
        if parent_text:
            parent_id = str(metadata.get("parent_id", item.chunk.id))
            if parent_id in seen:
                continue
            seen.add(parent_id)
            kept = {k: v for k, v in metadata.items() if k != "parent_text"}
            chunk = item.chunk.model_copy(update={"text": parent_text, "metadata": kept})
            item = item.model_copy(update={"chunk": chunk})
        expanded.append(item)
        if len(expanded) >= limit:
            break
    return expanded


class RetrievalPipeline:
    def __init__(
        self,
        config: RAGConfig,
        retriever: Retriever,
        *,
        reranker: Reranker | None = None,
        transforms: Sequence[QueryTransformer] = (),
        condenser: Condenser | None = None,
        access: AccessPolicy | None = None,
    ) -> None:
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
        self.transforms = list(transforms)
        self.condenser = condenser
        self.access = access or AccessPolicy(config.access)

    async def run(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        context: RequestContext | None = None,
        filter: Filter | None = None,
        final_k: int | None = None,
    ) -> RetrievalResult:
        settings = self.config.retrieval
        limit = final_k or settings.final_k
        depth = max(settings.top_k, limit)
        trace: list[TraceEvent] = []
        # Computed first: without the identity an access policy needs, nothing is searched at all.
        access_filter = self.access.filter_for(context)

        standalone = question
        if self.condenser is not None and history:
            started = time.perf_counter()
            standalone = await self.condenser.condense(question, history)
            trace.append(
                TraceEvent(
                    stage="condense",
                    name="condense",
                    duration_ms=_ms(started),
                    data={"original": question, "standalone": standalone},
                )
            )

        queries = [standalone]
        for transform in self.transforms:
            started = time.perf_counter()
            produced = await transform.transform(standalone, history)
            queries.extend(
                q for q in produced if q.casefold() not in {x.casefold() for x in queries}
            )
            trace.append(
                TraceEvent(
                    stage="transform",
                    name=type(transform).__name__,
                    duration_ms=_ms(started),
                    data={"queries": list(queries)},
                )
            )

        started = time.perf_counter()
        # The access filter is ANDed in last-but-never-loosened: a caller's `filter` can only narrow.
        scope = combine(settings.filter, access_filter, filter)
        lists = await asyncio.gather(
            *(self.retriever.retrieve(q, k=depth, filter=scope, context=context) for q in queries)
        )
        candidates = (
            lists[0]
            if len(lists) == 1
            else fuse_rrf(lists, k=depth, rrf_k=settings.rrf_k, label="multi-query")
        )
        pool = limit * 2  # parent expansion can merge results, so keep a margin
        trace.append(
            TraceEvent(
                stage="retrieve",
                name="retrieve",
                duration_ms=_ms(started),
                data={
                    "mode": settings.mode,
                    "agentic": settings.agentic is not None,
                    "queries": len(queries),
                    "candidates": len(candidates),
                },
            )
        )

        if self.reranker is not None and candidates:
            started = time.perf_counter()
            before = [c.chunk.id for c in candidates[:limit]]
            candidates = await self.reranker.rerank(standalone, candidates, top_n=pool)
            trace.append(
                TraceEvent(
                    stage="rerank",
                    name=type(self.reranker).__name__,
                    duration_ms=_ms(started),
                    data={
                        "changed_top": before != [c.chunk.id for c in candidates[:limit]],
                        "scores": [round(c.score, 4) for c in candidates[:limit]],
                    },
                )
            )

        chunks = expand_parents(candidates[:pool], limit)
        trace.append(
            TraceEvent(
                stage="select",
                name="selected",
                data={
                    "count": len(chunks),
                    "scores": [round(c.score, 4) for c in chunks],
                    "relevance": [round(c.relevance, 4) for c in chunks],
                    "sources": [str(c.chunk.metadata.get("source", "")) for c in chunks],
                },
            )
        )
        dense_used = settings.mode in {"dense", "hybrid"}
        return RetrievalResult(
            chunks=chunks,
            queries=queries,
            trace=trace,
            embedding_calls=len(queries) if dense_used else 0,
        )
