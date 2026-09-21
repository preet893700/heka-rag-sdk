"""Stage contracts. Adapters implement these; the pipelines depend only on these.

All I/O methods are async (the sync API is a thin wrapper over them). Protocols are structural:
an adapter needs no base class, it just has to have the right methods. `runtime_checkable` lets
contract tests assert conformance.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from kbsdk.types import (
    Answer,
    Chunk,
    Document,
    Filter,
    GuardrailResult,
    GuardrailStage,
    LLMResponse,
    Message,
    MetricResult,
    RequestContext,
    ScoredChunk,
    TraceEvent,
)

if TYPE_CHECKING:
    from kbsdk.eval.dataset import EvalCase


@runtime_checkable
class Loader(Protocol):
    """Reads one location (file, folder, URL, connector) and yields whole documents."""

    def load(self, location: str) -> AsyncIterator[Document]: ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into chunks, carrying metadata such as title and heading path."""

    async def chunk(self, document: Document) -> list[Chunk]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent chunk + vector storage. `filter` uses the SDK's metadata filter DSL."""

    @property
    def supports_keyword(self) -> bool:
        """True if `keyword_search` is native; otherwise the SDK builds a local BM25 index."""
        ...

    async def upsert(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]
    ) -> None: ...

    async def delete(
        self, *, ids: Sequence[str] | None = None, filter: Filter | None = None
    ) -> int: ...

    async def search(
        self, embedding: Sequence[float], *, k: int, filter: Filter | None = None
    ) -> list[ScoredChunk]: ...

    async def keyword_search(
        self, query: str, *, k: int, filter: Filter | None = None
    ) -> list[ScoredChunk]: ...

    async def get(self, ids: Sequence[str]) -> list[Chunk]: ...

    async def count(self) -> int: ...

    async def clear(self) -> None:
        """Remove every chunk (used by full rebuilds)."""
        ...

    async def all_chunks(self) -> list[Chunk]:
        """Every stored chunk. Only used to build the SDK's own keyword (BM25) index when
        `supports_keyword` is False, so stores with native keyword search never need to implement it
        efficiently."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """Query -> ranked chunks. Dense, sparse and hybrid retrievers look the same from outside."""

    async def retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: Filter | None = None,
        context: RequestContext | None = None,
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], *, top_n: int
    ) -> list[ScoredChunk]: ...


@runtime_checkable
class QueryTransformer(Protocol):
    """Rewrites a question before search: rewrite, multi-query, HyDE, follow-up condensation.

    Returns one or more queries to search with; the first is the primary one.
    """

    async def transform(self, query: str, history: Sequence[Message]) -> list[str]: ...


@runtime_checkable
class LLM(Protocol):
    @property
    def name(self) -> str: ...

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse: ...

    def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]: ...


@runtime_checkable
class Guardrail(Protocol):
    """Checks the question (`input`), each retrieved chunk (`context`) or the drafted answer
    (`output`). A guardrail returns `allow` for stages it has no opinion on."""

    async def check(
        self,
        text: str,
        *,
        stage: GuardrailStage,
        context: RequestContext | None = None,
    ) -> GuardrailResult: ...


@runtime_checkable
class Cache(Protocol):
    async def get(self, key: str) -> Any | None: ...

    async def set(self, key: str, value: Any, ttl_s: float | None = None) -> None: ...


@runtime_checkable
class OCREngine(Protocol):
    """Reads text from an image (a scanned page, a photo of a document)."""

    async def recognize(self, image: Any) -> str: ...


@runtime_checkable
class Tracer(Protocol):
    """Receives structured trace events (retrieval, rerank, LLM call, guardrail decision, ...)."""

    def emit(self, event: TraceEvent) -> None: ...


@runtime_checkable
class Metric(Protocol):
    """Scores one answer against one eval case."""

    name: str

    async def score(self, case: EvalCase, answer: Answer) -> MetricResult: ...
