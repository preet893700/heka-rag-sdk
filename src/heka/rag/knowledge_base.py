"""`KnowledgeBase`: the documents behind an agent - ingest them, search them."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from heka.rag import factory
from heka.rag.access import AccessPolicy
from heka.rag.aio import run_sync
from heka.rag.config import RAGConfig
from heka.rag.interfaces import LLM, Cache, Chunker, Embedder, OCREngine, Retriever, VectorStore
from heka.rag.pipelines.health import HealthReport
from heka.rag.pipelines.ingest import IngestReport, analyze_documents, run_ingest
from heka.rag.pipelines.retrieve import RetrievalPipeline
from heka.rag.text import slugify
from heka.rag.types import Filter, Message, RequestContext, ScoredChunk
from heka.rag.usage import MeteredLLM


class KnowledgeBase:
    """Owns the index for one agent. Components come from the config, or can be passed in directly.

    The index lives in `<persist_dir>/<agent name>/`, so several agents can share a project.
    Underlying pieces are public (`store`, `embedder`, `retriever`, `retrieval`) as escape hatches.
    """

    def __init__(
        self,
        config: RAGConfig,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        cache: Cache | None = None,
    ) -> None:
        factory.validate_supported(config)
        factory.apply_env_file(config)
        self.config = config
        self.access = AccessPolicy(config.access)
        self.dir = Path(config.knowledge.persist_dir) / slugify(config.name)
        self.cache: Cache | None = cache if cache is not None else factory.build_cache(config)
        self.embedder: Embedder = embedder or factory.build_embedder(config, self.cache)
        self.store: VectorStore = store or factory.build_store(config, self.dir)
        self.chunker: Chunker = factory.build_chunker(config, self.embedder)
        self.ocr: OCREngine | None = factory.build_ocr(config)
        self.retriever: Retriever = factory.build_retriever(config, self.store, self.embedder)
        self._llm: LLM | None = None
        self.retrieval: RetrievalPipeline = self.build_retrieval()

    def _default_llm(self) -> LLM:
        """The generation model from the config, built only if an LLM-based retrieval step needs it."""
        if self._llm is None:
            self._llm = MeteredLLM(
                factory.build_llm(
                    self.config,
                    self.config.generation.llm,
                    self.cache,
                    defaults=factory.generation_defaults(self.config),
                )
            )
        return self._llm

    def build_retrieval(
        self, llm: LLM | None = None, config: RAGConfig | None = None
    ) -> RetrievalPipeline:
        """A retrieval pipeline over this knowledge base. `llm` serves any LLM-based steps."""
        provider = (lambda: llm) if llm is not None else self._default_llm
        return factory.build_retrieval(config or self.config, self.retriever, provider)

    async def aingest(self) -> IngestReport:
        """Index the configured sources; only new, changed or removed files are processed."""
        return await run_ingest(self)

    def ingest(self) -> IngestReport:
        return run_sync(self.aingest())

    async def aretrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: Filter | None = None,
        context: RequestContext | None = None,
        history: Sequence[Message] = (),
    ) -> list[ScoredChunk]:
        """The chunks the model would read for `query`: the full retrieval pipeline, top `k`."""
        result = await self.retrieval.run(
            query, history=history, context=context, filter=filter, final_k=k
        )
        return result.chunks

    def retrieve(
        self,
        query: str,
        *,
        k: int | None = None,
        filter: Filter | None = None,
        context: RequestContext | None = None,
        history: Sequence[Message] = (),
    ) -> list[ScoredChunk]:
        return run_sync(self.aretrieve(query, k=k, filter=filter, context=context, history=history))

    async def aanalyze(self) -> HealthReport:
        """Check how every configured document was read (no model, no index changes)."""
        return await analyze_documents(self)

    def analyze(self) -> HealthReport:
        return run_sync(self.aanalyze())

    async def acount(self) -> int:
        return await self.store.count()

    def count(self) -> int:
        return run_sync(self.acount())
