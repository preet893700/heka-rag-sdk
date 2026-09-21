"""Builds concrete components from a `RAGConfig` via the registry, wiring in reliability features."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path
from typing import Any

from kbsdk.adapters.agentic import AgenticRetriever
from kbsdk.adapters.embedders import ResilientEmbedder
from kbsdk.adapters.llms import ResilientLLM
from kbsdk.adapters.retrievers import DenseRetriever, HybridRetriever, SparseRetriever
from kbsdk.adapters.transforms import Condenser
from kbsdk.config import ComponentConfig, RAGConfig
from kbsdk.env import load_env_file
from kbsdk.errors import ConfigError
from kbsdk.interfaces import (
    LLM,
    Cache,
    Chunker,
    Embedder,
    OCREngine,
    Reranker,
    Retriever,
    Tracer,
    VectorStore,
)
from kbsdk.pipelines.guardrails import GuardrailRunner
from kbsdk.pipelines.retrieve import RetrievalPipeline
from kbsdk.registry import registry
from kbsdk.resilience import shared_limiter
from kbsdk.text import slugify
from kbsdk.types import LLMResponse, Message

_ON_DEVICE_EMBEDDERS = {"local", "hashing"}  # cheap to recompute, so not worth caching


def validate_supported(config: RAGConfig) -> None:
    """Fail early and clearly on options that are in the schema but not built yet."""
    pending: list[str] = []
    if config.knowledge.update_mode == "versioned":
        pending.append("knowledge.update_mode='versioned' [later]")
    if config.knowledge.extraction == "multimodal":
        pending.append("knowledge.extraction='multimodal' [Phase 4]")
    if config.generation.citations.mode == "native":
        pending.append(
            "generation.citations.mode='native' (needs a live provider to verify; use 'sdk')"
        )
    if pending:
        raise ConfigError(
            "This configuration uses options that are not available yet:\n  - "
            + "\n  - ".join(pending)
            + "\nRemove them from the config."
        )


def build_guardrails(config: RAGConfig) -> GuardrailRunner:
    return GuardrailRunner(
        [registry.create("guardrail", g.provider, g.params) for g in config.guardrails]
    )


def build_tracers(config: RAGConfig) -> list[Tracer]:
    return [registry.create("tracer", t.provider, t.params) for t in config.tracing]


def apply_env_file(config: RAGConfig) -> None:
    """Load the config's opt-in `env_file`, if it names one. A no-op otherwise."""
    if config.env_file:
        load_env_file(config.env_file)


def _model_key(component: ComponentConfig) -> str:
    return f"{component.provider}:{component.params.get('model', '')}"


def build_cache(config: RAGConfig) -> Cache | None:
    if config.cache is None:
        return None
    params: dict[str, Any] = dict(config.cache.params)
    if config.cache.provider == "disk":
        params.setdefault("path", str(Path(config.knowledge.persist_dir) / "cache.sqlite"))
    cache: Cache = registry.create("cache", config.cache.provider, params)
    return cache


def build_embedder(config: RAGConfig, cache: Cache | None) -> Embedder:
    component = config.embedder
    inner: Embedder = registry.create("embedder", component.provider, component.params)
    return ResilientEmbedder(
        inner,
        cache=None if component.provider in _ON_DEVICE_EMBEDDERS else cache,
        max_retries=config.reliability.max_retries,
    )


def build_store(config: RAGConfig, kb_dir: Path) -> VectorStore:
    params: dict[str, Any] = dict(config.store.params)
    if config.store.provider == "local":
        params.setdefault("path", str(kb_dir / "index"))
    elif config.store.provider == "qdrant":
        if "url" not in params:  # embedded and on disk, unless a server is named
            params.setdefault("location", str(kb_dir / "qdrant"))
        params.setdefault("collection", slugify(config.name))  # agents sharing a server stay apart
    store: VectorStore = registry.create("store", config.store.provider, params)
    return store


def build_chunker(config: RAGConfig, embedder: Embedder | None = None) -> Chunker:
    """Some chunkers (semantic) need the embedder; it is passed only to those that ask for it."""
    chunker: Chunker = registry.create(
        "chunker",
        config.chunker.provider,
        config.chunker.params,
        deps={"embedder": embedder} if embedder is not None else None,
    )
    return chunker


def build_ocr(config: RAGConfig) -> OCREngine | None:
    if config.knowledge.ocr is None:
        return None
    engine: OCREngine = registry.create(
        "ocr", config.knowledge.ocr.provider, config.knowledge.ocr.params
    )
    return engine


def build_retriever(config: RAGConfig, store: VectorStore, embedder: Embedder) -> Retriever:
    settings = config.retrieval
    if settings.mode == "dense":
        return DenseRetriever(store, embedder)
    if settings.mode == "sparse":
        return SparseRetriever(store)
    return HybridRetriever(
        DenseRetriever(store, embedder),
        SparseRetriever(store),
        fusion=settings.fusion,
        rrf_k=settings.rrf_k,
        dense_weight=settings.dense_weight,
    )


class LazyLLM:
    """An LLM that is only built (and so only needs its API key) when it is first used.

    Retrieval-only workflows (`inspect`, `eval --retrieval-only`) never call the model, so they work
    without credentials even when the config enables an LLM-based feature.
    """

    def __init__(self, provider: Callable[[], LLM]) -> None:
        self._provider = provider
        self._inner: LLM | None = None

    def _get(self) -> LLM:
        if self._inner is None:
            self._inner = self._provider()
        return self._inner

    @property
    def name(self) -> str:
        return self._inner.name if self._inner is not None else "lazy"

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        return await self._get().generate(
            messages, system=system, temperature=temperature, max_output_tokens=max_output_tokens
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        async for piece in self._get().stream(
            messages, system=system, temperature=temperature, max_output_tokens=max_output_tokens
        ):
            yield piece


def build_retrieval(
    config: RAGConfig, retriever: Retriever, llm_provider: Callable[[], LLM]
) -> RetrievalPipeline:
    """Assemble condense -> transform -> retrieve -> rerank from config.

    `llm_provider` supplies the model for LLM-based steps; it is only called if one is used.
    """
    settings = config.retrieval
    llm = LazyLLM(llm_provider)
    deps = {"llm": llm}
    if settings.agentic is not None:
        retriever = AgenticRetriever(retriever, llm, max_steps=settings.agentic.max_steps)
    reranker: Reranker | None = None
    if settings.reranker is not None:
        reranker = registry.create(
            "reranker", settings.reranker.provider, settings.reranker.params, deps=deps
        )
    transforms = [
        registry.create("query_transform", t.provider, t.params, deps=deps)
        for t in settings.query_transforms
    ]
    return RetrievalPipeline(
        config,
        retriever,
        reranker=reranker,
        transforms=transforms,
        condenser=Condenser(llm) if settings.condense_followups else None,
    )


def build_llm(
    config: RAGConfig,
    component: ComponentConfig,
    cache: Cache | None,
    *,
    defaults: dict[str, Any] | None = None,
    with_fallbacks: bool = True,
) -> LLM:
    """Build an LLM wrapped with caching, throttling, retries and (optionally) fallbacks.

    `defaults` (e.g. the config's temperature) are only passed to providers whose settings declare
    them, so a plug-in that knows nothing about a default is never handed an argument it can't take.
    """
    settings_model = getattr(registry.resolve("llm", component.provider), "settings_model", None)
    declared = set(settings_model.model_fields) if settings_model is not None else set()
    params = {**{k: v for k, v in (defaults or {}).items() if k in declared}, **component.params}
    inner: LLM = registry.create("llm", component.provider, params)
    fallbacks = (
        [
            build_llm(config, fallback, cache, defaults=defaults, with_fallbacks=False)
            for fallback in config.reliability.fallback_llms
        ]
        if with_fallbacks
        else []
    )
    return ResilientLLM(
        inner,
        cache=cache,
        limiter=shared_limiter(
            f"llm:{_model_key(component)}", config.reliability.requests_per_minute
        ),
        max_retries=config.reliability.max_retries,
        fallbacks=fallbacks,
    )


def generation_defaults(config: RAGConfig) -> dict[str, Any]:
    generation = config.generation
    defaults: dict[str, Any] = {"temperature": generation.temperature}
    if generation.max_output_tokens is not None:
        defaults["max_output_tokens"] = generation.max_output_tokens
    return defaults
