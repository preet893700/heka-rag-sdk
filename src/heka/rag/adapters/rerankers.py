"""Rerankers: a second, more careful model re-orders the candidates retrieval found.

Retrieval must be fast over everything, so it compares a query and a chunk only through vectors or
words. A reranker reads the query and each candidate *together*, which is far better at telling the
passage that answers the question from the ones that merely sound similar.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from heka.rag.adapters._deps import Settings, require
from heka.rag.interfaces import LLM
from heka.rag.prompts import source_label
from heka.rag.text import embedding_text, extract_json
from heka.rag.types import Message, ScoredChunk


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))


class LocalCrossEncoderSettings(Settings):
    # Apache-2.0 MiniLM. BAAI/bge-reranker-base (MIT) is stronger but ~13x larger. Avoid
    # jinaai/jina-reranker-v2-base-multilingual: its licence (CC-BY-NC) forbids commercial use.
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    cache_dir: str | None = None
    batch_size: int = 32
    max_chars: int = 2000  # each candidate is truncated to this before scoring


class LocalCrossEncoder:
    """On-device cross-encoder (fastembed/ONNX). Scores are squashed to 0..1 with a sigmoid."""

    settings_model = LocalCrossEncoderSettings

    def __init__(self, settings: LocalCrossEncoderSettings | None = None) -> None:
        self.settings = settings or LocalCrossEncoderSettings()
        module = require(
            "fastembed.rerank.cross_encoder",
            stage="reranker",
            provider="local_cross_encoder",
            extra="local",
        )
        self._cls = module.TextCrossEncoder
        self._model: Any = None

    def _score(self, query: str, documents: list[str]) -> list[float]:
        if self._model is None:
            self._model = self._cls(
                model_name=self.settings.model, cache_dir=self.settings.cache_dir
            )
        return [
            float(s)
            for s in self._model.rerank(query, documents, batch_size=self.settings.batch_size)
        ]

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], *, top_n: int
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        documents = [embedding_text(c.chunk)[: self.settings.max_chars] for c in chunks]
        raw = await asyncio.to_thread(self._score, query, documents)
        rescored = [
            ScoredChunk(
                chunk=c.chunk,
                score=_sigmoid(logit),
                retriever="rerank",
                signals={**c.signals, "rerank": _sigmoid(logit)},
            )
            for c, logit in zip(chunks, raw, strict=True)
        ]
        rescored.sort(key=lambda item: -item.score)
        return rescored[:top_n]


class LLMRerankSettings(Settings):
    max_chars: int = 700  # per candidate shown to the model


RERANK_SYSTEM = (
    "You rank passages by how directly they answer a question. Reply with only JSON: "
    '{"ranking": [<passage numbers, most relevant first>]}. Include only passages that help answer '
    "the question; leave out irrelevant ones."
)


class LLMReranker:
    """Listwise reranking with an LLM: one call ranks all candidates. Costs a model call per question,
    so prefer the local cross-encoder unless you need a stronger judge of relevance."""

    settings_model = LLMRerankSettings

    def __init__(self, settings: LLMRerankSettings, llm: LLM) -> None:
        self.settings = settings
        self.llm = llm

    async def rerank(
        self, query: str, chunks: Sequence[ScoredChunk], *, top_n: int
    ) -> list[ScoredChunk]:
        if not chunks:
            return []
        passages = "\n\n".join(
            f"[{number}] {source_label(c)}\n{c.chunk.text[: self.settings.max_chars]}"
            for number, c in enumerate(chunks, start=1)
        )
        response = await self.llm.generate(
            [Message(role="user", content=f"Question: {query}\n\nPassages:\n{passages}")],
            system=RERANK_SYSTEM,
        )
        parsed = extract_json(response.text) or {}
        ranking: list[int] = []
        for value in parsed.get("ranking", []) if isinstance(parsed.get("ranking"), list) else []:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if 1 <= number <= len(chunks) and number not in ranking:
                ranking.append(number)
        if not ranking:  # unusable reply: keep the retrieval order rather than lose candidates
            return list(chunks[:top_n])
        ordered = [chunks[n - 1] for n in ranking]
        total = len(ordered)
        return [
            ScoredChunk(
                chunk=c.chunk,
                score=1.0 - position / (total + 1),
                retriever="rerank",
                signals={**c.signals, "rerank": 1.0 - position / (total + 1)},
            )
            for position, c in enumerate(ordered[:top_n])
        ]
