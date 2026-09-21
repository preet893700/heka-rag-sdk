"""Data types shared by every stage. These are the SDK's own; LangChain types never leak in here."""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Metadata filter DSL (one dialect for every vector store; each adapter translates it):
#   {"region": "EU"}                          equality
#   {"grade": {"$in": ["L3", "L4"]}}          membership
#   {"effective_date": {"$gte": "2025-01-01"}} comparison ($gt, $gte, $lt, $lte, $ne)
#   {"$and": [f1, f2]}, {"$or": [f1, f2]}     boolean combinations
Filter = dict[str, Any]

AbstainReason = Literal[
    "no_relevant_context",
    "low_confidence",
    "out_of_scope",
    "guardrail_blocked",
    "unverified_claims",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequestContext(_Model):
    """Who is asking. Set by the host application, never by the model.

    Used for access-control filters, tenant isolation and guardrails.
    """

    tenant_id: str | None = None
    user_id: str | None = None
    roles: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class Document(_Model):
    id: str
    source: str  # file path, URL or connector URI
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(_Model):
    id: str
    doc_id: str
    text: str
    index: int = 0
    parent_id: str | None = None  # set by parent-child chunking
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(_Model):
    chunk: Chunk
    score: float  # the score that ranks this result at its current stage
    retriever: str = ""  # which stage produced the score (dense, sparse, hybrid, rerank, ...)
    # Individual signals kept along the way, e.g. {"dense": 0.83, "sparse": 7.2, "rerank": 0.91}.
    signals: dict[str, float] = Field(default_factory=dict)

    @property
    def relevance(self) -> float:
        """How relevant the chunk looks, for abstention: rerank score, else dense similarity, else score."""
        return self.signals.get("rerank", self.signals.get("dense", self.score))


class Citation(_Model):
    source: str
    chunk_id: str
    quote: str  # exact span copied from the chunk
    location: str | None = None  # page, section or heading path
    verified: bool = False  # True once the quote was found verbatim in the cited chunk


class Usage(_Model):
    input_tokens: int = 0
    output_tokens: int = 0
    llm_calls: int = 0
    embedding_calls: int = 0
    cache_hits: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        cost = None
        if self.cost_usd is not None or other.cost_usd is not None:
            cost = (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            llm_calls=self.llm_calls + other.llm_calls,
            embedding_calls=self.embedding_calls + other.embedding_calls,
            cache_hits=self.cache_hits + other.cache_hits,
            cost_usd=cost,
        )


class TraceEvent(_Model):
    stage: str
    name: str
    data: dict[str, Any] = Field(default_factory=dict)
    duration_ms: float | None = None
    timestamp: float = Field(default_factory=time.time)


class Message(_Model):
    role: Literal["user", "assistant"]
    content: str


class LLMResponse(_Model):
    text: str
    usage: Usage = Field(default_factory=Usage)
    model: str | None = None
    raw: Any = Field(default=None, exclude=True)  # provider payload, never serialized


class Answer(_Model):
    text: str
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: AbstainReason | None = None
    # Set when a guardrail routed the question to a human (e.g. a medical emergency): the rule name.
    escalation: str | None = None
    # Non-fatal notes: claims a verification pass could not support, chunks dropped by a guardrail, ...
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None = None  # 0..1 when the pipeline can estimate it
    retrieved: list[ScoredChunk] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    trace: list[TraceEvent] = Field(default_factory=list)
    model: str | None = None

    @property
    def sources(self) -> list[str]:
        """Distinct cited sources, in citation order."""
        seen: dict[str, None] = {}
        for citation in self.citations:
            seen.setdefault(citation.source)
        return list(seen)


GuardrailStage = Literal[
    "input", "context", "output"
]  # question / retrieved chunk / drafted answer


class GuardrailResult(_Model):
    action: Literal["allow", "block", "redact", "escalate"] = "allow"
    reason: str | None = None
    message: str | None = None  # what to tell the user when blocked or escalated
    text: str | None = None  # replacement text when action == "redact"


class MetricResult(_Model):
    name: str
    score: float | None = None  # 0..1 where applicable
    passed: bool | None = None
    detail: str | None = None
