"""The query pipeline:

    input guardrails -> retrieve -> context guardrails -> (abstain?) -> grounded generation
    -> verified citations -> (answer verification) -> output guardrails

Each step appends a `TraceEvent`, so every answer says what was retrieved, what was asked of the model,
and how many citations survived verification. All model calls made along the way (query rewriting,
reranking, the answer itself, verification) are summed into `Answer.usage`.
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import Sequence
from typing import Any

from kbsdk.config import RAGConfig
from kbsdk.interfaces import LLM, Tracer
from kbsdk.pipelines.guardrails import GuardrailOutcome, GuardrailRunner
from kbsdk.pipelines.retrieve import RetrievalPipeline
from kbsdk.pipelines.verify import verify_answer
from kbsdk.prompts import RETRY_MESSAGE, build_system_prompt, build_user_message
from kbsdk.text import extract_json, quote_in_text
from kbsdk.types import (
    AbstainReason,
    Answer,
    Citation,
    Filter,
    Message,
    RequestContext,
    ScoredChunk,
    TraceEvent,
    Usage,
)
from kbsdk.usage import track_usage

MAX_HISTORY_MESSAGES = 6
_log = logging.getLogger("kbsdk")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)


def _source_number(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _location(chunk: ScoredChunk) -> str | None:
    metadata = chunk.chunk.metadata
    parts = [str(metadata.get("heading_path") or "")]
    if metadata.get("file_type") == "pdf" and metadata.get("page"):
        parts.append(f"p. {metadata['page']}")
    return ", ".join(p for p in parts if p) or None


def build_citations(raw: Any, retrieved: Sequence[ScoredChunk]) -> list[Citation]:
    """Turn the model's proposed citations into `Citation`s, checking each quote against its source."""
    citations: list[Citation] = []
    seen: set[tuple[str, str]] = set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        number = _source_number(item.get("source"))
        quote = str(item.get("quote") or "").strip()
        if number is None or not 1 <= number <= len(retrieved) or not quote:
            continue
        chunk = retrieved[number - 1]
        if (chunk.chunk.id, quote) in seen:
            continue
        seen.add((chunk.chunk.id, quote))
        citations.append(
            Citation(
                source=str(chunk.chunk.metadata.get("source", "")),
                chunk_id=chunk.chunk.id,
                quote=quote,
                location=_location(chunk),
                verified=quote_in_text(quote, chunk.chunk.text),
            )
        )
    return citations


class QueryPipeline:
    def __init__(
        self,
        config: RAGConfig,
        retrieval: RetrievalPipeline,
        llm: LLM,
        *,
        guardrails: GuardrailRunner | None = None,
        tracers: Sequence[Tracer] = (),
    ) -> None:
        self.config = config
        self.retrieval = retrieval
        self.llm = llm
        self.guardrails = guardrails or GuardrailRunner([])
        self.tracers = list(tracers)

    # -- outcomes that end the run early ------------------------------------------------------

    def _abstain(
        self,
        reason: AbstainReason,
        retrieved: list[ScoredChunk],
        trace: list[TraceEvent],
        model: str | None = None,
        *,
        text: str | None = None,
        escalation: str | None = None,
        warnings: Sequence[str] = (),
    ) -> Answer:
        return Answer(
            text=text or self.config.generation.abstention.message,
            abstained=True,
            abstain_reason=reason,
            escalation=escalation,
            warnings=list(warnings),
            retrieved=retrieved,
            trace=trace,
            model=model,
        )

    def _stopped(
        self,
        outcome: GuardrailOutcome,
        retrieved: list[ScoredChunk],
        trace: list[TraceEvent],
        model: str | None = None,
    ) -> Answer:
        return self._abstain(
            "guardrail_blocked",
            retrieved,
            trace,
            model,
            text=outcome.message,
            escalation=outcome.reason if outcome.action == "escalate" else None,
        )

    # -- entry point --------------------------------------------------------------------------

    async def run(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        context: RequestContext | None = None,
        filter: Filter | None = None,
    ) -> Answer:
        with track_usage() as meter:
            answer, embedding_calls = await self._run(question, history, context, filter)
        answer = answer.model_copy(
            update={"usage": meter.total + Usage(embedding_calls=embedding_calls)}
        )
        self._emit(question, answer)
        return answer

    def _emit(self, question: str, answer: Answer) -> None:
        """Send the trace to every tracer. A broken tracer must never break answering."""
        if not self.tracers:
            return
        run_id = uuid.uuid4().hex[:12]
        usage = answer.usage
        summary = TraceEvent(
            stage="answer",
            name="answer",
            data={
                "abstained": answer.abstained,
                "abstain_reason": answer.abstain_reason,
                "escalation": answer.escalation,
                "citations": len(answer.citations),
                "chunks": len(answer.retrieved),
                "model": answer.model,
                "llm_calls": usage.llm_calls,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "cost_usd": usage.cost_usd,
                "warnings": len(answer.warnings),
                "question": question,  # stripped by tracers unless include_text is set
                "answer": answer.text,
            },
        )
        for event in [*answer.trace, summary]:
            tagged = event.model_copy(update={"data": {**event.data, "run_id": run_id}})
            for tracer in self.tracers:
                try:
                    tracer.emit(tagged)
                except Exception:
                    _log.warning("tracer %s failed", type(tracer).__name__, exc_info=True)

    # -- the pipeline -------------------------------------------------------------------------

    async def _run(
        self,
        question: str,
        history: Sequence[Message],
        context: RequestContext | None,
        filter: Filter | None,
    ) -> tuple[Answer, int]:
        config = self.config
        trace: list[TraceEvent] = []
        warnings: list[str] = []

        if self.guardrails:
            checked = await self.guardrails.check(question, "input", context)
            trace.extend(checked.events)
            if checked.stopped:
                return self._stopped(checked, [], trace), 0
            question = checked.text  # PII redacted (note: `history` is passed through as given)

        result = await self.retrieval.run(question, history=history, context=context, filter=filter)
        trace.extend(result.trace)
        retrieved = result.chunks
        embeds = result.embedding_calls

        if self.guardrails and retrieved:
            kept: list[ScoredChunk] = []
            for item in retrieved:
                checked = await self.guardrails.check(item.chunk.text, "context", context)
                trace.extend(checked.events)
                if checked.stopped:
                    continue
                if checked.action == "redact":
                    chunk = item.chunk.model_copy(update={"text": checked.text})
                    item = item.model_copy(update={"chunk": chunk})
                kept.append(item)
            if len(kept) < len(retrieved):
                warnings.append(
                    f"{len(retrieved) - len(kept)} retrieved passage(s) removed by guardrails"
                )
            if not kept:
                return (
                    self._abstain("guardrail_blocked", [], trace, warnings=warnings),
                    embeds,
                )
            retrieved = kept

        abstention = config.generation.abstention
        weak = (
            not retrieved
            or len(retrieved) < abstention.min_chunks
            or (
                abstention.min_top_score is not None
                and retrieved[0].relevance < abstention.min_top_score
            )
        )
        if not retrieved or (weak and abstention.enabled):
            trace.append(
                TraceEvent(stage="abstain", name="weak_retrieval", data={"llm_called": False})
            )
            return self._abstain("no_relevant_context", retrieved, trace, warnings=warnings), embeds

        system = build_system_prompt(config.name, config.instructions)
        prior = [
            Message(role=m.role, content=m.content) for m in list(history)[-MAX_HISTORY_MESSAGES:]
        ]
        messages = [*prior, Message(role="user", content=build_user_message(question, retrieved))]

        started = time.perf_counter()
        response = await self.llm.generate(messages, system=system)
        parsed = extract_json(response.text)
        if parsed is None:
            trace.append(TraceEvent(stage="generate", name="invalid_json_retry", data={}))
            retry = [
                *messages,
                Message(role="assistant", content=response.text or "(no reply)"),
                Message(role="user", content=RETRY_MESSAGE),
            ]
            response = await self.llm.generate(retry, system=system)
            parsed = extract_json(response.text)
        trace.append(
            TraceEvent(
                stage="generate",
                name="generate",
                duration_ms=_elapsed_ms(started),
                data={
                    "model": response.model,
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "cache_hit": response.usage.cache_hits > 0,
                },
            )
        )
        if parsed is None:
            return (
                self._abstain(
                    "low_confidence", retrieved, trace, response.model, warnings=warnings
                ),
                embeds,
            )

        answerable = parsed.get("answerable", True)
        if isinstance(answerable, str):
            answerable = answerable.strip().lower() == "true"
        answer_text = str(parsed.get("answer") or "").strip()
        if not answerable or not answer_text:
            trace.append(
                TraceEvent(
                    stage="abstain", name="model_abstained", data={"reason": answer_text[:300]}
                )
            )
            return (
                self._abstain(
                    "no_relevant_context", retrieved, trace, response.model, warnings=warnings
                ),
                embeds,
            )

        citation_config = config.generation.citations
        citations: list[Citation] = []
        if citation_config.mode != "off":
            proposed = build_citations(parsed.get("citations"), retrieved)
            verified = [c for c in proposed if c.verified]
            trace.append(
                TraceEvent(
                    stage="verify",
                    name="citations",
                    data={
                        "proposed": len(proposed),
                        "verified": len(verified),
                        "dropped": len(proposed) - len(verified)
                        if citation_config.require_verified
                        else 0,
                    },
                )
            )
            if citation_config.require_verified:
                if not verified:
                    return (
                        self._abstain(
                            "unverified_claims", retrieved, trace, response.model, warnings=warnings
                        ),
                        embeds,
                    )
                citations = verified
            else:
                citations = proposed

        if config.generation.verify_answer:
            started = time.perf_counter()
            verdict = await verify_answer(self.llm, question, answer_text, retrieved)
            trace.append(
                TraceEvent(
                    stage="verify",
                    name="claims",
                    duration_ms=_elapsed_ms(started),
                    data={
                        "supported": verdict.supported,
                        "unsupported_claims": len(verdict.unsupported_claims),
                    },
                )
            )
            claims = [f"unsupported claim: {claim}" for claim in verdict.unsupported_claims]
            if verdict.supported is False:
                if config.generation.verify_action == "abstain":
                    return (
                        self._abstain(
                            "unverified_claims",
                            retrieved,
                            trace,
                            response.model,
                            warnings=[*warnings, *claims],
                        ),
                        embeds,
                    )
                warnings.extend(claims or ["the verification pass found the answer unsupported"])
            elif verdict.supported is None:
                warnings.append("answer verification was inconclusive")

        if self.guardrails:
            checked = await self.guardrails.check(answer_text, "output", context)
            trace.extend(checked.events)
            if checked.stopped:
                return self._stopped(checked, retrieved, trace, response.model), embeds
            answer_text = checked.text
            cleaned: list[Citation] = []
            for citation in citations:
                quote = await self.guardrails.check(citation.quote, "output", context)
                trace.extend(quote.events)
                if (
                    not quote.stopped
                ):  # redacted quotes stay marked verified: they were checked first
                    cleaned.append(citation.model_copy(update={"quote": quote.text}))
            citations = cleaned

        answer = Answer(
            text=answer_text,
            citations=citations,
            warnings=warnings,
            retrieved=retrieved,
            trace=trace,
            model=response.model,
        )
        return answer, embeds
