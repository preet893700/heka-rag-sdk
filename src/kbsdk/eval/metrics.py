"""Deterministic metrics: no LLM involved, so they are free, repeatable and never flaky.

They answer three questions independently of how well the model writes:
  * retrieval  - did the right passage come back?
  * citations  - are the quoted spans real, and do they point at the right documents?
  * abstention - does the agent decline exactly when the documents cannot answer?
"""

from __future__ import annotations

from kbsdk.eval.dataset import EvalCase, SourceRef
from kbsdk.text import quote_in_text
from kbsdk.types import Answer, Chunk, ScoredChunk

# For these metrics a smaller number is better, so thresholds act as maximums.
LOWER_IS_BETTER = {"answerable_abstain_rate", "error_rate"}


def _norm(path: str) -> str:
    return path.replace("\\", "/").casefold().strip()


def source_matches(ref: SourceRef, chunk: Chunk) -> bool:
    """Does `chunk` come from the place `ref` points at?

    The document must match. Then, if the reference has an exact `quote`, the chunk must contain it:
    that pins down the passage more precisely than a section name, and it is fair to chunkers that
    record no headings. Without a quote, the `section` (if given) must appear in the heading path.
    """
    have, want = _norm(str(chunk.metadata.get("source", ""))), _norm(ref.source)
    if not (have == want or have.endswith("/" + want) or have.rsplit("/", 1)[-1] == want):
        return False
    if ref.quote:
        return quote_in_text(ref.quote, chunk.text)
    if ref.section:
        haystack = _norm(
            f"{chunk.metadata.get('heading_path', '')} {chunk.metadata.get('context', '')}"
        )
        return _norm(ref.section) in haystack
    return True


def retrieval_metrics(case: EvalCase, retrieved: list[ScoredChunk]) -> dict[str, float]:
    if not case.answerable or not case.gold_sources:
        return {}
    found = [any(source_matches(ref, r.chunk) for r in retrieved) for ref in case.gold_sources]
    first_rank = next(
        (
            rank
            for rank, r in enumerate(retrieved, start=1)
            if any(source_matches(ref, r.chunk) for ref in case.gold_sources)
        ),
        None,
    )
    return {
        "retrieval_hit": float(any(found)),
        "retrieval_recall": sum(found) / len(found),
        "retrieval_mrr": 1.0 / first_rank if first_rank else 0.0,
    }


def citation_metrics(case: EvalCase, answer: Answer) -> dict[str, float]:
    metrics: dict[str, float] = {}
    verify = next((e for e in answer.trace if e.stage == "verify"), None)
    if verify is not None and verify.data.get("proposed"):
        metrics["citation_valid_rate"] = verify.data["verified"] / verify.data["proposed"]
    if case.answerable and case.gold_sources and answer.citations and not answer.abstained:
        by_id = {r.chunk.id: r.chunk for r in answer.retrieved}
        matching = sum(
            1
            for citation in answer.citations
            if (chunk := by_id.get(citation.chunk_id)) is not None
            and any(source_matches(ref, chunk) for ref in case.gold_sources)
        )
        metrics["citation_source_match"] = matching / len(answer.citations)
    return metrics


def abstention_metrics(case: EvalCase, answer: Answer) -> dict[str, float]:
    key = "answerable_abstain_rate" if case.answerable else "unanswerable_abstain_rate"
    return {
        "abstain_correct": float(answer.abstained != case.answerable),
        key: float(answer.abstained),
    }
