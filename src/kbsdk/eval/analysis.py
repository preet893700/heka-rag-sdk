"""Reading an evaluation honestly: how sure are the numbers, and why did the misses miss?

* Confidence intervals. With 50-100 questions a score of 0.90 could really be anywhere from about 0.80
  to 0.96, so a difference of a few points between two configs is usually noise. Yes/no metrics get a
  Wilson score interval; graded metrics (0, 0.5, 1 judge scores; reciprocal ranks) get a normal
  approximation of the mean, which is rougher for small samples.
* Outcomes. Every case is put in exactly one bucket that says where the pipeline failed (nothing
  retrieved, wrong answer despite the right passage, declined an answerable question, ...), because the
  fix for each is different.
* Question types. The same metrics broken down by case tag (lookup, follow-up, table, ...).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kbsdk.eval.report import CaseResult

Z_95 = 1.96


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int


def wilson_interval(successes: float, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion (behaves well at 0, 1 and small n)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def interval_of(values: Sequence[float], z: float = Z_95) -> Interval | None:
    """A 95% interval for the mean of `values` in [0, 1]; None when there are fewer than 2 values."""
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    if all(v in (0.0, 1.0) for v in values):
        low, high = wilson_interval(sum(values), n, z)
    else:
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        half = z * math.sqrt(variance / n)
        low, high = max(0.0, mean - half), min(1.0, mean + half)
    return Interval(mean, low, high, n)


def metric_intervals(results: Iterable[CaseResult]) -> dict[str, Interval]:
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in result.metrics.items():
            values[name].append(value)
    intervals = {}
    for name, vs in values.items():
        interval = interval_of(vs)
        if interval is not None:
            intervals[name] = interval
    return intervals


# -- outcomes ------------------------------------------------------------------------------------

OK = "ok"
ERROR = "error"
ANSWERED_UNANSWERABLE = "answered_unanswerable"
RETRIEVAL_MISS = "retrieval_miss"
WRONG_WITH_RIGHT_CONTEXT = "wrong_with_right_context"
WRONG_ABSTENTION = "wrong_abstention"
CITATION_PROBLEM = "citation_problem"

# outcome -> (what it means, where to look first)
OUTCOMES: dict[str, tuple[str, str]] = {
    OK: ("no problem found by the metrics that were run", ""),
    ERROR: (
        "the run itself failed for this case",
        "read the error; rate limits and timeouts are the usual cause",
    ),
    ANSWERED_UNANSWERABLE: (
        "answered a question the documents cannot answer (a hallucination)",
        "verify_answer, a stricter abstention setting, or a stronger model",
    ),
    RETRIEVAL_MISS: (
        "the expected passage was not among the retrieved chunks",
        "chunking, hybrid search, a reranker, query rewriting or condensation, a larger final_k; "
        "and check the document was extracted properly (kbsdk ingest --report)",
    ),
    WRONG_WITH_RIGHT_CONTEXT: (
        "the right passage was retrieved but the answer is still not correct",
        "a stronger model, the instructions, a larger final_k if the answer spans passages, "
        "or a gold answer that is itself wrong",
    ),
    WRONG_ABSTENTION: (
        "declined to answer although the right passage was retrieved",
        "the abstention rules and instructions, or the answering model being too cautious",
    ),
    CITATION_PROBLEM: (
        "the answer looks right but a citation was unverifiable or pointed at the wrong document",
        "check the quotes; the answer may be right for the wrong reason",
    ),
}


def outcome_of(result: CaseResult) -> str:
    """The single most useful label for how this case went (root cause first)."""
    m = result.metrics
    if result.error:
        return ERROR
    if not result.answerable:
        return ANSWERED_UNANSWERABLE if m.get("abstain_correct") == 0.0 else OK
    missed = m.get("retrieval_hit") == 0.0
    correctness = m.get("correctness")
    if m.get("answerable_abstain_rate") == 1.0:
        # Declined. If nothing useful was retrieved that is the root cause, not the abstention.
        return RETRIEVAL_MISS if missed else WRONG_ABSTENTION
    if correctness is not None and correctness < 1.0:
        return RETRIEVAL_MISS if missed else WRONG_WITH_RIGHT_CONTEXT
    if correctness is None and missed:
        return RETRIEVAL_MISS  # no judge ran, but retrieval alone already failed
    if m.get("citation_valid_rate", 1.0) < 1.0 or m.get("citation_source_match", 1.0) < 1.0:
        return CITATION_PROBLEM
    return OK


def outcome_breakdown(results: Iterable[CaseResult]) -> dict[str, list[str]]:
    """Outcome -> case ids, in a stable display order (ok first, then the problems)."""
    grouped: dict[str, list[str]] = defaultdict(list)
    for result in results:
        grouped[outcome_of(result)].append(result.case_id)
    return {name: grouped[name] for name in OUTCOMES if name in grouped}


# -- question types ------------------------------------------------------------------------------

TAG_METRICS = ("correctness", "retrieval_hit", "retrieval_mrr", "abstain_correct")


def by_tag(
    results: Iterable[CaseResult], metrics: Sequence[str] = TAG_METRICS
) -> dict[str, dict[str, float]]:
    """Per tag: number of cases (`n`) and the mean of each metric that was measured for it."""
    buckets: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        for tag in result.tags:
            buckets[tag].append(result)
    table: dict[str, dict[str, float]] = {}
    for tag in sorted(buckets):
        rows = buckets[tag]
        entry: dict[str, float] = {"n": float(len(rows))}
        for metric in metrics:
            values = [r.metrics[metric] for r in rows if metric in r.metrics]
            if values:
                entry[metric] = sum(values) / len(values)
        table[tag] = entry
    return table
