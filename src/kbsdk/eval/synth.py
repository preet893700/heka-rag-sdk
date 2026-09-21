"""Synthetic test-set generation: draft an evaluation dataset from the indexed documents.

A model reads a passage and writes a question, a reference answer and an exact supporting quote. Every
draft is then checked without a model where possible: the quote must appear in the passage, numbers in
the reference answer must appear in the passage, the question must not copy the passage's wording or
point at "the passage", and near-duplicates are dropped. Unanswerable questions are additionally
checked against what retrieval returns for them.

These checks make the set *usable*, not *representative*. Synthetic questions sit closer to the
documents' wording than real employees' questions do, so retrieval scores on them run optimistic, and a
model grading its own family of questions is friendlier than a stranger. Treat the output as a starter
set to read through and replace with real questions as they arrive.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from kbsdk.aio import gather_limited
from kbsdk.eval.dataset import EvalCase, EvalDataset, SourceRef
from kbsdk.filters import matches
from kbsdk.interfaces import LLM
from kbsdk.knowledge_base import KnowledgeBase
from kbsdk.prompts import build_user_message
from kbsdk.text import extract_json, quote_in_text, stable_hash
from kbsdk.types import Chunk, Message, RequestContext

COPIED_RUN_WORDS = (
    6  # a question sharing this many consecutive words with the passage is too lexical
)
CHECK_TOP_K = 8  # passages shown to the answerability check for each proposed unanswerable question
DUPLICATE_OVERLAP = 0.75  # word-set overlap above which two questions count as the same

ANSWERABLE_SYSTEM = """You write test questions for a company document assistant. You are given one \
passage from a company document. Write ONE realistic question that an employee would ask and that this \
passage answers.
Rules:
- Use everyday wording, as a person who has not read the document would. Do not reuse phrases of five \
or more words from the passage.
- The question must stand on its own: never mention "the passage", "the document", "the text" or \
"above".
- Do not put the answer in the question.
- "answer": a short reference answer using only facts stated in the passage.
- "quote": one contiguous piece of the passage, copied exactly, that contains the answer (at most two \
sentences).
If the passage has no useful fact to ask about (headings only, boilerplate, navigation), reply \
{"skip": true}.
Reply with only JSON: {"question": "...", "answer": "...", "quote": "..."}"""

FOLLOWUP_SYSTEM = """You write test conversations for a company document assistant. You are given one \
passage from a company document. Write a two-turn exchange the passage supports:
1. "question" and "answer": a first question an employee would ask, and its answer from the passage.
2. "followup": a SHORT second question that only makes sense after the first exchange (it uses \
"that", "it", "those", or leaves words out, like "and for contractors?"). It must be answered by the \
same passage.
3. "followup_answer": its answer, using only facts stated in the passage.
4. "quote": one contiguous piece of the passage, copied exactly, that contains the follow-up's answer.
Use everyday wording; never mention "the passage" or "the document".
If the passage cannot support this, reply {"skip": true}.
Reply with only JSON: {"question": "...", "answer": "...", "followup": "...", \
"followup_answer": "...", "quote": "..."}"""

UNANSWERABLE_SYSTEM = """You write test questions for a company document assistant. You are given an \
outline of the company documents it has. Write questions an employee might plausibly ask, sounding \
related to these documents, whose answers are NOT in them. Mix two kinds:
- a topic the documents do not cover at all (but a colleague could reasonably ask HR about);
- a specific detail the documents leave out about a topic they do cover (a figure, a deadline, a name).
Do not write absurd or off-topic questions (weather, trivia), and do not write questions the outline \
suggests are answered. Use everyday wording.
Reply with only JSON: {"questions": ["...", "..."]}"""

ANSWERED_SYSTEM = """You decide whether any of the numbered passages answers a question, fully or in part. Say true if a passage states, or plainly implies, information that answers it, even if the question is worded differently. A passage that is only on the same topic does not count. If you are unsure, say true.
Reply with only JSON: {"answered": true or false}"""

_WORD = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3})")
_REFERS_TO_PASSAGE = re.compile(
    r"\b(passage|excerpt)\b|\b(this|the above|given) (document|text|section)\b|\babove\b",
    re.IGNORECASE,
)


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


def copies_wording(question: str, passage: str, n: int = COPIED_RUN_WORDS) -> bool:
    """True if `question` shares `n` consecutive words with `passage`."""
    q = _words(question)
    haystack = f" {' '.join(_words(passage))} "
    return any(f" {' '.join(q[i : i + n])} " in haystack for i in range(len(q) - n + 1))


def numbers_supported(answer: str, passage: str) -> bool:
    """True if every number in `answer` also appears in `passage` (thousands separators ignored)."""
    have = set(_NUMBER.findall(_THOUSANDS.sub("", passage)))
    return all(n in have for n in _NUMBER.findall(_THOUSANDS.sub("", answer)))


def too_similar(question: str, others: Sequence[str]) -> bool:
    words = set(_words(question))
    if not words:
        return True
    return any(
        len(words & other_words) / len(words | other_words) >= DUPLICATE_OVERLAP
        for other_words in (set(_words(o)) for o in others)
        if other_words
    )


@dataclass
class SynthOptions:
    answerable: int = 30
    followups: int = 6
    unanswerable: int = 6
    test_fraction: float = 0.3
    seed: int = 0
    min_chunk_chars: int = 200
    concurrency: int = 3
    context: RequestContext | None = None  # who is asking; required when access control is on


@dataclass
class SynthResult:
    dataset: EvalDataset
    kept: Counter[str] = field(default_factory=Counter)
    dropped: Counter[str] = field(default_factory=Counter)
    attempted: Counter[str] = field(default_factory=Counter)
    sources_total: int = 0
    sources_covered: int = 0
    warnings: list[str] = field(default_factory=list)


def _source(chunk: Chunk) -> str:
    return str(chunk.metadata.get("source") or chunk.doc_id)


def _section(chunk: Chunk) -> str | None:
    path = str(chunk.metadata.get("heading_path") or "")
    return path.split(">")[-1].strip() or None


def _candidates(chunks: Sequence[Chunk], options: SynthOptions) -> list[Chunk]:
    """Usable passages, spread evenly across documents, in a seed-determined order."""
    seen: set[str] = set()
    by_source: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        text = " ".join(chunk.text.split())
        if len(text) < options.min_chunk_chars or text in seen:
            continue
        seen.add(text)
        by_source.setdefault(_source(chunk), []).append(chunk)
    for group in by_source.values():
        group.sort(key=lambda c: stable_hash(options.seed, c.id))
    ordered: list[Chunk] = []
    groups = [by_source[name] for name in sorted(by_source)]
    for rank in range(max((len(g) for g in groups), default=0)):
        ordered.extend(g[rank] for g in groups if rank < len(g))
    return ordered


async def _ask_json(llm: LLM, system: str, prompt: str) -> dict[str, Any] | None:
    for _ in range(2):
        reply = await llm.generate([Message(role="user", content=prompt)], system=system)
        parsed = extract_json(reply.text)
        if parsed is not None:
            return parsed
    return None


def _text(parsed: dict[str, Any], key: str) -> str:
    value = parsed.get(key)
    return " ".join(value.split()) if isinstance(value, str) else ""


def _is_skip(parsed: dict[str, Any]) -> bool:
    return parsed.get("skip") is True


Outcome = tuple[EvalCase | None, str | None]  # a draft case, or the reason it was dropped


def _passage_prompt(chunk: Chunk) -> str:
    title = chunk.metadata.get("context") or _source(chunk)
    return f"Document section: {title}\n\nPassage:\n{chunk.text}"


async def _draft_answerable(llm: LLM, chunk: Chunk) -> Outcome:
    parsed = await _ask_json(llm, ANSWERABLE_SYSTEM, _passage_prompt(chunk))
    if parsed is None:
        return None, "unusable_reply"
    if _is_skip(parsed):
        return None, "passage_not_suitable"
    question, answer, quote = (_text(parsed, k) for k in ("question", "answer", "quote"))
    if not (question and answer and quote):
        return None, "incomplete_reply"
    if _REFERS_TO_PASSAGE.search(question):
        return None, "refers_to_passage"
    if not quote_in_text(quote, chunk.text):
        return None, "quote_not_in_passage"
    if not numbers_supported(answer, chunk.text):
        return None, "answer_number_not_in_passage"
    if copies_wording(question, chunk.text):
        return None, "copies_passage_wording"
    case = EvalCase(
        id="pending",
        question=question,
        gold_answer=answer,
        gold_sources=[SourceRef(source=_source(chunk), section=_section(chunk), quote=quote)],
        tags=["synthetic"],
    )
    return case, None


async def _draft_followup(llm: LLM, chunk: Chunk) -> Outcome:
    parsed = await _ask_json(llm, FOLLOWUP_SYSTEM, _passage_prompt(chunk))
    if parsed is None:
        return None, "unusable_reply"
    if _is_skip(parsed):
        return None, "passage_not_suitable"
    keys = ("question", "answer", "followup", "followup_answer", "quote")
    question, answer, followup, followup_answer, quote = (_text(parsed, k) for k in keys)
    if not all((question, answer, followup, followup_answer, quote)):
        return None, "incomplete_reply"
    if _REFERS_TO_PASSAGE.search(question) or _REFERS_TO_PASSAGE.search(followup):
        return None, "refers_to_passage"
    if len(followup.split()) > 14 or followup.casefold() == question.casefold():
        return None, "followup_not_short_or_distinct"
    if not quote_in_text(quote, chunk.text):
        return None, "quote_not_in_passage"
    if not (
        numbers_supported(followup_answer, chunk.text) and numbers_supported(answer, chunk.text)
    ):
        return None, "answer_number_not_in_passage"
    if copies_wording(followup, chunk.text) or copies_wording(question, chunk.text):
        return None, "copies_passage_wording"
    case = EvalCase(
        id="pending",
        question=followup,
        gold_answer=followup_answer,
        gold_sources=[SourceRef(source=_source(chunk), section=_section(chunk), quote=quote)],
        history=[
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ],
        tags=["synthetic", "followup"],
    )
    return case, None


def _outline(chunks: Sequence[Chunk], limit_chars: int = 6000) -> str:
    headings: dict[str, list[str]] = {}
    for chunk in chunks:
        heading = str(chunk.metadata.get("heading_path") or "").strip()
        bucket = headings.setdefault(_source(chunk), [])
        if heading and heading not in bucket:
            bucket.append(heading)
    lines = [
        f"- {name}: " + "; ".join(items[:12]) if items else f"- {name}"
        for name, items in sorted(headings.items())
    ]
    return "\n".join(lines)[:limit_chars]


async def _draft_unanswerable(
    llm: LLM, kb: KnowledgeBase, question: str, context: RequestContext | None
) -> Outcome:
    hits = await kb.aretrieve(question, k=CHECK_TOP_K, context=context)
    prompt = build_user_message(question, hits)
    parsed = await _ask_json(llm, ANSWERED_SYSTEM, prompt)
    if parsed is None or not isinstance(parsed.get("answered"), bool):
        return None, "check_failed"
    if parsed["answered"]:
        return None, "actually_answerable"
    return EvalCase(
        id="pending",
        question=question,
        answerable=False,
        tags=["synthetic", "unanswerable"],
    ), None


async def _fill(
    target: int,
    pool: list[Chunk],
    attempt: Any,
    *,
    kind: str,
    existing: list[str],
    result: SynthResult,
    concurrency: int,
) -> list[tuple[Chunk, EvalCase]]:
    """Attempt passages in waves until `target` drafts pass every check or the pool runs out."""
    kept: list[tuple[Chunk, EvalCase]] = []
    while len(kept) < target and pool:
        wave = [pool.pop(0) for _ in range(min(target - len(kept), len(pool)))]
        result.attempted[kind] += len(wave)
        outcomes = await gather_limited(wave, attempt, limit=concurrency)
        for chunk, (case, reason) in zip(wave, outcomes, strict=True):
            if case is not None and too_similar(case.question, existing):
                case, reason = None, "duplicate_question"
            if case is None:
                result.dropped[f"{kind}: {reason}"] += 1
                continue
            existing.append(case.question)
            kept.append((chunk, case))
    return kept


async def generate_dataset(
    kb: KnowledgeBase, llm: LLM, options: SynthOptions | None = None, *, name: str = "synthetic"
) -> SynthResult:
    """Draft an `EvalDataset` from the documents in `kb`'s index (which must already be built)."""
    options = options or SynthOptions()
    access_filter = kb.access.filter_for(options.context)
    chunks = [
        c
        for c in await kb.store.all_chunks()
        if access_filter is None or matches(access_filter, c.metadata)
    ]
    result = SynthResult(dataset=EvalDataset(name=name, cases=[]))
    result.sources_total = len({_source(c) for c in chunks})
    pool = _candidates(chunks, options)
    if not pool:
        result.warnings.append(
            f"no passages of at least {options.min_chunk_chars} characters were found to write questions from"
        )
        return result

    existing: list[str] = []
    # Follow-ups are rarer and exercise a different code path, so they choose their passages first.
    followups = await _fill(
        options.followups,
        pool,
        lambda chunk: _draft_followup(llm, chunk),
        kind="followup",
        existing=existing,
        result=result,
        concurrency=options.concurrency,
    )

    answerable = await _fill(
        options.answerable,
        pool,
        lambda chunk: _draft_answerable(llm, chunk),
        kind="answerable",
        existing=existing,
        result=result,
        concurrency=options.concurrency,
    )

    unanswerable: list[EvalCase] = []
    if options.unanswerable > 0:
        parsed = await _ask_json(
            llm,
            UNANSWERABLE_SYSTEM,
            f"Write {options.unanswerable * 2} questions.\n\nDocuments:\n{_outline(chunks)}",
        )
        proposals = parsed.get("questions") if parsed else None
        proposed = (
            [q.strip() for q in proposals if isinstance(q, str) and q.strip()]
            if isinstance(proposals, list)
            else []
        )
        for proposal in proposed:
            if len(unanswerable) >= options.unanswerable:
                break
            if too_similar(proposal, existing):
                result.dropped["unanswerable: duplicate_question"] += 1
                continue
            case, reason = await _draft_unanswerable(llm, kb, proposal, options.context)
            if case is None:
                result.dropped[f"unanswerable: {reason}"] += 1
                continue
            existing.append(proposal)
            unanswerable.append(case)
        if not proposed:
            result.dropped["unanswerable: unusable_reply"] += 1

    drafts: list[tuple[str, EvalCase]] = (
        [("a", case) for _, case in answerable]
        + [("f", case) for _, case in followups]
        + [("u", case) for case in unanswerable]
    )
    counters: Counter[str] = Counter()
    cases: list[EvalCase] = []
    for kind, case in drafts:
        counters[kind] += 1
        case_id = f"syn-{kind}-{counters[kind]:03d}"
        test = (
            int(stable_hash(options.seed, case_id, length=8), 16) % 100
            < options.test_fraction * 100
        )
        cases.append(
            case.model_copy(
                update={
                    "id": case_id,
                    "split": "test" if test else "dev",
                    "context": options.context,
                }
            )
        )
    result.dataset = EvalDataset(name=name, cases=cases)
    result.kept.update(
        {
            "answerable": len(answerable),
            "followup": len(followups),
            "unanswerable": len(unanswerable),
        }
    )
    result.sources_covered = len({ref.source for case in cases for ref in case.gold_sources})
    for label, wanted, got in (
        ("answerable", options.answerable, len(answerable)),
        ("follow-up", options.followups, len(followups)),
        ("unanswerable", options.unanswerable, len(unanswerable)),
    ):
        if got >= wanted:
            continue
        tried = result.attempted[label.replace("follow-up", "followup")]
        if label == "unanswerable":
            result.warnings.append(
                f"only {got} of {wanted} unanswerable questions passed the checks"
            )
        elif tried < wanted or not pool:
            result.warnings.append(
                f"only {got} of {wanted} {label} questions: ran out of passages "
                f"({tried} tried); add documents, lower --min-chars or ask for fewer"
            )
        else:
            result.warnings.append(
                f"only {got} of {wanted} {label} questions passed the checks ({tried} tried)"
            )
    if result.sources_covered < result.sources_total:
        result.warnings.append(
            f"questions cover {result.sources_covered} of {result.sources_total} documents; "
            "raise the counts to reach the rest"
        )
    return result
