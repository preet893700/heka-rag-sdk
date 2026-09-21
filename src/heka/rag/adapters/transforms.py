"""LLM-based query transforms and follow-up condensation.

Each costs one extra model call per question, so they are opt-in. They target different failures:

* `rewrite`     - the question is phrased differently from the documents ("time off" vs "leave").
* `multi_query` - one phrasing may miss; several paraphrases searched together recall more.
* `hyde`        - search with a *hypothetical answer* instead of the question; answers look like
                  documents more than questions do.
* condensation  - "what about contractors?" only makes sense with the conversation before it.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field

from heka.rag.adapters._deps import Settings
from heka.rag.interfaces import LLM
from heka.rag.text import extract_json
from heka.rag.types import Message

MAX_HISTORY = 6


class RewriteSettings(Settings):
    pass


class MultiQuerySettings(Settings):
    n: int = Field(default=3, ge=1, le=8, description="How many alternative phrasings to add")


class HydeSettings(Settings):
    max_words: int = Field(default=80, ge=20, le=300)


def _dedupe(queries: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for query in queries:
        cleaned = " ".join(query.split())
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            unique.append(cleaned)
    return unique


def _history_text(history: Sequence[Message]) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in list(history)[-MAX_HISTORY:])


class RewriteTransform:
    settings_model = RewriteSettings
    SYSTEM = (
        "Rewrite the user's question as a concise search query for a company document search. Use the "
        "vocabulary a policy document would use (e.g. 'time off' -> 'leave'; expand abbreviations). "
        "Keep every specific name, number, code and place. Reply with only the rewritten query."
    )

    def __init__(self, settings: RewriteSettings, llm: LLM) -> None:
        self.llm = llm

    async def transform(self, query: str, history: Sequence[Message]) -> list[str]:
        reply = await self.llm.generate([Message(role="user", content=query)], system=self.SYSTEM)
        return _dedupe([query, reply.text.strip().strip('"')])


class MultiQueryTransform:
    settings_model = MultiQuerySettings

    def __init__(self, settings: MultiQuerySettings, llm: LLM) -> None:
        self.settings = settings
        self.llm = llm

    async def transform(self, query: str, history: Sequence[Message]) -> list[str]:
        system = (
            f"Write {self.settings.n} different search queries that would find the answer to the user's "
            "question in company policy documents: vary the wording and vocabulary, keep every "
            'specific name, number and code. Reply with only JSON: {"queries": ["...", "..."]}'
        )
        reply = await self.llm.generate([Message(role="user", content=query)], system=system)
        parsed = extract_json(reply.text) or {}
        raw = parsed.get("queries")
        variants = [str(q) for q in raw] if isinstance(raw, list) else []
        return _dedupe([query, *variants])[: self.settings.n + 1]


class HydeTransform:
    settings_model = HydeSettings

    def __init__(self, settings: HydeSettings, llm: LLM) -> None:
        self.settings = settings
        self.llm = llm

    async def transform(self, query: str, history: Sequence[Message]) -> list[str]:
        system = (
            f"Write a short passage (under {self.settings.max_words} words) in the style of a company "
            "policy document that would answer the user's question. Invent plausible specifics; they "
            "are only used to find real documents. Reply with only the passage."
        )
        reply = await self.llm.generate([Message(role="user", content=query)], system=system)
        return _dedupe([query, reply.text.strip()])


CONDENSE_SYSTEM = (
    "Given a conversation and a follow-up question, rewrite the follow-up as one standalone question "
    "that can be understood without the conversation. Keep all specific names, numbers and codes. If "
    "it is already standalone, repeat it unchanged. Reply with only the question."
)


class Condenser:
    """Turns a follow-up into a standalone question. Skips the model call when there is no history."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def condense(self, question: str, history: Sequence[Message]) -> str:
        if not history:
            return question
        prompt = f"Conversation:\n{_history_text(history)}\n\nFollow-up question: {question}"
        reply = await self.llm.generate(
            [Message(role="user", content=prompt)], system=CONDENSE_SYSTEM
        )
        standalone = " ".join(reply.text.split()).strip('"')
        return standalone or question
