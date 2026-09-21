"""Answer verification: a second model pass that checks the answer against its sources.

Citation checking proves the *quotes* are real; it cannot prove the answer's *claims* follow from
them (a model can quote one true sentence and then say something the sources don't support). This
pass asks the model to list any claim the sources do not support. It costs one extra model call per
answered question, and, being a model, it can be wrong in both directions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from kbsdk.interfaces import LLM
from kbsdk.prompts import build_user_message
from kbsdk.text import extract_json
from kbsdk.types import Message, ScoredChunk

VERIFY_SYSTEM = """You are a strict fact-checker. You are given numbered source passages, a question \
and a drafted answer. List every factual claim in the answer that is NOT stated in, or directly \
implied by, the sources. Judge only against the sources, not against what you know.
Do not list statements that the documents do not cover a point, or suggestions to contact someone: \
admitting missing information is not a claim. (If the sources DO address that point, the statement is \
wrong and must be listed.) Wrong numbers, limits, dates, names and conditions must always be listed.
Reply with only JSON: {"supported": true or false, "unsupported_claims": ["<claim>", ...]}
"supported" is true only if the list is empty."""


@dataclass
class Verification:
    supported: bool | None  # None: the check itself failed (unusable reply)
    unsupported_claims: list[str] = field(default_factory=list)


async def verify_answer(
    llm: LLM, question: str, answer: str, sources: Sequence[ScoredChunk]
) -> Verification:
    prompt = f"{build_user_message(question, sources)}\n\nDrafted answer to check: {answer}"
    for _ in range(2):
        reply = await llm.generate([Message(role="user", content=prompt)], system=VERIFY_SYSTEM)
        parsed = extract_json(reply.text)
        if parsed is None or not isinstance(parsed.get("supported"), bool | str):
            continue
        raw = parsed.get("unsupported_claims")
        claims = [str(c) for c in raw] if isinstance(raw, list) else []
        supported = parsed["supported"]
        if isinstance(supported, str):
            supported = supported.strip().lower() == "true"
        return Verification(supported=bool(supported) and not claims, unsupported_claims=claims)
    return Verification(supported=None)
