"""LLM-as-judge metrics: answer correctness and groundedness.

An LLM judge is itself imperfect. Treat its scores as a fast proxy, and calibrate them by reading a
sample of its verdicts (the report includes the judge's reason for each) before trusting a number.
"""

from __future__ import annotations

from heka.rag.eval.dataset import EvalCase
from heka.rag.interfaces import LLM
from heka.rag.prompts import build_user_message
from heka.rag.text import extract_json
from heka.rag.types import Answer, Message

_VERDICT_SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
_GROUNDED_SCORES = {"supported": 1.0, "partial": 0.5, "unsupported": 0.0}

CORRECTNESS_SYSTEM = """You grade an assistant's answer against a reference answer written by a domain \
expert. Judge the facts only; ignore style, tone and length.
- "correct": the answer states the same key facts as the reference and does not contradict it. Extra \
harmless detail is fine.
- "partial": some key facts match but an essential fact is missing or wrong.
- "incorrect": it contradicts the reference or lacks the key fact.
Reply with only JSON: {"verdict": "correct" | "partial" | "incorrect", "reason": "<one sentence>"}"""

GROUNDEDNESS_SYSTEM = """You check whether an assistant's answer is supported by the source passages it \
was given. Judge only whether each factual claim in the answer follows from the sources - not whether \
it is true in the real world.
- "supported": every factual claim is stated in or directly follows from the sources.
- "partial": some claims are supported, others are not in the sources.
- "unsupported": the key claims are not in the sources.
Reply with only JSON: {"verdict": "supported" | "partial" | "unsupported", "reason": "<one sentence>"}"""


class LLMJudge:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    async def _grade(
        self, system: str, prompt: str, scores: dict[str, float]
    ) -> tuple[float | None, str]:
        parsed = None
        for _ in range(2):
            response = await self.llm.generate(
                [Message(role="user", content=prompt)], system=system
            )
            parsed = extract_json(response.text)
            if parsed is not None and str(parsed.get("verdict", "")).lower() in scores:
                break
            parsed = None
        if parsed is None:
            return None, "judge did not return a valid verdict"
        return scores[str(parsed["verdict"]).lower()], str(parsed.get("reason", "")).strip()

    async def correctness(self, case: EvalCase, answer: Answer) -> tuple[float | None, str]:
        prompt = (
            f"Question: {case.question}\n\nReference answer: {case.gold_answer}\n\n"
            f"Assistant answer: {answer.text}"
        )
        return await self._grade(CORRECTNESS_SYSTEM, prompt, _VERDICT_SCORES)

    async def groundedness(self, case: EvalCase, answer: Answer) -> tuple[float | None, str]:
        prompt = (
            f"{build_user_message(case.question, answer.retrieved)}\n\n"
            f"Assistant answer to check: {answer.text}"
        )
        return await self._grade(GROUNDEDNESS_SYSTEM, prompt, _GROUNDED_SCORES)
