"""Runs the configured guardrails in order at one stage and combines their verdicts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from kbsdk.interfaces import Guardrail
from kbsdk.types import GuardrailStage, RequestContext, TraceEvent


@dataclass
class GuardrailOutcome:
    text: str  # the text to continue with (redactions applied)
    action: str = "allow"  # allow | redact | block | escalate
    reason: str | None = None
    message: str | None = None
    events: list[TraceEvent] = field(default_factory=list)

    @property
    def stopped(self) -> bool:
        return self.action in {"block", "escalate"}


class GuardrailRunner:
    def __init__(self, guardrails: Sequence[Guardrail]) -> None:
        self.guardrails = list(guardrails)

    def __bool__(self) -> bool:
        return bool(self.guardrails)

    async def check(
        self, text: str, stage: GuardrailStage, context: RequestContext | None = None
    ) -> GuardrailOutcome:
        outcome = GuardrailOutcome(text=text)
        for guardrail in self.guardrails:
            result = await guardrail.check(outcome.text, stage=stage, context=context)
            if result.action == "allow":
                continue
            outcome.events.append(
                TraceEvent(
                    stage="guardrail",
                    name=type(guardrail).__name__,
                    data={"at": stage, "action": result.action, "reason": result.reason},
                )
            )
            if result.action == "redact":
                if result.text is not None:
                    outcome.text = result.text
                outcome.action = "redact"
                outcome.reason = result.reason
                continue
            outcome.action, outcome.reason, outcome.message = (
                result.action,
                result.reason,
                result.message,
            )
            return outcome  # block / escalate: nothing further matters
        return outcome
