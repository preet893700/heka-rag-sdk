"""Built-in guardrails. All are deterministic (regex / keyword), so they are free, fast and testable.

* `pii`       detects (and redacts or blocks) emails, phone numbers, payment cards, US SSNs, IPv4.
* `injection` flags text that tries to override the assistant's instructions.
* `scope`     routes topics you never want answered automatically (legal advice, emergencies, ...).

These are defence in depth, not a security boundary: the answering prompt already tells the model to
ignore instructions inside documents, and pattern matching can be evaded. Treat them as a cheap first
filter and a safety net, and add your own `Guardrail` for anything stricter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import Field

from kbsdk.adapters._deps import Settings
from kbsdk.types import GuardrailResult, GuardrailStage, RequestContext

PiiEntity = Literal["email", "phone", "card", "ssn", "ip"]


# -- PII ----------------------------------------------------------------------------------------

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_CARD = re.compile(r"(?<![\d-])(?:\d[ -]?){12,18}\d(?![\d-])")
_SSN = re.compile(r"(?<!\d)(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}(?!\d)")
_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"
)
_PHONE = re.compile(r"(?<![\w])\+?\d[\d\s().\-]{7,18}\d(?![\w])")
_NOT_A_PHONE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$|^\d{3}-\d{2}-\d{4}$")
_DATE = re.compile(r"^\d{4}[-/.]\d{1,2}[-/.]\d{1,2}$|^\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}$")


def _default_entities() -> list[PiiEntity]:
    return ["email", "phone", "card", "ssn"]


def _stages(*names: GuardrailStage) -> list[GuardrailStage]:
    return list(names)


def _luhn(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


@dataclass
class _Span:
    start: int
    end: int
    kind: str
    text: str


def find_pii(text: str, entities: list[str]) -> list[_Span]:
    """Non-overlapping PII spans, most specific kind winning (card > ssn > email > ip > phone)."""
    found: list[_Span] = []

    def add(kind: str, pattern: re.Pattern[str]) -> None:
        for match in pattern.finditer(text):
            value = match.group()
            digits = re.sub(r"\D", "", value)
            if kind == "card" and not (13 <= len(digits) <= 19 and _luhn(digits)):
                continue
            if kind == "phone" and (
                not 9 <= len(digits) <= 15
                or _DATE.match(value.strip())
                or _NOT_A_PHONE.match(value.strip())
            ):
                continue
            found.append(_Span(match.start(), match.end(), kind, value))

    if "card" in entities:
        add("card", _CARD)
    if "ssn" in entities:
        add("ssn", _SSN)
    if "email" in entities:
        add("email", _EMAIL)
    if "ip" in entities:
        add("ip", _IPV4)
    if "phone" in entities:
        add("phone", _PHONE)
    accepted: list[_Span] = []
    for span in found:  # earlier kinds are more specific, so they claim the text first
        if not any(span.start < other.end and other.start < span.end for other in accepted):
            accepted.append(span)
    return sorted(accepted, key=lambda s: s.start)


class PiiSettings(Settings):
    entities: list[PiiEntity] = Field(default_factory=lambda: _default_entities())
    action: Literal["redact", "block"] = "redact"
    # Default is the question only: it stops personal data being sent to a model provider. Redacting
    # the answer or retrieved documents would also strip legitimate contact details (an HR mailbox).
    stages: list[GuardrailStage] = Field(default_factory=lambda: _stages("input"))
    replacement: str = "[{type}]"  # {type} becomes EMAIL, PHONE, CARD, SSN or IP
    allow: list[str] = Field(default_factory=list)  # regexes; a match on any of them is kept as-is
    message: str = (
        "Please don't include personal data such as card numbers or ID numbers in questions."
    )


class PiiGuardrail:
    settings_model = PiiSettings

    def __init__(self, settings: PiiSettings | None = None) -> None:
        self.settings = settings or PiiSettings()
        self._allow = [re.compile(p, re.IGNORECASE) for p in self.settings.allow]

    async def check(
        self, text: str, *, stage: GuardrailStage, context: RequestContext | None = None
    ) -> GuardrailResult:
        if stage not in self.settings.stages:
            return GuardrailResult()
        spans = [
            s
            for s in find_pii(text, list(self.settings.entities))
            if not any(p.search(s.text) for p in self._allow)
        ]
        if not spans:
            return GuardrailResult()
        kinds = sorted({s.kind for s in spans})
        if self.settings.action == "block":
            return GuardrailResult(
                action="block", reason=f"pii: {', '.join(kinds)}", message=self.settings.message
            )
        redacted = text
        for span in reversed(spans):
            token = self.settings.replacement.format(type=span.kind.upper())
            redacted = redacted[: span.start] + token + redacted[span.end :]
        return GuardrailResult(action="redact", reason=f"pii: {', '.join(kinds)}", text=redacted)


# -- prompt injection ---------------------------------------------------------------------------

_INJECTION_PATTERNS: dict[str, str] = {
    "override-instructions": (
        r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,25}\b(previous|prior|above|earlier|"
        r"preceding|all|any|your|these|the)\b[^.\n]{0,20}\b(instructions?|prompts?|rules|guidelines|"
        r"directions|constraints|restrictions)\b"
    ),
    "reveal-prompt": (
        # Needs "your ..." or a system/hidden/initial qualifier, so "show you the instructions for
        # the laptop" (ordinary) is not flagged but "print the system prompt" is.
        r"\b(reveal|show|print|repeat|output|display|leak|tell me)\b[^.\n]{0,25}\b("
        r"your (system |initial |hidden |original )?(prompt|instructions)"
        r"|the (system|initial|hidden|original) (prompt|instructions))\b"
    ),
    "role-reassignment": (
        r"\byou are now\b[^.\n]{0,30}\b(dan|unrestricted|unfiltered|jailbroken|in developer mode|"
        r"a different (ai|assistant|model))\b"
    ),
    "jailbreak-terms": r"\b(jailbreak|developer mode|dan mode|do anything now)\b",
    "no-restrictions": r"\b(act|behave|respond|pretend)\b[^.\n]{0,30}\b(no|without any|without)\s+(restrictions|rules|filters|limits|limitations)\b",
    "prompt-delimiter-break": r"<\s*/?\s*(system|assistant|sources?|instructions?)\s*>",
    "chat-role-markers": r"(^|\n)\s*(system|assistant)\s*:\s*\S",
}


class InjectionSettings(Settings):
    stages: list[GuardrailStage] = Field(default_factory=lambda: _stages("input", "context"))
    # Applied to the question. A retrieved chunk that matches is always dropped, never escalated.
    action: Literal["block", "escalate"] = "block"
    extra_patterns: dict[str, str] = Field(default_factory=dict)  # name -> regex
    message: str = "I can't help with that request."


class InjectionGuardrail:
    settings_model = InjectionSettings

    def __init__(self, settings: InjectionSettings | None = None) -> None:
        self.settings = settings or InjectionSettings()
        patterns = {**_INJECTION_PATTERNS, **self.settings.extra_patterns}
        self._patterns = {n: re.compile(p, re.IGNORECASE) for n, p in patterns.items()}

    async def check(
        self, text: str, *, stage: GuardrailStage, context: RequestContext | None = None
    ) -> GuardrailResult:
        if stage not in self.settings.stages:
            return GuardrailResult()
        hits = [name for name, pattern in self._patterns.items() if pattern.search(text)]
        if not hits:
            return GuardrailResult()
        action = "block" if stage != "input" else self.settings.action
        return GuardrailResult(
            action=action,
            reason=f"possible prompt injection: {', '.join(hits)}",
            message=self.settings.message,
        )


# -- scope / topic routing ----------------------------------------------------------------------


class ScopeRule(Settings):
    name: str
    keywords: list[str] = Field(default_factory=list)  # whole words / phrases, case-insensitive
    patterns: list[str] = Field(default_factory=list)  # regexes, case-insensitive
    action: Literal["block", "escalate"] = "escalate"
    message: str
    stages: list[GuardrailStage] = Field(default_factory=lambda: _stages("input"))


class ScopeSettings(Settings):
    rules: list[ScopeRule]


class ScopeGuardrail:
    """Refuse or escalate topics that must not be answered automatically (legal advice, medical
    emergencies, pay disputes, ...). The first matching rule wins. `escalate` sets `Answer.escalation`
    so your application can route the user to a person."""

    settings_model = ScopeSettings

    def __init__(self, settings: ScopeSettings) -> None:
        self.settings = settings
        self._compiled: list[tuple[ScopeRule, re.Pattern[str]]] = []
        for rule in settings.rules:
            parts = [rf"\b{re.escape(k)}\b" for k in rule.keywords] + list(rule.patterns)
            if parts:
                self._compiled.append(
                    (rule, re.compile("|".join(f"(?:{p})" for p in parts), re.IGNORECASE))
                )

    async def check(
        self, text: str, *, stage: GuardrailStage, context: RequestContext | None = None
    ) -> GuardrailResult:
        for rule, pattern in self._compiled:
            if stage in rule.stages and pattern.search(text):
                return GuardrailResult(action=rule.action, reason=rule.name, message=rule.message)
        return GuardrailResult()
