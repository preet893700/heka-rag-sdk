"""Shared test doubles. Nothing here touches the network or needs API keys."""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable, Sequence
from pathlib import Path

import pytest

import heka.rag  # noqa: F401  (registers built-in adapters)
from heka.rag import KnowledgeBase, RAGConfig
from heka.rag.registry import registry
from heka.rag.types import LLMResponse, Message, Usage

Responder = Callable[[Sequence[Message], str | None], str]


def question_and_sources(messages: Sequence[Message]) -> tuple[str, list[tuple[int, str]]]:
    """Parse the user prompt the pipeline builds: numbered sources plus the question."""
    prompt = messages[-1].content
    question = prompt.rsplit("Question:", 1)[-1].strip()
    sources = [
        (int(n), body.strip())
        for n, body in re.findall(
            r"^\[(\d+)\] [^\n]*\n(.*?)(?=^\[\d+\] |\n</sources>)", prompt, re.M | re.S
        )
    ]
    return question, sources


def grounded_reply(messages: Sequence[Message], system: str | None) -> str:
    """A well-behaved model: answers from source 1 and quotes its first sentence verbatim."""
    _, sources = question_and_sources(messages)
    number, body = sources[0]
    quote = re.split(r"(?<=[.!?])\s", body.strip())[0]
    return json.dumps(
        {"answerable": True, "answer": quote, "citations": [{"source": number, "quote": quote}]}
    )


class ScriptedLLM:
    """An `LLM` that replies with whatever `responder` returns and records every call."""

    def __init__(self, responder: Responder = grounded_reply, name: str = "fake:scripted") -> None:
        self.responder = responder
        self.name = name
        self.calls: list[tuple[list[Message], str | None]] = []

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        self.calls.append((list(messages), system))
        text = self.responder(messages, system)
        return LLMResponse(
            text=text, usage=Usage(input_tokens=10, output_tokens=5, llm_calls=1), model=self.name
        )

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        yield self.responder(messages, system)


def answering_reply(messages: Sequence[Message], system: str | None) -> str:
    """Like `grounded_reply`, but declines questions about zebras (a stand-in for 'not in the docs')."""
    question, _ = question_and_sources(messages)
    if "zebra" in question.lower():
        return json.dumps({"answerable": False, "answer": "Not covered.", "citations": []})
    return grounded_reply(messages, system)


def judge_reply(messages: Sequence[Message], system: str | None) -> str:
    """A lenient judge: everything is correct and supported."""
    if system and "check whether" in system:
        return json.dumps({"verdict": "supported", "reason": "matches the sources"})
    return json.dumps({"verdict": "correct", "reason": "matches the reference"})


class ProviderLLM(ScriptedLLM):
    """Registered as the `scripted` provider so config files and the CLI can use it."""

    def __init__(self) -> None:
        super().__init__(answering_reply)


class ProviderJudge(ScriptedLLM):
    def __init__(self) -> None:
        super().__init__(judge_reply, name="fake:judge")


registry.register("llm", "scripted")(ProviderLLM)
registry.register("llm", "scripted_judge")(ProviderJudge)

CORPUS = {
    "leave.md": (
        "# Leave Policy\n\n## Casual leave\n\nEmployees receive 12 days of casual leave per year. "
        "Unused casual leave does not carry over.\n\n## Sick leave\n\nEmployees receive 10 days of "
        "paid sick leave. A medical certificate is required after 2 consecutive days.\n"
    ),
    "expenses.md": (
        "# Expense Policy\n\n## Deadlines\n\nSubmit expense claims within 30 days of the expense "
        "date. Older claims are rejected.\n\n## Approvals\n\nManagers approve claims up to 1000 "
        "dollars.\n"
    ),
}


@pytest.fixture
def docs_dir(tmp_path: Path) -> Path:
    folder = tmp_path / "docs"
    folder.mkdir()
    for name, text in CORPUS.items():
        (folder / name).write_text(text, encoding="utf-8")
    return folder


@pytest.fixture
def make_config(tmp_path: Path, docs_dir: Path) -> Callable[..., RAGConfig]:
    def build(**overrides: object) -> RAGConfig:
        data: dict[str, object] = {
            "name": "Test Agent",
            "embedder": {"provider": "hashing"},
            "knowledge": {
                "sources": [{"location": str(docs_dir)}],
                "persist_dir": str(tmp_path / "idx"),
            },
            "generation": {"llm": {"provider": "scripted"}},
            **overrides,
        }
        return RAGConfig.from_dict(data)

    return build


@pytest.fixture
def kb(make_config: Callable[..., RAGConfig]) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(make_config())
    knowledge_base.ingest()
    return knowledge_base
