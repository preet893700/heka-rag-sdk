"""Evaluation dataset format: real questions with the answer a human expects and where it lives.

JSONL (one case per line), JSON (a list or {"cases": [...]}) and YAML are accepted. Example line:

    {"id": "leave-01", "question": "How many casual leave days do I get?",
     "gold_answer": "12 days per calendar year.",
     "gold_sources": [{"source": "leave-policy.pdf", "section": "3.2 Casual leave"}],
     "answerable": true, "split": "dev", "tags": ["leave"]}

Questions the documents cannot answer are first-class (`"answerable": false`, no gold sources): they
measure whether the agent abstains instead of guessing. Mark a share of cases `"split": "test"` and
never tune against them - they are the honest accuracy number.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kbsdk.errors import ConfigError
from kbsdk.types import Message, RequestContext


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceRef(_Model):
    source: str  # document name or path as it appears in citations
    section: str | None = None
    quote: str | None = None  # optional exact passage that supports the answer


class EvalCase(_Model):
    id: str
    question: str
    gold_answer: str | None = None
    gold_sources: list[SourceRef] = Field(default_factory=list)
    answerable: bool = True
    tags: list[str] = Field(default_factory=list)
    split: Literal["dev", "test"] = "dev"
    history: list[Message] = Field(default_factory=list)  # earlier turns, for follow-up questions
    context: RequestContext | None = None  # who is asking, for access-control cases

    @model_validator(mode="after")
    def _consistent(self) -> EvalCase:
        if not self.answerable and self.gold_sources:
            raise ValueError(f"case {self.id!r}: an unanswerable question cannot have gold_sources")
        return self


class EvalDataset(_Model):
    name: str = "dataset"
    cases: list[EvalCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> EvalDataset:
        duplicates = [i for i, n in Counter(c.id for c in self.cases).items() if n > 1]
        if duplicates:
            raise ValueError(f"duplicate case ids: {', '.join(sorted(duplicates))}")
        return self

    def subset(
        self, *, split: Literal["dev", "test"] | None = None, tags: list[str] | None = None
    ) -> EvalDataset:
        wanted = set(tags or [])
        cases = [
            c
            for c in self.cases
            if (split is None or c.split == split) and (not wanted or wanted & set(c.tags))
        ]
        return EvalDataset(name=self.name, cases=cases)

    def warnings(self) -> list[str]:
        """Things that make the dataset weaker without making it invalid."""
        found: list[str] = []
        for case in self.cases:
            if case.answerable and not case.gold_answer:
                found.append(
                    f"{case.id}: answerable but has no gold_answer (correctness can't be scored)"
                )
            if case.answerable and not case.gold_sources:
                found.append(
                    f"{case.id}: answerable but has no gold_sources (retrieval can't be scored)"
                )
        answerable = sum(c.answerable for c in self.cases)
        if self.cases and answerable == len(self.cases):
            found.append("no unanswerable questions: abstention behaviour can't be measured")
        if self.cases and not any(c.split == "test" for c in self.cases):
            found.append("no held-out 'test' split: every result is measured on tuned-against data")
        return found


def load_dataset(path: str | Path) -> EvalDataset:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read eval dataset {path}: {exc}") from exc

    raw: Any
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    elif path.suffix == ".json":
        raw = json.loads(text)
    elif path.suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(text)
    else:
        raise ConfigError(
            f"Unsupported dataset format {path.suffix!r} (use .jsonl, .json or .yaml)"
        )

    if isinstance(raw, list):
        return EvalDataset(name=path.stem, cases=raw)
    if isinstance(raw, dict):
        return EvalDataset.model_validate({"name": path.stem, **raw})
    raise ConfigError(f"{path} must contain a list of cases or a mapping with a 'cases' key")
