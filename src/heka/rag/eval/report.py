"""Result types for an evaluation run, with aggregation, threshold gating and text/markdown output."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

from heka.rag.eval.analysis import (
    OK,
    OUTCOMES,
    TAG_METRICS,
    Interval,
    by_tag,
    metric_intervals,
    outcome_breakdown,
    outcome_of,
)
from heka.rag.eval.metrics import LOWER_IS_BETTER
from heka.rag.types import Answer, Usage

# Display order; anything else is appended alphabetically.
METRIC_ORDER = [
    "correctness",
    "groundedness",
    "abstain_correct",
    "unanswerable_abstain_rate",
    "answerable_abstain_rate",
    "retrieval_hit",
    "retrieval_recall",
    "retrieval_mrr",
    "citation_valid_rate",
    "citation_source_match",
    "error_rate",
]


class CaseResult(BaseModel):
    case_id: str
    question: str
    split: str
    tags: list[str] = Field(default_factory=list)
    answerable: bool = True
    answer: Answer | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    judge_notes: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    seconds: float = 0.0

    @property
    def outcome(self) -> str:
        """Where this case failed (or "ok"); see `heka.rag.eval.analysis.OUTCOMES`."""
        return outcome_of(self)

    @property
    def problems(self) -> list[str]:
        found = []
        if self.error:
            found.append(f"error: {self.error}")
        if self.metrics.get("correctness", 1.0) < 1.0:
            found.append("answer not fully correct")
        if self.metrics.get("abstain_correct", 1.0) < 1.0:
            found.append(
                "answered an unanswerable question" if not self.answerable else "abstained"
            )
        if self.metrics.get("retrieval_hit", 1.0) < 1.0:
            found.append("right passage not retrieved")
        if self.metrics.get("groundedness", 1.0) < 1.0:
            found.append("answer not fully grounded")
        return found


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    """Mean of each metric over the cases where it applies, plus counts."""
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in result.metrics.items():
            values[name].append(value)
    summary = {name: sum(vs) / len(vs) for name, vs in values.items()}
    if results:
        summary["error_rate"] = sum(1 for r in results if r.error) / len(results)
    return summary


class EvalReport(BaseModel):
    name: str
    created_at: str
    agent: str
    agent_model: str | None = None
    judge_model: str | None = None
    results: list[CaseResult]
    summary: dict[str, float] = Field(default_factory=dict)
    by_split: dict[str, dict[str, float]] = Field(default_factory=dict)
    counts: dict[str, int] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    thresholds: dict[str, float] = Field(default_factory=dict)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool | None:
        """None when no thresholds were configured."""
        return None if not self.thresholds else not self.failures

    def check_thresholds(self) -> None:
        self.failures = []
        for metric, limit in self.thresholds.items():
            value = self.summary.get(metric)
            if value is None:
                self.failures.append(f"{metric}: not measured, cannot check against {limit:.2f}")
            elif metric in LOWER_IS_BETTER and value > limit:
                self.failures.append(f"{metric}: {value:.2f} is above the maximum {limit:.2f}")
            elif metric not in LOWER_IS_BETTER and value < limit:
                self.failures.append(f"{metric}: {value:.2f} is below the minimum {limit:.2f}")

    def compare(self, baseline: EvalReport) -> dict[str, float]:
        """Metric deltas versus an earlier run (positive = this run has the larger number)."""
        return {
            name: value - baseline.summary[name]
            for name, value in self.summary.items()
            if name in baseline.summary
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EvalReport:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    # -- analysis -----------------------------------------------------------------------------

    def intervals(self) -> dict[str, Interval]:
        """95% intervals for each metric over all cases (see `heka.rag.eval.analysis`)."""
        return metric_intervals(self.results)

    def outcomes(self) -> dict[str, list[str]]:
        """Outcome -> case ids: where each case went right or wrong."""
        return outcome_breakdown(self.results)

    def question_types(self) -> dict[str, dict[str, float]]:
        """Per tag: `n` and the mean of the headline metrics."""
        return by_tag(self.results)

    # -- rendering ----------------------------------------------------------------------------

    def _ordered(self, names: set[str]) -> list[str]:
        known = [m for m in METRIC_ORDER if m in names]
        return known + sorted(names - set(known))

    def _analysis_lines(self) -> list[str]:
        lines: list[str] = []
        intervals = self.intervals()
        shown = [m for m in self._ordered(set(intervals)) if m != "error_rate"]
        if shown:
            lines += [
                "",
                "95% intervals, all cases (approximate; graded metrics are rougher than yes/no ones)",
            ]
            for name in shown:
                i = intervals[name]
                lines.append(f"  {name:<28}{i.mean:.2f}  [{i.low:.2f}, {i.high:.2f}]  n={i.n}")
        outcomes = self.outcomes()
        if outcomes:
            total = sum(len(ids) for ids in outcomes.values())
            lines += ["", "outcomes"]
            for name, ids in outcomes.items():
                sample = ", ".join(ids[:6]) + (", ..." if len(ids) > 6 else "")
                listing = "" if name == OK else f"   {sample}"
                lines.append(f"  {name:<28}{len(ids):>4}  ({len(ids) / total:>4.0%}){listing}")
            for name in outcomes:
                if name != OK:
                    meaning, hint = OUTCOMES[name]
                    lines.append(f"    {name}: {meaning}" + (f"; look at: {hint}" if hint else ""))
        types = self.question_types()
        if types:
            columns = [m for m in TAG_METRICS if any(m in row for row in types.values())]
            lines += [
                "",
                "by question type",
                f"  {'tag':<20}{'n':>4}" + "".join(f"{c:>17}" for c in columns),
            ]
            for tag, row in types.items():
                cells = "".join(f"{row[c]:>17.2f}" if c in row else f"{'-':>17}" for c in columns)
                lines.append(f"  {tag:<20}{int(row['n']):>4}{cells}")
        return lines

    def to_text(self, baseline: EvalReport | None = None, *, worst: int = 8) -> str:
        deltas = self.compare(baseline) if baseline else {}
        lines = [
            f"Evaluation: {self.name}  |  agent: {self.agent}  |  model: {self.agent_model or '?'}"
            f"  |  judge: {self.judge_model or 'none'}",
            f"cases: {self.counts.get('cases', 0)} "
            f"({self.counts.get('answerable', 0)} answerable, {self.counts.get('unanswerable', 0)} "
            f"unanswerable, {self.counts.get('errors', 0)} errors)",
            "",
        ]
        splits = ["all", *sorted(self.by_split)]
        header = (
            f"{'metric':<28}"
            + "".join(f"{s:>9}" for s in splits)
            + ("   vs baseline" if deltas else "")
        )
        lines.append(header)
        lines.append("-" * len(header))
        names = set(self.summary).union(*(set(v) for v in self.by_split.values()))
        for name in self._ordered(names):
            row = f"{name:<28}"
            for split in splits:
                source = self.summary if split == "all" else self.by_split[split]
                row += f"{source[name]:>9.2f}" if name in source else f"{'-':>9}"
            if name in deltas:
                row += f"   {deltas[name]:+.2f}"
            lines.append(row)
        lines += self._analysis_lines()
        lines.append("")
        lines.append(
            f"tokens: {self.usage.input_tokens} in / {self.usage.output_tokens} out "
            f"| llm calls: {self.usage.llm_calls} | cache hits: {self.usage.cache_hits}"
            + (f" | cost: ${self.usage.cost_usd:.4f}" if self.usage.cost_usd is not None else "")
        )
        if self.thresholds:
            verdict = "PASSED" if self.passed else "FAILED"
            lines.append(f"thresholds: {verdict}")
            lines += [f"  - {failure}" for failure in self.failures]
        lines += [f"warning: {w}" for w in self.warnings]
        problem_cases = [r for r in self.results if r.problems][:worst]
        if problem_cases:
            lines += ["", "cases to look at:"]
            for result in problem_cases:
                lines.append(f"  {result.case_id}: {'; '.join(result.problems)}")
                lines.append(f"    Q: {result.question}")
                if result.answer is not None:
                    lines.append(f"    A: {result.answer.text[:200]}")
                for kind, note in result.judge_notes.items():
                    lines.append(f"    {kind}: {note[:200]}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        splits = ["all", *sorted(self.by_split)]
        rows = ["| metric | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
        names = set(self.summary).union(*(set(v) for v in self.by_split.values()))
        for name in self._ordered(names):
            cells = []
            for split in splits:
                source = self.summary if split == "all" else self.by_split[split]
                cells.append(f"{source[name]:.2f}" if name in source else "-")
            rows.append(f"| {name} | " + " | ".join(cells) + " |")
        head = f"# Evaluation: {self.name}\n\nagent `{self.agent}` on `{self.agent_model}`, {self.created_at}\n\n"
        text = head + "\n".join(rows) + "\n"
        intervals = self.intervals()
        shown = [m for m in self._ordered(set(intervals)) if m != "error_rate"]
        if shown:
            text += "\n## 95% intervals (all cases)\n\n| metric | mean | low | high | n |\n|---|---|---|---|---|\n"
            for name in shown:
                i = intervals[name]
                text += f"| {name} | {i.mean:.2f} | {i.low:.2f} | {i.high:.2f} | {i.n} |\n"
        outcomes = self.outcomes()
        if outcomes:
            text += "\n## Outcomes\n\n| outcome | cases | which |\n|---|---|---|\n"
            for name, ids in outcomes.items():
                which = "" if name == OK else ", ".join(ids[:12])
                text += f"| {name} | {len(ids)} | {which} |\n"
        types = self.question_types()
        if types:
            columns = [m for m in TAG_METRICS if any(m in row for row in types.values())]
            text += "\n## By question type\n\n| tag | n | " + " | ".join(columns) + " |\n"
            text += "|---|---|" + "---|" * len(columns) + "\n"
            for tag, row in types.items():
                joined = " | ".join(f"{row[c]:.2f}" if c in row else "-" for c in columns)
                text += f"| {tag} | {int(row['n'])} | {joined} |\n"
        return text
