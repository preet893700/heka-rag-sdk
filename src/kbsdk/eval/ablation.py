"""Ablation: score the same questions under several configs, so each lever's effect is measured.

    baseline:      {}
    hybrid:        {retrieval: {mode: hybrid}}
    hybrid+rerank: {retrieval: {mode: hybrid, reranker: {provider: local_cross_encoder}}}

Each variant is a set of overrides deep-merged onto the base config and gets its own index folder,
so switching chunkers or embedders never makes variants rebuild each other's indexes. The first
variant is the baseline that deltas are measured against.
"""

from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from kbsdk import factory
from kbsdk.agent import Agent
from kbsdk.config import RAGConfig, deep_merge
from kbsdk.eval.dataset import EvalDataset
from kbsdk.eval.judge import LLMJudge
from kbsdk.eval.report import EvalReport
from kbsdk.eval.runner import EvalRunner, run_retrieval_eval
from kbsdk.knowledge_base import KnowledgeBase
from kbsdk.text import slugify

HEADLINE = [
    "retrieval_hit",
    "retrieval_mrr",
    "correctness",
    "groundedness",
    "abstain_correct",
    "answerable_abstain_rate",
    "unanswerable_abstain_rate",
]


class VariantResult(BaseModel):
    name: str
    overrides: dict[str, Any]
    report: EvalReport | None = None
    error: str | None = None
    chunks: int = 0
    llm_calls: int = 0
    seconds: float = 0.0


class AblationReport(BaseModel):
    dataset: str
    created_at: str
    retrieval_only: bool
    variants: list[VariantResult]

    def _ok(self) -> list[VariantResult]:
        return [v for v in self.variants if v.report is not None]

    def metric_names(self) -> list[str]:
        present: set[str] = set()
        for variant in self._ok():
            assert variant.report is not None
            present.update(variant.report.summary)
        return [m for m in HEADLINE if m in present] + sorted(
            present - set(HEADLINE) - {"error_rate"}
        )

    def tag_scores(self, metric: str = "retrieval_mrr") -> dict[str, dict[str, float]]:
        """metric averaged per question tag, per variant: {tag: {variant: value}}."""
        table: dict[str, dict[str, float]] = defaultdict(dict)
        for variant in self._ok():
            assert variant.report is not None
            by_tag: dict[str, list[float]] = defaultdict(list)
            for result in variant.report.results:
                if metric in result.metrics:
                    for tag in result.tags:
                        by_tag[tag].append(result.metrics[metric])
            for tag, values in by_tag.items():
                table[tag][variant.name] = sum(values) / len(values)
        return dict(sorted(table.items()))

    def to_text(self, *, tag_metric: str = "retrieval_mrr") -> str:
        ok = self._ok()
        names = [v.name for v in ok]
        width = max([12, *(len(n) + 2 for n in names)])
        lines = [
            f"Ablation on '{self.dataset}' ({'retrieval only' if self.retrieval_only else 'full pipeline'})",
            "",
        ]

        def row(label: str, values: dict[str, float | None], baseline: float | None) -> str:
            cells = []
            for name in names:
                value = values.get(name)
                if value is None:
                    cells.append(f"{'-':>{width}}")
                    continue
                delta = (
                    "" if baseline is None or name == names[0] else f" ({value - baseline:+.2f})"
                )
                cells.append(f"{f'{value:.2f}{delta}':>{width}}")
            return f"{label:<28}" + "".join(cells)

        header = f"{'metric':<28}" + "".join(f"{n:>{width}}" for n in names)
        lines += [header, "-" * len(header)]
        for metric in self.metric_names():
            values = {v.name: (v.report.summary.get(metric) if v.report else None) for v in ok}
            lines.append(row(metric, values, values.get(names[0]) if names else None))

        tags = self.tag_scores(tag_metric)
        if tags:
            lines += ["", f"{tag_metric} by question type", "-" * len(header)]
            for tag, scores in tags.items():
                lines.append(row(tag, dict(scores), scores.get(names[0])))

        lines += [
            "",
            f"{'cost':<28}"
            + "".join(f"{v.llm_calls} llm/{v.seconds:.0f}s".rjust(width) for v in ok),
        ]
        for variant in self.variants:
            if variant.report is None:
                lines.append(f"SKIPPED {variant.name}: {variant.error}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        ok = self._ok()
        names = [v.name for v in ok]
        lines = [
            f"# Ablation: {self.dataset}",
            "",
            f"{'Retrieval only' if self.retrieval_only else 'Full pipeline'}, {self.created_at}",
            "",
            "| metric | " + " | ".join(names) + " |",
            "|---|" + "---|" * len(names),
        ]
        for metric in self.metric_names():
            cells = [
                f"{v.report.summary[metric]:.2f}"
                if v.report and metric in v.report.summary
                else "-"
                for v in ok
            ]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=1), encoding="utf-8")


def variant_config(base: RAGConfig, name: str, overrides: dict[str, Any]) -> RAGConfig:
    """`base` + `overrides`, with an index folder of its own (and a shared response cache)."""
    data = deep_merge(base.to_dict(), overrides)
    root = base.knowledge.persist_dir
    data.setdefault("knowledge", {})["persist_dir"] = str(Path(root) / f"ablation-{slugify(name)}")
    cache = data.get("cache")
    if isinstance(cache, dict) and cache.get("provider") == "disk":
        cache.setdefault("params", {}).setdefault("path", str(Path(root) / "cache.sqlite"))
    return RAGConfig.from_dict(data)


async def run_ablation(
    base: RAGConfig,
    variants: dict[str, dict[str, Any]],
    dataset: EvalDataset,
    *,
    retrieval_only: bool = True,
    k: int | None = None,
    concurrency: int = 2,
    sample_size: int | None = None,
    no_judge: bool = False,
    progress: Callable[[str], None] | None = None,
) -> AblationReport:
    """Run every variant. A variant that cannot run (missing key, unbuilt option) is reported as
    skipped rather than aborting the whole comparison."""
    results: list[VariantResult] = []
    judge: LLMJudge | None = None
    for name, overrides in variants.items():
        if progress:
            progress(f"variant: {name}")
        started = time.perf_counter()
        outcome = VariantResult(name=name, overrides=overrides)
        try:
            config = variant_config(base, name, overrides)
            kb = KnowledgeBase(config)
            await kb.aingest()
            outcome.chunks = await kb.acount()
            if retrieval_only:
                report = await run_retrieval_eval(kb, dataset, k=k)
            else:
                if judge is None and not no_judge:
                    component = base.evaluation.judge_llm or base.generation.llm
                    judge = LLMJudge(
                        factory.build_llm(base, component, kb.cache, defaults={"temperature": 0.0})
                    )
                runner = EvalRunner(
                    Agent(kb, config), judge=judge, concurrency=concurrency, sample_size=sample_size
                )
                report = await runner.arun(dataset)
            outcome.report = report
            outcome.llm_calls = report.usage.llm_calls
        except Exception as exc:  # one broken variant must not lose the others' results
            outcome.error = f"{type(exc).__name__}: {exc}"
        outcome.seconds = time.perf_counter() - started
        results.append(outcome)
    return AblationReport(
        dataset=dataset.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        retrieval_only=retrieval_only,
        variants=results,
    )
