"""Runs an agent over an eval dataset and produces an `EvalReport`.

Built for rate-limited providers: bounded concurrency, per-case checkpointing (so an interrupted run
can resume), and the LLM cache underneath means re-running an unchanged setup costs nothing.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from kbsdk.agent import Agent
from kbsdk.aio import gather_limited, run_sync
from kbsdk.eval.dataset import EvalCase, EvalDataset
from kbsdk.eval.judge import LLMJudge
from kbsdk.eval.metrics import abstention_metrics, citation_metrics, retrieval_metrics
from kbsdk.eval.report import CaseResult, EvalReport, aggregate
from kbsdk.knowledge_base import KnowledgeBase
from kbsdk.types import Answer, Usage
from kbsdk.usage import track_usage

Progress = Callable[[int, int, str], None]


async def run_retrieval_eval(
    kb: KnowledgeBase,
    dataset: EvalDataset,
    *,
    k: int | None = None,
    thresholds: dict[str, float] | None = None,
) -> EvalReport:
    """Score retrieval alone: did the right passage come back in the top `k`?

    Needs no LLM (so no API key, no quota, no cost) and is deterministic, which makes it the fastest way
    to tune chunking and embeddings. Only answerable cases with `gold_sources` can be scored.
    """
    top = k or kb.config.retrieval.final_k
    results: list[CaseResult] = []
    for case in dataset.cases:
        if not case.answerable or not case.gold_sources:
            continue
        started = time.perf_counter()
        with track_usage() as meter:  # query transforms and condensation call the model
            retrieved = await kb.aretrieve(
                case.question, k=top, context=case.context, history=case.history
            )
        results.append(
            CaseResult(
                case_id=case.id,
                question=case.question,
                split=case.split,
                tags=case.tags,
                answer=Answer(text="", retrieved=retrieved, usage=meter.total),
                metrics=retrieval_metrics(case, retrieved),
                seconds=time.perf_counter() - started,
            )
        )
    by_split = {
        split: aggregate([r for r in results if r.split == split])
        for split in sorted({r.split for r in results})
    }
    scoped = {m: v for m, v in (thresholds or {}).items() if m.startswith("retrieval_")}
    report = EvalReport(
        name=dataset.name,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        agent=f"{kb.config.name} (retrieval only, top {top})",
        results=results,
        summary=aggregate(results),
        by_split=by_split if len(by_split) > 1 else {},
        counts={
            "cases": len(results),
            "answerable": len(results),
            "unanswerable": 0,
            "errors": 0,
        },
        thresholds=scoped,
        usage=sum((r.answer.usage for r in results if r.answer), Usage()),
        warnings=[
            f"retrieval only: {len(dataset.cases) - len(results)} case(s) skipped "
            "(unanswerable, or no gold_sources); no answers were generated",
        ],
    )
    report.check_thresholds()
    return report


class EvalRunner:
    def __init__(
        self,
        agent: Agent,
        *,
        judge: LLMJudge | None = None,
        concurrency: int = 2,
        sample_size: int | None = None,
        seed: int = 0,
        thresholds: dict[str, float] | None = None,
    ) -> None:
        """`sample_size` limits how many cases the LLM judge grades (deterministic given `seed`);
        the deterministic metrics always cover every case. Thresholds default to the agent config's."""
        self.agent = agent
        self.judge = judge
        self.concurrency = concurrency
        self.sample_size = sample_size
        self.seed = seed
        self.thresholds = (
            thresholds if thresholds is not None else dict(agent.config.evaluation.thresholds)
        )

    async def _run_case(self, case: EvalCase, judged: set[str]) -> CaseResult:
        started = time.perf_counter()
        result = CaseResult(
            case_id=case.id,
            question=case.question,
            split=case.split,
            tags=case.tags,
            answerable=case.answerable,
        )
        try:
            answer = await self.agent.aask(
                case.question, history=case.history, context=case.context
            )
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            result.seconds = time.perf_counter() - started
            return result

        result.answer = answer
        result.metrics.update(retrieval_metrics(case, answer.retrieved))
        result.metrics.update(citation_metrics(case, answer))
        result.metrics.update(abstention_metrics(case, answer))

        if self.judge is not None and case.id in judged:
            try:
                if case.answerable and case.gold_answer:
                    if answer.abstained:
                        result.metrics["correctness"] = 0.0
                        result.judge_notes["correctness"] = "agent abstained"
                    else:
                        score, note = await self.judge.correctness(case, answer)
                        if score is not None:
                            result.metrics["correctness"] = score
                        result.judge_notes["correctness"] = note
                if not answer.abstained:
                    score, note = await self.judge.groundedness(case, answer)
                    if score is not None:
                        result.metrics["groundedness"] = score
                    result.judge_notes["groundedness"] = note
            except Exception as exc:
                result.judge_notes["judge_error"] = f"{type(exc).__name__}: {exc}"
        result.seconds = time.perf_counter() - started
        return result

    def _choose_judged(self, dataset: EvalDataset) -> set[str]:
        ids = [c.id for c in dataset.cases]
        if self.sample_size is None or self.sample_size >= len(ids):
            return set(ids)
        return set(random.Random(self.seed).sample(ids, self.sample_size))

    async def arun(
        self,
        dataset: EvalDataset,
        *,
        run_dir: str | Path | None = None,
        resume: bool = False,
        progress: Progress | None = None,
    ) -> EvalReport:
        judged = self._choose_judged(dataset)
        checkpoint = Path(run_dir) / "results.jsonl" if run_dir else None
        done: dict[str, CaseResult] = {}
        if checkpoint and resume and checkpoint.exists():
            questions = {c.id: c.question for c in dataset.cases}
            for line in checkpoint.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    previous = CaseResult.model_validate_json(line)
                    if (
                        previous.error is None
                        and questions.get(previous.case_id) == previous.question
                    ):
                        done[previous.case_id] = previous
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            if not resume:
                checkpoint.write_text("", encoding="utf-8")

        finished = len(done)
        total = len(dataset.cases)

        async def worker(case: EvalCase) -> CaseResult:
            nonlocal finished
            if case.id in done:
                return done[case.id]
            result = await self._run_case(case, judged)
            if checkpoint:
                with checkpoint.open("a", encoding="utf-8") as handle:
                    handle.write(result.model_dump_json() + "\n")
            finished += 1
            if progress:
                progress(finished, total, case.id)
            return result

        results = await gather_limited(dataset.cases, worker, limit=self.concurrency)
        return self._report(dataset, results)

    def run(
        self,
        dataset: EvalDataset,
        *,
        run_dir: str | Path | None = None,
        resume: bool = False,
        progress: Progress | None = None,
    ) -> EvalReport:
        return run_sync(self.arun(dataset, run_dir=run_dir, resume=resume, progress=progress))

    def _report(self, dataset: EvalDataset, results: list[CaseResult]) -> EvalReport:
        usage = Usage()
        for result in results:
            if result.answer is not None:
                usage = usage + result.answer.usage
        by_split = {
            split: aggregate([r for r in results if r.split == split])
            for split in sorted({r.split for r in results})
        }
        answerable = sum(1 for r in results if r.answerable)
        report = EvalReport(
            name=dataset.name,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            agent=self.agent.config.name,
            agent_model=self.agent.llm.name,
            judge_model=self.judge.llm.name if self.judge else None,
            results=results,
            summary=aggregate(results),
            by_split=by_split if len(by_split) > 1 else {},
            counts={
                "cases": len(results),
                "answerable": answerable,
                "unanswerable": len(results) - answerable,
                "errors": sum(1 for r in results if r.error),
            },
            usage=usage,
            thresholds=self.thresholds,
            warnings=dataset.warnings(),
        )
        if self.judge is None:
            report.warnings.append("no judge model: correctness and groundedness were not measured")
        report.check_thresholds()
        return report
