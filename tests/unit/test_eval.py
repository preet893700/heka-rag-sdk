import json

import pytest

from conftest import ScriptedLLM, answering_reply, judge_reply
from heka.rag import Agent, Answer, Chunk, Citation, KnowledgeBase, ScoredChunk
from heka.rag.eval import EvalCase, EvalDataset, EvalReport, EvalRunner, LLMJudge
from heka.rag.eval.dataset import SourceRef
from heka.rag.eval.metrics import (
    abstention_metrics,
    citation_metrics,
    retrieval_metrics,
    source_matches,
)
from heka.rag.types import TraceEvent


def chunk(
    source="leave.md",
    heading_path="Leave Policy > Casual leave",
    text="12 days of casual leave",
    cid="c1",
):
    return Chunk(
        id=cid,
        doc_id="d",
        text=text,
        metadata={"source": source, "heading_path": heading_path, "context": heading_path},
    )


def scored(*chunks):
    return [ScoredChunk(chunk=c, score=1.0 - i / 10) for i, c in enumerate(chunks)]


# -- deterministic metrics ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        (SourceRef(source="leave.md"), True),
        (SourceRef(source="LEAVE.MD"), True),
        (SourceRef(source="docs/leave.md"), False),
        (SourceRef(source="expenses.md"), False),
        (SourceRef(source="leave.md", section="casual leave"), True),
        (SourceRef(source="leave.md", section="Sick leave"), False),
        # with a quote, the quote decides: the section is only a fallback (chunkers without headings)
        (
            SourceRef(
                source="leave.md", section="No such section", quote="12 days of casual leave"
            ),
            True,
        ),
        (SourceRef(source="leave.md", section="Casual leave", quote="15 days"), False),
        (SourceRef(source="leave.md", quote="12 days of casual leave"), True),
        (SourceRef(source="leave.md", quote="15 days"), False),
    ],
)
def test_source_matching(ref, expected):
    assert source_matches(ref, chunk()) is expected


def test_source_matching_handles_folders_and_slashes():
    nested = chunk(source="hr/policies/leave.md")
    assert source_matches(SourceRef(source="leave.md"), nested)
    assert source_matches(SourceRef(source="policies\\leave.md"), nested)


def test_retrieval_metrics_hit_recall_and_rank():
    case = EvalCase(
        id="q",
        question="?",
        gold_answer="a",
        gold_sources=[
            SourceRef(source="leave.md", section="Casual"),
            SourceRef(source="expenses.md"),
        ],
    )
    other = chunk(source="other.md", cid="o")
    hit = retrieval_metrics(case, scored(other, chunk(cid="a")))
    assert hit == {"retrieval_hit": 1.0, "retrieval_recall": 0.5, "retrieval_mrr": 0.5}
    miss = retrieval_metrics(case, scored(other))
    assert miss == {"retrieval_hit": 0.0, "retrieval_recall": 0.0, "retrieval_mrr": 0.0}
    assert retrieval_metrics(EvalCase(id="u", question="?", answerable=False), scored(other)) == {}
    assert retrieval_metrics(EvalCase(id="n", question="?", gold_answer="a"), scored(other)) == {}


def test_citation_metrics():
    case = EvalCase(
        id="q", question="?", gold_answer="a", gold_sources=[SourceRef(source="leave.md")]
    )
    good, bad = chunk(cid="g"), chunk(source="other.md", cid="b")
    answer = Answer(
        text="x",
        citations=[
            Citation(source="leave.md", chunk_id="g", quote="q", verified=True),
            Citation(source="other.md", chunk_id="b", quote="q", verified=True),
        ],
        retrieved=scored(good, bad),
        trace=[
            TraceEvent(
                stage="verify", name="citations", data={"proposed": 4, "verified": 2, "dropped": 2}
            )
        ],
    )
    metrics = citation_metrics(case, answer)
    assert metrics == {"citation_valid_rate": 0.5, "citation_source_match": 0.5}
    assert citation_metrics(case, Answer(text="x", abstained=True)) == {}


def test_abstention_metrics():
    answerable = EvalCase(id="a", question="?", gold_answer="x")
    unanswerable = EvalCase(id="u", question="?", answerable=False)
    abstained, answered = Answer(text="no", abstained=True), Answer(text="yes")
    assert abstention_metrics(unanswerable, abstained) == {
        "abstain_correct": 1.0,
        "unanswerable_abstain_rate": 1.0,
    }
    assert abstention_metrics(unanswerable, answered) == {
        "abstain_correct": 0.0,
        "unanswerable_abstain_rate": 0.0,
    }
    assert abstention_metrics(answerable, abstained) == {
        "abstain_correct": 0.0,
        "answerable_abstain_rate": 1.0,
    }
    assert abstention_metrics(answerable, answered) == {
        "abstain_correct": 1.0,
        "answerable_abstain_rate": 0.0,
    }


# -- runner -------------------------------------------------------------------------------------


def dataset():
    return EvalDataset(
        name="hr",
        cases=[
            EvalCase(
                id="casual",
                question="How many days of casual leave?",
                gold_answer="12 days",
                gold_sources=[SourceRef(source="leave.md", section="Casual leave")],
                split="dev",
            ),
            EvalCase(
                id="claims",
                question="When must expense claims be submitted?",
                gold_answer="30 days",
                gold_sources=[SourceRef(source="expenses.md", section="Deadlines")],
                split="test",
            ),
            EvalCase(
                id="zebra", question="zebra astronomy question", answerable=False, split="test"
            ),
        ],
    )


def make_runner(kb, agent_reply=answering_reply, judge_llm=None, **kwargs):
    agent_llm = ScriptedLLM(agent_reply)
    judge = LLMJudge(judge_llm or ScriptedLLM(judge_reply, name="fake:judge"))
    return EvalRunner(Agent(kb, llm=agent_llm), judge=judge, **kwargs), agent_llm


async def test_full_run_scores_a_good_agent(kb):
    runner, _ = make_runner(kb)
    report = await runner.arun(dataset())
    s = report.summary
    assert s["retrieval_hit"] == 1.0
    assert s["correctness"] == 1.0
    assert s["groundedness"] == 1.0  # measured on the two non-abstained answers only
    assert s["abstain_correct"] == 1.0
    assert s["unanswerable_abstain_rate"] == 1.0 and s["answerable_abstain_rate"] == 0.0
    assert s["citation_valid_rate"] == 1.0
    assert s["error_rate"] == 0.0
    assert report.counts == {"cases": 3, "answerable": 2, "unanswerable": 1, "errors": 0}
    assert set(report.by_split) == {"dev", "test"}
    assert report.judge_model == "fake:judge" and report.agent_model == "fake:scripted"
    assert report.usage.llm_calls > 0
    assert report.passed is None  # no thresholds configured


async def test_an_agent_that_always_abstains_is_caught(kb):
    always_no = lambda m, s: json.dumps({"answerable": False, "answer": "", "citations": []})  # noqa: E731
    runner, _ = make_runner(
        kb, agent_reply=always_no, thresholds={"answerable_abstain_rate": 0.1, "correctness": 0.5}
    )
    report = await runner.arun(dataset())
    assert report.summary["correctness"] == 0.0
    assert report.summary["answerable_abstain_rate"] == 1.0
    assert report.passed is False
    assert len(report.failures) == 2  # max exceeded for one metric, min missed for the other
    assert any("above the maximum" in f for f in report.failures)
    assert any("below the minimum" in f for f in report.failures)


async def test_a_hallucinating_agent_is_caught_on_unanswerable_questions(kb):
    def agent_reply(messages, system):
        from conftest import grounded_reply

        return grounded_reply(messages, system)  # answers even the zebra question

    runner, _ = make_runner(
        kb, agent_reply=agent_reply, thresholds={"unanswerable_abstain_rate": 0.9}
    )
    report = await runner.arun(dataset())
    assert report.summary["unanswerable_abstain_rate"] == 0.0
    assert report.passed is False


async def test_unmeasured_threshold_fails_rather_than_passes_silently(kb):
    runner, _ = make_runner(kb, thresholds={"no_such_metric": 0.5})
    report = await runner.arun(dataset())
    assert report.passed is False and "not measured" in report.failures[0]


async def test_thresholds_default_to_the_agent_config(make_config):
    from heka.rag import KnowledgeBase

    config = make_config(evaluation={"thresholds": {"abstain_correct": 0.99}})
    kb = KnowledgeBase(config)
    await kb.aingest()
    runner, _ = make_runner(kb)
    report = await runner.arun(dataset())
    assert report.thresholds == {"abstain_correct": 0.99} and report.passed is True


async def test_sampling_limits_judge_calls_but_not_deterministic_metrics(kb):
    judge_llm = ScriptedLLM(judge_reply, name="fake:judge")
    runner, _ = make_runner(kb, judge_llm=judge_llm, sample_size=1, seed=3)
    report = await runner.arun(dataset())
    judged = [
        r for r in report.results if "correctness" in r.metrics or "groundedness" in r.metrics
    ]
    assert len(judged) <= 1
    assert all("abstain_correct" in r.metrics for r in report.results)
    assert len(judge_llm.calls) <= 2
    again, _ = make_runner(kb, sample_size=1, seed=3)
    same = await again.arun(dataset())
    assert [r.case_id for r in same.results if "abstain_correct" in r.metrics] == [
        r.case_id for r in report.results
    ]


async def test_without_a_judge_only_deterministic_metrics_run(kb):
    runner = EvalRunner(Agent(kb, llm=ScriptedLLM(answering_reply)))
    report = await runner.arun(dataset())
    assert "correctness" not in report.summary and "retrieval_hit" in report.summary
    assert any("no judge model" in w for w in report.warnings)


async def test_agent_errors_are_recorded_not_raised(kb):
    def explode(messages, system):
        raise RuntimeError("provider down")

    runner, _ = make_runner(kb, agent_reply=explode)
    report = await runner.arun(dataset())
    assert report.summary["error_rate"] == 1.0
    assert report.counts["errors"] == 3
    assert all("provider down" in r.error for r in report.results)
    assert report.results[0].problems


async def test_a_failing_judge_does_not_lose_the_case(kb):
    def broken(messages, system):
        raise RuntimeError("judge unavailable")

    runner, _ = make_runner(kb, judge_llm=ScriptedLLM(broken, name="fake:judge"))
    report = await runner.arun(dataset())
    assert "abstain_correct" in report.results[0].metrics
    assert "judge_error" in report.results[0].judge_notes


async def test_invalid_judge_output_is_reported_not_scored(kb):
    runner, _ = make_runner(kb, judge_llm=ScriptedLLM(lambda m, s: "not json", name="fake:judge"))
    report = await runner.arun(dataset())
    assert "correctness" not in report.summary
    assert "valid verdict" in report.results[0].judge_notes["correctness"]


async def test_checkpointing_and_resume(kb, tmp_path):
    runner, agent_llm = make_runner(kb)
    first = await runner.arun(dataset(), run_dir=tmp_path / "run")
    assert (tmp_path / "run" / "results.jsonl").read_text().count("\n") == 3
    calls_before = len(agent_llm.calls)

    resumed_runner, resumed_llm = make_runner(kb)
    second = await resumed_runner.arun(dataset(), run_dir=tmp_path / "run", resume=True)
    assert resumed_llm.calls == [] and calls_before > 0  # nothing re-run
    assert second.summary == first.summary


async def test_resume_reruns_failed_cases_only(kb, tmp_path):
    def explode(messages, system):
        raise RuntimeError("down")

    broken, _ = make_runner(kb, agent_reply=explode)
    await broken.arun(dataset(), run_dir=tmp_path / "run")
    healthy, llm = make_runner(kb)
    report = await healthy.arun(dataset(), run_dir=tmp_path / "run", resume=True)
    assert report.counts["errors"] == 0 and llm.calls


async def test_progress_callback(kb):
    seen = []
    runner, _ = make_runner(kb)
    await runner.arun(dataset(), progress=lambda done, total, cid: seen.append((done, total, cid)))
    assert [d for d, _, _ in seen] == [1, 2, 3] and {c for _, _, c in seen} == {
        "casual",
        "claims",
        "zebra",
    }


async def test_retrieval_only_eval_needs_no_llm(kb):
    from heka.rag.eval import run_retrieval_eval

    report = await run_retrieval_eval(kb, dataset(), k=3)
    assert report.counts["cases"] == 2  # the unanswerable case has nothing to retrieve
    assert report.summary["retrieval_hit"] == 1.0 and "correctness" not in report.summary
    assert "retrieval only" in report.agent and "1 case(s) skipped" in report.warnings[0]
    assert all(r.answer.text == "" for r in report.results)


async def test_retrieval_only_eval_reports_model_calls_made_by_query_transforms(make_config):
    """Rewrite / multi-query / HyDE / condensation call the model even without generation; that cost
    must show up (it once read "0 llm" for variants that took minutes of model time)."""
    from heka.rag.eval import run_retrieval_eval
    from heka.rag.usage import MeteredLLM

    config = make_config(retrieval={"query_transforms": [{"provider": "rewrite"}]})
    kb = KnowledgeBase(config)
    await kb.aingest()
    kb.retrieval = kb.build_retrieval(llm=MeteredLLM(ScriptedLLM(lambda m, s: "casual leave days")))
    report = await run_retrieval_eval(kb, dataset(), k=3)
    assert report.counts["cases"] == 2
    assert report.usage.llm_calls == 2  # one rewrite per scored question
    assert all(r.answer.usage.llm_calls == 1 for r in report.results)
    plain = await run_retrieval_eval(KnowledgeBase(make_config()), dataset(), k=3)
    assert plain.usage.llm_calls == 0


async def test_retrieval_only_thresholds_ignore_answer_metrics(kb):
    from heka.rag.eval import run_retrieval_eval

    thresholds = {"retrieval_hit": 0.9, "unanswerable_abstain_rate": 0.9}  # 2nd can't be measured
    report = await run_retrieval_eval(kb, dataset(), thresholds=thresholds)
    assert report.thresholds == {"retrieval_hit": 0.9} and report.passed is True
    miss = await run_retrieval_eval(kb, dataset(), thresholds={"retrieval_hit": 1.1})
    assert miss.passed is False


def test_sync_run(kb):
    runner, _ = make_runner(kb)
    assert runner.run(dataset()).counts["cases"] == 3


# -- reports ------------------------------------------------------------------------------------


async def test_report_round_trip_compare_and_rendering(kb, tmp_path):
    good, _ = make_runner(kb)
    baseline = await good.arun(dataset())
    always_no = lambda m, s: json.dumps({"answerable": False, "answer": "", "citations": []})  # noqa: E731
    worse, _ = make_runner(kb, agent_reply=always_no)
    current = await worse.arun(dataset())

    path = tmp_path / "out" / "report.json"
    baseline.save(path)
    loaded = EvalReport.load(path)
    assert loaded.summary == baseline.summary and loaded.results[0].case_id == "casual"

    deltas = current.compare(loaded)
    assert deltas["correctness"] == -1.0 and deltas["answerable_abstain_rate"] == 1.0

    text = current.to_text(loaded)
    assert "correctness" in text and "vs baseline" in text and "-1.00" in text
    assert "cases to look at:" in text and "casual" in text and "abstained" in text
    markdown = current.to_markdown()
    assert markdown.startswith("# Evaluation: hr") and "| correctness |" in markdown


async def test_lower_is_better_metrics_pass_when_under_the_limit(kb):
    runner, _ = make_runner(kb, thresholds={"answerable_abstain_rate": 0.0, "error_rate": 0.0})
    assert (await runner.arun(dataset())).passed is True
