import json
import logging

import pytest

from conftest import ScriptedLLM, grounded_reply
from heka.rag import Agent, BudgetExceededError, ConfigError, KnowledgeBase, registry
from heka.rag.pipelines.verify import verify_answer
from heka.rag.types import Chunk, Message, ScoredChunk, TraceEvent

SOURCES = [
    ScoredChunk(
        chunk=Chunk(
            id="c1", doc_id="d", text="Employees get 12 days.", metadata={"source": "a.md"}
        ),
        score=1.0,
    )
]


def router(verdict):
    """Answers questions from the sources; replies to the fact-checker with `verdict`."""

    def respond(messages, system):
        if system and "fact-checker" in system:
            return verdict if isinstance(verdict, str) else json.dumps(verdict)
        return grounded_reply(messages, system)

    return respond


@pytest.fixture
def make_agent(make_config):
    async def build(llm, **overrides):
        kb = KnowledgeBase(make_config(**overrides))
        await kb.aingest()
        return Agent(kb, llm=llm)

    return build


def verifying(action="abstain", **extra):
    return {
        "generation": {
            "llm": {"provider": "scripted"},
            "verify_answer": True,
            "verify_action": action,
        },
        **extra,
    }


# -- answer verification ------------------------------------------------------------------------


async def test_verify_answer_parses_verdicts():
    llm = ScriptedLLM(lambda m, s: json.dumps({"supported": True, "unsupported_claims": []}))
    assert (await verify_answer(llm, "q", "a", SOURCES)).supported is True
    bad = ScriptedLLM(lambda m, s: json.dumps({"supported": False, "unsupported_claims": ["x"]}))
    verdict = await verify_answer(bad, "q", "a", SOURCES)
    assert verdict.supported is False and verdict.unsupported_claims == ["x"]
    contradictory = ScriptedLLM(lambda m, s: '{"supported": true, "unsupported_claims": ["y"]}')
    assert (await verify_answer(contradictory, "q", "a", SOURCES)).supported is False  # claims win
    as_text = ScriptedLLM(lambda m, s: '{"supported": "true", "unsupported_claims": []}')
    assert (await verify_answer(as_text, "q", "a", SOURCES)).supported is True


async def test_an_unusable_reply_is_inconclusive_after_one_retry():
    llm = ScriptedLLM(lambda m, s: "I think it is fine.")
    verdict = await verify_answer(llm, "q", "a", SOURCES)
    assert verdict.supported is None and len(llm.calls) == 2


async def test_a_supported_answer_passes_and_costs_one_extra_call(make_agent):
    agent = await make_agent(
        ScriptedLLM(router({"supported": True, "unsupported_claims": []})), **verifying()
    )
    answer = await agent.aask("How many days of casual leave?")
    assert not answer.abstained and answer.warnings == []
    assert answer.usage.llm_calls == 2  # the answer + the fact-check
    claims = next(e for e in answer.trace if e.name == "claims")
    assert claims.data == {"supported": True, "unsupported_claims": 0}


async def test_an_unsupported_answer_is_declined_by_default(make_agent):
    verdict = {"supported": False, "unsupported_claims": ["employees get 99 days"]}
    agent = await make_agent(ScriptedLLM(router(verdict)), **verifying())
    answer = await agent.aask("How many days of casual leave?")
    assert answer.abstained and answer.abstain_reason == "unverified_claims"
    assert answer.warnings == ["unsupported claim: employees get 99 days"]


async def test_flag_mode_keeps_the_answer_but_lists_the_problems(make_agent):
    verdict = {"supported": False, "unsupported_claims": ["employees get 99 days"]}
    agent = await make_agent(ScriptedLLM(router(verdict)), **verifying("flag"))
    answer = await agent.aask("How many days of casual leave?")
    assert not answer.abstained and answer.citations
    assert answer.warnings == ["unsupported claim: employees get 99 days"]


async def test_an_inconclusive_check_does_not_block_the_answer(make_agent):
    agent = await make_agent(ScriptedLLM(router("not json")), **verifying())
    answer = await agent.aask("How many days of casual leave?")
    assert not answer.abstained and answer.warnings == ["answer verification was inconclusive"]


async def test_verification_is_skipped_for_abstentions(make_agent):
    llm = ScriptedLLM(lambda m, s: json.dumps({"answerable": False, "answer": "", "citations": []}))
    agent = await make_agent(llm, **verifying())
    answer = await agent.aask("zebra?")
    assert answer.abstained and len(llm.calls) == 1  # no fact-check of a non-answer


# -- tracing ------------------------------------------------------------------------------------


async def test_tracers_receive_every_step_and_a_summary(make_agent):
    agent = await make_agent(ScriptedLLM(grounded_reply), tracing=[{"provider": "memory"}])
    tracer = agent.pipeline.tracers[0]
    answer = await agent.aask("How many days of casual leave?")
    stages = [e.stage for e in tracer.events]
    assert stages == [e.stage for e in answer.trace] + ["answer"]
    assert len({e.data["run_id"] for e in tracer.events}) == 1  # one id ties the steps together
    summary = tracer.events[-1].data
    assert summary["abstained"] is False and summary["citations"] == 1 and summary["llm_calls"] == 1
    await agent.aask("How many days of casual leave?")
    assert len({e.data["run_id"] for e in tracer.events}) == 2


async def test_question_and_answer_text_stay_out_of_traces_by_default(make_agent):
    agent = await make_agent(ScriptedLLM(grounded_reply), tracing=[{"provider": "memory"}])
    await agent.aask("How many days of casual leave for jo@example.com?")
    summary = agent.pipeline.tracers[0].events[-1].data
    assert "question" not in summary and "answer" not in summary
    assert "jo@example.com" not in json.dumps(
        [e.model_dump() for e in agent.pipeline.tracers[0].events]
    )


MARKER = "zz-private-marker"


def echoing_reply(messages, system):
    """Rewrites, condenses and abstains with text that quotes the user's words."""
    if "answerable" in (system or ""):
        return json.dumps(
            {"answerable": False, "answer": f"Nothing about {MARKER}", "citations": []}
        )
    return f"{MARKER} casual leave days"


def echoing_config(include_text):
    return {
        "retrieval": {"condense_followups": True, "query_transforms": [{"provider": "rewrite"}]},
        "tracing": [{"provider": "memory", "params": {"include_text": include_text}}],
    }


async def test_no_question_derived_text_reaches_any_trace_event_by_default(make_agent):
    """Condensed and rewritten queries and the model's abstention note all echo the question."""
    agent = await make_agent(ScriptedLLM(echoing_reply), **echoing_config(False))
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    await agent.aask("And for contractors?", history=history)
    events = agent.pipeline.tracers[0].events
    assert {"condense", "transform", "abstain"} <= {e.stage for e in events}  # they did run
    assert MARKER not in json.dumps([e.model_dump() for e in events])
    retrieve = next(e for e in events if e.name == "retrieve")
    assert isinstance(retrieve.data["queries"], int)  # a count is not text and is kept


async def test_include_text_keeps_the_query_and_abstention_text(make_agent):
    agent = await make_agent(ScriptedLLM(echoing_reply), **echoing_config(True))
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    await agent.aask("And for contractors?", history=history)
    dump = json.dumps([e.model_dump() for e in agent.pipeline.tracers[0].events])
    assert dump.count(MARKER) >= 3


async def test_include_text_keeps_them(make_agent):
    tracing = [{"provider": "memory", "params": {"include_text": True}}]
    agent = await make_agent(ScriptedLLM(grounded_reply), tracing=tracing)
    await agent.aask("How many days of casual leave?")
    summary = agent.pipeline.tracers[0].events[-1].data
    assert summary["question"] == "How many days of casual leave?" and summary["answer"]


async def test_jsonl_tracer_appends_lines(make_agent, tmp_path):
    path = tmp_path / "logs" / "trace.jsonl"
    agent = await make_agent(
        ScriptedLLM(grounded_reply), tracing=[{"provider": "jsonl", "params": {"path": str(path)}}]
    )
    await agent.aask("How many days of casual leave?")
    await agent.aask("How many days of casual leave?")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert (
        sum(1 for r in rows if r["stage"] == "answer") == 2
        and len({r["data"]["run_id"] for r in rows}) == 2
    )


async def test_logging_tracer(make_agent, caplog):
    agent = await make_agent(ScriptedLLM(grounded_reply), tracing=[{"provider": "logging"}])
    with caplog.at_level(logging.INFO, logger="heka.rag.trace"):
        await agent.aask("How many days of casual leave?")
    assert any(json.loads(r.message)["stage"] == "answer" for r in caplog.records)
    with pytest.raises(ValueError, match="log level"):
        registry.create("tracer", "logging", {"level": "LOUD"})


async def test_a_failing_tracer_never_breaks_answering(make_agent, caplog):
    class Broken:
        def emit(self, event):
            raise RuntimeError("collector down")

    agent = await make_agent(ScriptedLLM(grounded_reply))
    agent.pipeline.tracers = [Broken()]
    with caplog.at_level(logging.WARNING, logger="heka.rag"):
        answer = await agent.aask("How many days of casual leave?")
    assert not answer.abstained and any("tracer Broken failed" in r.message for r in caplog.records)


async def test_otel_tracer_emits_spans(make_agent):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = registry.create("tracer", "otel", {})
    tracer._tracer = provider.get_tracer("test")  # use a private provider, not the global one
    agent = await make_agent(ScriptedLLM(grounded_reply))
    agent.pipeline.tracers = [tracer]
    await agent.aask("How many days of casual leave?")
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "heka.rag.answer.answer" in spans and "heka.rag.generate.generate" in spans
    attributes = spans["heka.rag.answer.answer"].attributes
    assert attributes["heka.rag.citations"] == 1 and attributes["heka.rag.stage"] == "answer"
    assert "heka.rag.question" not in attributes  # text is scrubbed by default here too
    del trace


def test_otel_tracer_flattens_awkward_values():
    from heka.rag.adapters.tracers import _attribute

    assert (
        _attribute(3) == 3
        and _attribute(["a", "b"]) == ["a", "b"]
        and _attribute([1, 2.5]) == [1, 2.5]
    )
    assert (
        _attribute({"k": 1}) == '{"k": 1}'
        and _attribute(["a", 1]) == '["a", 1]'
        and _attribute([]) == "[]"
    )
    event = TraceEvent(stage="s", name="n", data={"question": "q"})
    from heka.rag.adapters.tracers import scrub

    assert (
        "question" not in scrub(event, include_text=False).data
        and scrub(event, include_text=True) is event
    )


# -- cost and budgets ---------------------------------------------------------------------------

PRICING = {"fake:scripted": {"input_per_mtok": 1.0, "output_per_mtok": 4.0}}


async def test_cost_is_computed_from_supplied_prices(make_agent):
    agent = await make_agent(ScriptedLLM(grounded_reply), pricing=PRICING)
    answer = await agent.aask("How many days of casual leave?")
    assert answer.usage.cost_usd == pytest.approx((10 * 1.0 + 5 * 4.0) / 1e6)  # 10 in, 5 out tokens


async def test_cost_is_unknown_without_a_price(make_agent):
    agent = await make_agent(ScriptedLLM(grounded_reply))
    assert (await agent.aask("How many days of casual leave?")).usage.cost_usd is None


async def test_every_call_in_a_question_is_priced(make_agent):
    agent = await make_agent(
        ScriptedLLM(router({"supported": True, "unsupported_claims": []})),
        pricing=PRICING,
        **verifying(),
    )
    answer = await agent.aask("How many days of casual leave?")
    assert answer.usage.llm_calls == 2 and answer.usage.cost_usd == pytest.approx(2 * 30 / 1e6)


async def test_a_dollar_budget_needs_a_price(make_agent):
    with pytest.raises(ConfigError, match="needs a price for 'fake:scripted'"):
        await make_agent(ScriptedLLM(grounded_reply), budget={"max_total_usd": 1.0})


async def test_the_budget_stops_further_model_calls(make_agent):
    # each answer costs 0.00003; a 0.00005 budget allows two answers to start, then refuses
    agent = await make_agent(
        ScriptedLLM(grounded_reply), pricing=PRICING, budget={"max_total_usd": 0.00005}
    )
    await agent.aask("How many days of casual leave?")
    await agent.aask("How many days of casual leave?")
    with pytest.raises(BudgetExceededError, match="budget is used up"):
        await agent.aask("How many days of casual leave?")
    assert agent.budget.spent_usd == pytest.approx(60 / 1e6)


async def test_a_per_question_call_cap_stops_runaway_loops(make_agent):
    llm = ScriptedLLM(router({"supported": True, "unsupported_claims": []}))
    agent = await make_agent(llm, budget={"max_llm_calls_per_question": 1}, **verifying())
    with pytest.raises(BudgetExceededError, match="already made 1 model calls"):
        await agent.aask("How many days of casual leave?")
    # the cap is per question: the next question starts from zero (and stops at the same point)
    with pytest.raises(BudgetExceededError):
        await agent.aask("How many days of casual leave?")


async def test_eval_reports_show_cost(make_agent):
    from heka.rag.eval import EvalCase, EvalDataset, EvalRunner, SourceRef

    agent = await make_agent(ScriptedLLM(grounded_reply), pricing=PRICING)
    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="a",
                question="How many days of casual leave?",
                gold_answer="12",
                gold_sources=[SourceRef(source="leave.md")],
            )
        ]
    )
    report = await EvalRunner(agent).arun(dataset)
    assert report.usage.cost_usd == pytest.approx(30 / 1e6) and "cost: $0.0000" in report.to_text()
