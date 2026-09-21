import json

import pytest

pytest.importorskip("langgraph")

from conftest import ScriptedLLM, grounded_reply  # noqa: E402
from kbsdk import Agent, BudgetExceededError, KnowledgeBase, RequestContext  # noqa: E402
from kbsdk.adapters.agentic import AgenticRetriever  # noqa: E402
from kbsdk.adapters.retrievers import SparseRetriever  # noqa: E402

PEOPLE = "# People\n\n## Alice\n\nAlice Novak works at the Zurich site.\n"
SITES = (
    "# Sites\n\n## Zurich site\n\nThe Zurich site opens at 7am. The Zurich site closes at 8pm. "
    "The Zurich site badge desk is on level one.\n"
)
QUESTION = "What are the opening hours where Alice Novak works?"


@pytest.fixture
async def hop_kb(make_config, tmp_path):
    docs = tmp_path / "hop-docs"
    docs.mkdir()
    (docs / "people.md").write_text(PEOPLE, encoding="utf-8")
    (docs / "sites.md").write_text(SITES, encoding="utf-8")
    config = make_config(
        knowledge={"sources": [{"location": str(docs)}], "persist_dir": str(tmp_path / "hop-idx")},
        retrieval={"mode": "sparse", "top_k": 5, "final_k": 5},
    )
    kb = KnowledgeBase(config)
    await kb.aingest()
    return kb


def verdicts(*replies):
    """An LLM that answers the assessor with each reply in turn (then 'sufficient')."""
    queue = list(replies)

    def respond(messages, system):
        return json.dumps(queue.pop(0) if queue else {"sufficient": True, "next_query": ""})

    return ScriptedLLM(respond)


def sources(results):
    return {r.chunk.metadata["source"] for r in results}


async def test_one_search_is_not_enough_for_a_two_document_question(hop_kb):
    base = SparseRetriever(hop_kb.store)
    single = await base.retrieve(QUESTION, k=1)
    assert sources(single) == {"people.md"}  # the first hop only finds Alice's office

    llm = verdicts({"sufficient": False, "next_query": "Zurich site opens closes"})
    agentic = AgenticRetriever(base, llm, max_steps=3)
    found = await agentic.retrieve(QUESTION, k=2)
    assert sources(found) == {"people.md", "sites.md"}  # the second hop finds the opening hours
    assert len(llm.calls) == 2  # asked twice: after hop 1 (needs more), after hop 2 (enough)
    assert "Zurich site" in llm.calls[0][0][-1].content  # it sees what the first search found


async def test_it_stops_immediately_when_the_first_search_is_enough(hop_kb):
    llm = verdicts({"sufficient": True, "next_query": ""})
    found = await AgenticRetriever(SparseRetriever(hop_kb.store), llm).retrieve(QUESTION, k=2)
    assert len(llm.calls) == 1 and sources(found) == {"people.md"}


async def test_max_steps_bounds_the_loop(hop_kb):
    llm = verdicts(*[{"sufficient": False, "next_query": f"query {i}"} for i in range(10)])
    agentic = AgenticRetriever(SparseRetriever(hop_kb.store), llm, max_steps=2)
    await agentic.retrieve(QUESTION, k=2)
    assert len(llm.calls) == 1  # 2 searches, judged once: the second search ends the loop unjudged


async def test_a_repeated_query_ends_the_loop_instead_of_spinning(hop_kb):
    llm = verdicts({"sufficient": False, "next_query": QUESTION.upper()})  # same as the original
    await AgenticRetriever(SparseRetriever(hop_kb.store), llm, max_steps=5).retrieve(QUESTION, k=2)
    assert len(llm.calls) == 1


@pytest.mark.parametrize(
    "reply",
    ["not json", "{}", '{"sufficient": false}', '{"sufficient": false, "next_query": "  "}'],
)
async def test_an_unusable_assessment_stops_gracefully(hop_kb, reply):
    llm = ScriptedLLM(lambda m, s: reply)
    found = await AgenticRetriever(SparseRetriever(hop_kb.store), llm).retrieve(QUESTION, k=2)
    assert sources(found) == {"people.md"} and len(llm.calls) == 1


async def test_it_wraps_any_retriever_and_passes_filters_and_context_through(hop_kb):
    seen = []

    class Spy:
        async def retrieve(self, query, *, k, filter=None, context=None):
            seen.append((query, filter, context))
            return await SparseRetriever(hop_kb.store).retrieve(
                query, k=k, filter=filter, context=context
            )

    ctx = RequestContext(user_id="u")
    llm = verdicts({"sufficient": False, "next_query": "Zurich site opens"})
    await AgenticRetriever(Spy(), llm).retrieve(
        QUESTION, k=1, filter={"source": "sites.md"}, context=ctx
    )
    assert [q for q, _, _ in seen] == [QUESTION, "Zurich site opens"]
    assert all(f == {"source": "sites.md"} and c is ctx for _, f, c in seen)


async def test_configured_end_to_end_through_the_agent(make_config, tmp_path):
    docs = tmp_path / "e2e-docs"
    docs.mkdir()
    (docs / "people.md").write_text(PEOPLE, encoding="utf-8")
    (docs / "sites.md").write_text(SITES, encoding="utf-8")
    config = make_config(
        knowledge={"sources": [{"location": str(docs)}], "persist_dir": str(tmp_path / "e2e-idx")},
        retrieval={"mode": "sparse", "top_k": 5, "final_k": 5, "agentic": {"max_steps": 3}},
    )
    kb = KnowledgeBase(config)
    await kb.aingest()

    def respond(messages, system):
        if system and "decide whether a document search" in system:
            found = messages[-1].content
            if "opens at 7am" in found:
                return json.dumps({"sufficient": True, "next_query": ""})
            return json.dumps({"sufficient": False, "next_query": "Zurich site opens closes"})
        return grounded_reply(messages, system)

    llm = ScriptedLLM(respond)
    answer = await Agent(kb, llm=llm).aask(QUESTION)
    assert {r.chunk.metadata["source"] for r in answer.retrieved} == {"people.md", "sites.md"}
    assert answer.usage.llm_calls == 3  # two assessments + the answer
    assert next(e for e in answer.trace if e.name == "retrieve").data["agentic"] is True


async def test_the_call_budget_also_guards_the_loop(make_config, tmp_path):
    docs = tmp_path / "b-docs"
    docs.mkdir()
    (docs / "people.md").write_text(PEOPLE, encoding="utf-8")
    config = make_config(
        knowledge={"sources": [{"location": str(docs)}], "persist_dir": str(tmp_path / "b-idx")},
        retrieval={"mode": "sparse", "top_k": 5, "agentic": {"max_steps": 8}},
        budget={"max_llm_calls_per_question": 2},
    )
    kb = KnowledgeBase(config)
    await kb.aingest()
    counter = iter(
        range(100)
    )  # every assessment asks for a brand-new query, so it never stops itself
    never_enough = ScriptedLLM(
        lambda m, s: json.dumps({"sufficient": False, "next_query": f"novel {next(counter)}"})
    )
    with pytest.raises(BudgetExceededError, match="already made 2 model calls"):
        await Agent(kb, llm=never_enough).aask(QUESTION)


async def test_the_langgraph_extra_gives_an_install_hint(monkeypatch, hop_kb):
    import importlib

    from kbsdk import MissingExtraError

    real = importlib.import_module

    def blocked(name, package=None):
        if name == "langgraph.graph":
            raise ImportError("no langgraph", name="langgraph")
        return real(name, package)

    monkeypatch.setattr("kbsdk.adapters._deps.importlib.import_module", blocked)
    agentic = AgenticRetriever(SparseRetriever(hop_kb.store), verdicts())
    with pytest.raises(MissingExtraError, match=r"kbsdk\[agentic\]"):
        await agentic.retrieve(QUESTION, k=1)
