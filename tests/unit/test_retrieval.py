import json

from conftest import ScriptedLLM
from kbsdk import Agent, KnowledgeBase, Message, RAGConfig, Usage
from kbsdk.adapters.retrievers import (
    DenseRetriever,
    HybridRetriever,
    SparseRetriever,
    fuse_rrf,
    fuse_weighted,
)
from kbsdk.adapters.sparse import Bm25Index, tokenize
from kbsdk.adapters.transforms import (
    Condenser,
    HydeSettings,
    HydeTransform,
    MultiQuerySettings,
    MultiQueryTransform,
    RewriteSettings,
    RewriteTransform,
)
from kbsdk.factory import LazyLLM, build_retrieval
from kbsdk.pipelines.retrieve import RetrievalPipeline, expand_parents
from kbsdk.types import Chunk, ScoredChunk
from kbsdk.usage import MeteredLLM, record_usage, track_usage


def chunk(cid, text, **metadata):
    return Chunk(id=cid, doc_id="d", text=text, metadata={"source": f"{cid}.md", **metadata})


def scored(cid, score=1.0, retriever="dense", **signals):
    return ScoredChunk(
        chunk=chunk(cid, f"text {cid}"), score=score, retriever=retriever, signals=signals
    )


# -- tokenizer and BM25 -------------------------------------------------------------------------


def test_tokenizer_folds_plurals_and_keeps_identifiers():
    assert tokenize("The policies for Casual Days") == ["policy", "casual", "day"]
    tokens = tokenize("Use Form HR-207 for leave; see 2.5 percent")
    assert "hr-207" in tokens and "hr" in tokens and "207" in tokens  # whole and split
    assert "2.5" in tokens
    assert "the" not in tokens and "for" not in tokens  # stopwords dropped
    assert tokenize("doesn't") == ["doesnt"]
    assert tokenize("class business") == ["class", "business"]  # ss/us words are not mangled


def build_index(*pairs):
    index = Bm25Index()
    index.build(
        [
            chunk(cid, text, **meta)
            for cid, text, *rest in pairs
            for meta in [rest[0] if rest else {}]
        ]
    )
    return index


def test_bm25_ranks_exact_terms_and_rare_terms_higher():
    index = build_index(
        ("a", "Form HR-207 requests parental leave extension."),
        ("b", "Leave policy for all employees. Leave leave leave."),
        ("c", "Expense claims are reimbursed monthly."),
    )
    top = index.search("what is form HR-207", k=3)
    assert [r.chunk.id for r in top][0] == "a"
    assert top[0].signals["sparse"] == top[0].score > 0
    assert index.search("zzz nonexistent", k=3) == []
    assert [r.chunk.id for r in index.search("expense claims", k=3)] == ["c"]


def test_bm25_uses_the_heading_context_too():
    index = Bm25Index()
    index.build(
        [
            Chunk(
                id="g",
                doc_id="d",
                text="Employees get 30 days.",
                metadata={"context": "Germany Supplement > Annual leave"},
            )
        ]
    )
    assert index.search("germany", k=1)[0].chunk.id == "g"


def test_bm25_filters_and_limits():
    index = build_index(
        ("a", "annual leave days", {"region": "EU"}),
        ("b", "annual leave days", {"region": "US"}),
        ("c", "annual leave days", {"region": "EU"}),
    )
    assert {r.chunk.id for r in index.search("annual leave", k=5, filter={"region": "EU"})} == {
        "a",
        "c",
    }
    assert len(index.search("annual leave", k=1)) == 1
    assert Bm25Index().search("anything", k=3) == []


# -- fusion -------------------------------------------------------------------------------------


def test_rrf_rewards_agreement_and_merges_signals():
    dense = [scored("a", 0.9, dense=0.9), scored("b", 0.8, dense=0.8), scored("c", 0.7, dense=0.7)]
    sparse = [scored("b", 9.0, "sparse", sparse=9.0), scored("d", 5.0, "sparse", sparse=5.0)]
    fused = fuse_rrf([dense, sparse], k=4)
    assert [r.chunk.id for r in fused][0] == "b"  # found by both lists
    assert {r.chunk.id for r in fused} == {"a", "b", "c", "d"}
    top = next(r for r in fused if r.chunk.id == "b")
    assert top.signals == {"dense": 0.8, "sparse": 9.0} and top.retriever == "hybrid"
    assert len(fuse_rrf([dense, sparse], k=2)) == 2


def test_rrf_weights_shift_the_balance():
    dense, sparse = [scored("a"), scored("b")], [scored("b"), scored("a")]
    assert fuse_rrf([dense, sparse], k=2, weights=[2, 0.5])[0].chunk.id == "a"
    assert fuse_rrf([dense, sparse], k=2, weights=[0.5, 2])[0].chunk.id == "b"


def test_weighted_fusion_normalises_scores():
    dense = [scored("a", 0.9, dense=0.9), scored("b", 0.1, dense=0.1)]
    sparse = [scored("b", 100.0, "sparse", sparse=100.0), scored("c", 1.0, "sparse", sparse=1.0)]
    assert fuse_weighted(dense, sparse, k=1, dense_weight=0.0)[0].chunk.id == "b"
    assert fuse_weighted(dense, sparse, k=1, dense_weight=1.0)[0].chunk.id == "a"
    assert fuse_weighted([], [], k=3, dense_weight=0.5) == []


def test_relevance_prefers_rerank_then_dense_then_score():
    assert scored("a", 0.02, rerank=0.9, dense=0.5).relevance == 0.9
    assert scored("a", 0.02, dense=0.5).relevance == 0.5
    assert scored("a", 0.02).relevance == 0.02


# -- retrievers over a real store ---------------------------------------------------------------


async def test_dense_sparse_and_hybrid_retrieve(kb):
    dense = DenseRetriever(kb.store, kb.embedder)
    sparse = SparseRetriever(kb.store)
    hybrid = HybridRetriever(dense, sparse)
    for retriever, label in ((dense, "dense"), (sparse, "sparse"), (hybrid, "hybrid")):
        results = await retriever.retrieve("casual leave days", k=3)
        assert results and results[0].chunk.metadata["source"] == "leave.md"
        assert results[0].retriever == label
    assert "dense" in (await dense.retrieve("leave", k=1))[0].signals
    fused = (await hybrid.retrieve("casual leave", k=2))[0].signals
    assert "dense" in fused and "sparse" in fused


async def test_sparse_index_rebuilds_when_the_store_changes(make_config, docs_dir):
    kb = KnowledgeBase(make_config())
    await kb.aingest()
    sparse = SparseRetriever(kb.store)
    assert await sparse.retrieve("zebra", k=3) == []
    (docs_dir / "zoo.md").write_text(
        "# Zoo\n\nThe zebra enclosure opens at nine.\n", encoding="utf-8"
    )
    await kb.aingest()
    assert (await sparse.retrieve("zebra", k=3))[0].chunk.metadata["source"] == "zoo.md"
    (docs_dir / "zoo.md").unlink()
    await kb.aingest()
    assert await sparse.retrieve("zebra", k=3) == []


async def test_hybrid_finds_an_exact_code_that_dense_blurs(make_config, docs_dir):
    (docs_dir / "forms.md").write_text(
        "# Forms\n\n## Parental leave extension\n\nUse form HR-207 to request a parental leave extension.\n\n"
        "## Sabbatical\n\nUse form HR-311 to request a sabbatical.\n",
        encoding="utf-8",
    )
    kb = KnowledgeBase(make_config(retrieval={"mode": "hybrid"}))
    await kb.aingest()
    top = await kb.aretrieve("HR-311", k=1)
    assert "HR-311" in top[0].chunk.text


# -- transforms ---------------------------------------------------------------------------------


def llm_returning(text):
    return ScriptedLLM(lambda m, s: text)


async def test_rewrite_multi_query_and_hyde():
    rewrite = RewriteTransform(RewriteSettings(), llm_returning('"annual leave entitlement"'))
    assert await rewrite.transform("how much time off", []) == [
        "how much time off",
        "annual leave entitlement",
    ]

    multi = MultiQueryTransform(
        MultiQuerySettings(n=2),
        llm_returning(json.dumps({"queries": ["vacation days", "leave days", "extra"]})),
    )
    assert await multi.transform("time off", []) == ["time off", "vacation days", "leave days"]
    junk = MultiQueryTransform(MultiQuerySettings(), llm_returning("not json"))
    assert await junk.transform("time off", []) == ["time off"]  # falls back to the original

    hyde = HydeTransform(
        HydeSettings(), llm_returning("Employees receive 20 days of annual leave.")
    )
    assert (await hyde.transform("time off", []))[1].startswith("Employees receive 20 days")


async def test_transform_output_is_deduplicated():
    same = RewriteTransform(RewriteSettings(), llm_returning("How much time off"))
    assert await same.transform("how much time off", []) == ["how much time off"]


async def test_condenser_only_calls_the_model_with_history():
    llm = llm_returning('"How many casual leave days do contractors get?"')
    condenser = Condenser(llm)
    assert await condenser.condense("casual leave?", []) == "casual leave?"
    assert llm.calls == []
    history = [
        Message(role="user", content="Tell me about leave"),
        Message(role="assistant", content="Sure"),
    ]
    assert await condenser.condense("what about contractors?", history) == (
        "How many casual leave days do contractors get?"
    )
    assert "Tell me about leave" in llm.calls[0][0][0].content
    assert await Condenser(llm_returning("  ")).condense("q?", history) == "q?"


# -- retrieval pipeline -------------------------------------------------------------------------


class RecordingRetriever:
    """Returns canned results per query and remembers the queries and filters it saw."""

    def __init__(self, by_query):
        self.by_query = by_query
        self.calls = []

    async def retrieve(self, query, *, k, filter=None, context=None):
        self.calls.append((query, k, filter))
        return self.by_query.get(query, [])


def pipeline(retriever, config=None, **kwargs):
    config = config or RAGConfig.from_dict({"generation": {"llm": {"provider": "scripted"}}})
    return RetrievalPipeline(config, retriever, **kwargs)


async def test_multi_query_fans_out_and_fuses():
    retriever = RecordingRetriever(
        {"q": [scored("a"), scored("b")], "q2": [scored("c"), scored("b")]}
    )
    transform = MultiQueryTransform(
        MultiQuerySettings(), llm_returning(json.dumps({"queries": ["q2"]}))
    )
    result = await pipeline(retriever, transforms=[transform]).run("q")
    assert [q for q, _, _ in retriever.calls] == ["q", "q2"]
    assert result.queries == ["q", "q2"] and result.chunks[0].chunk.id == "b"  # in both lists
    assert result.embedding_calls == 2  # dense mode embeds each query once
    assert {e.stage for e in result.trace} >= {"transform", "retrieve", "select"}


async def test_condense_uses_history_and_searches_the_standalone_question():
    retriever = RecordingRetriever({"standalone question": [scored("a")]})
    condenser = Condenser(llm_returning("standalone question"))
    history = [Message(role="user", content="earlier")]
    result = await pipeline(retriever, condenser=condenser).run("and them?", history=history)
    assert retriever.calls[0][0] == "standalone question" and result.chunks
    no_history = await pipeline(retriever, condenser=condenser).run("and them?")
    assert retriever.calls[-1][0] == "and them?" and no_history.chunks == []


class ReverseReranker:
    async def rerank(self, query, chunks, *, top_n):
        return [
            c.model_copy(update={"signals": {**c.signals, "rerank": 0.99 - i / 10}})
            for i, c in enumerate(reversed(list(chunks)))
        ][:top_n]


async def test_reranker_reorders_and_is_traced():
    retriever = RecordingRetriever({"q": [scored("a"), scored("b"), scored("c")]})
    config = RAGConfig.from_dict(
        {"generation": {"llm": {"provider": "scripted"}}, "retrieval": {"final_k": 2}}
    )
    result = await pipeline(retriever, config, reranker=ReverseReranker()).run("q")
    assert [c.chunk.id for c in result.chunks] == ["c", "b"]
    rerank = next(e for e in result.trace if e.stage == "rerank")
    assert rerank.data["changed_top"] is True


async def test_final_k_override_and_filters_are_passed_through():
    retriever = RecordingRetriever({"q": [scored(f"c{i}") for i in range(8)]})
    config = RAGConfig.from_dict(
        {
            "generation": {"llm": {"provider": "scripted"}},
            "retrieval": {"top_k": 8, "final_k": 3, "filter": {"a": 1}},
        }
    )
    result = await pipeline(retriever, config).run("q", filter={"b": 2}, final_k=5)
    assert len(result.chunks) == 5
    assert retriever.calls[0][2] == {"$and": [{"a": 1}, {"b": 2}]}


def test_expand_parents_swaps_in_the_section_and_merges_siblings():
    def child(cid, parent, score):
        c = chunk(cid, f"child {cid}", parent_id=parent, parent_text=f"WHOLE {parent}")
        return ScoredChunk(chunk=c, score=score)

    candidates = [
        child("c1", "p1", 0.9),
        child("c2", "p1", 0.8),
        child("c3", "p2", 0.7),
        scored("plain"),
    ]
    expanded = expand_parents(candidates, limit=3)
    assert [r.chunk.text for r in expanded] == ["WHOLE p1", "WHOLE p2", "text plain"]
    assert "parent_text" not in expanded[0].chunk.metadata  # not carried into the answer
    assert expanded[0].score == 0.9 and expanded[0].chunk.id == "c1"


async def test_pipeline_counts_embedding_calls_per_query(kb):
    config = RAGConfig.from_dict(
        {"generation": {"llm": {"provider": "scripted"}}, "retrieval": {"mode": "hybrid"}}
    )
    hybrid = HybridRetriever(DenseRetriever(kb.store, kb.embedder), SparseRetriever(kb.store))
    transform = MultiQueryTransform(
        MultiQuerySettings(), llm_returning(json.dumps({"queries": ["sick leave"]}))
    )
    result = await RetrievalPipeline(config, hybrid, transforms=[transform]).run("casual leave")
    assert result.embedding_calls == 2


# -- usage metering and lazy LLMs ---------------------------------------------------------------


async def test_usage_is_summed_across_every_model_call_in_a_question(kb):
    config = kb.config.model_copy(
        update={"retrieval": kb.config.retrieval.model_copy(update={"condense_followups": True})}
    )
    no_answer = json.dumps({"answerable": False, "answer": "", "citations": []})

    def router(messages, system):  # answers questions; rewrites follow-ups when asked to
        return (
            no_answer if system and "ONLY the numbered" in system else "standalone leave question"
        )

    agent = Agent(kb, config, llm=ScriptedLLM(router))
    history = [Message(role="user", content="hi"), Message(role="assistant", content="hello")]
    answer = await agent.aask("and casual?", history=history)
    assert answer.usage.llm_calls == 2  # one to condense the follow-up, one to answer
    assert answer.usage.input_tokens == 20


async def test_track_usage_scopes_do_not_leak():
    record_usage(Usage(llm_calls=5))  # no active meter: ignored
    with track_usage() as outer:
        record_usage(Usage(llm_calls=1))
        with track_usage() as inner:
            record_usage(Usage(llm_calls=2))
        record_usage(Usage(llm_calls=1))
    assert (outer.total.llm_calls, inner.total.llm_calls) == (2, 2)


async def test_metered_llm_records_and_passes_through():
    inner = ScriptedLLM(lambda m, s: "hi", name="x")
    metered = MeteredLLM(inner)
    with track_usage() as meter:
        response = await metered.generate([Message(role="user", content="q")])
        pieces = [p async for p in metered.stream([Message(role="user", content="q")])]
    assert response.text == "hi" and pieces == ["hi"] and metered.name == "x"
    assert meter.total.llm_calls == 1


async def test_lazy_llm_is_only_built_when_used():
    built = []

    def provider():
        built.append(1)
        return ScriptedLLM(lambda m, s: "ok", name="real")

    lazy = LazyLLM(provider)
    assert lazy.name == "lazy" and built == []
    assert (await lazy.generate([Message(role="user", content="q")])).text == "ok"
    assert [p async for p in lazy.stream([Message(role="user", content="q")])] == ["ok"]
    assert lazy.name == "real" and built == [1]


async def test_retrieval_without_llm_features_never_builds_the_llm(kb):
    def explode():
        raise AssertionError("the LLM must not be built for plain retrieval")

    pipe = build_retrieval(kb.config, kb.retriever, explode)
    assert (await pipe.run("casual leave")).chunks


def test_registry_passes_only_the_dependencies_an_adapter_declares():
    from kbsdk import registry

    marker = object()
    transform = registry.create(
        "query_transform", "rewrite", {}, deps={"llm": marker, "embedder": marker}
    )
    assert transform.llm is marker  # 'embedder' was not accepted, and did not break construction
    reranker = registry.create("reranker", "llm", {"max_chars": 100}, deps={"llm": marker})
    assert reranker.settings.max_chars == 100
