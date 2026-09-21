import pytest

from conftest import ScriptedLLM, grounded_reply
from heka.rag import AccessDeniedError, Agent, KnowledgeBase, RequestContext
from heka.rag.access import AccessPolicy
from heka.rag.config import AccessConfig
from heka.rag.filters import matches

CFG = AccessConfig(tenant_field="tenant_id", roles_field="allowed_roles")


def meta(tenant="acme", roles=("hr",)):
    return {"tenant_id": tenant, "allowed_roles": list(roles)}


# -- the policy itself --------------------------------------------------------------------------


def test_disabled_policy_adds_no_filter():
    policy = AccessPolicy(AccessConfig())
    assert not policy.enabled and policy.filter_for(None) is None


def test_filter_reflects_tenant_roles_and_shared_content():
    flt = AccessPolicy(CFG).filter_for(RequestContext(tenant_id="acme", roles=["hr", "manager"]))
    assert matches(flt, meta("acme", ["hr"]))
    assert matches(flt, meta("acme", ["manager", "finance"]))  # any overlapping role
    assert matches(flt, meta("acme", ["*"]))  # public within the tenant
    assert matches(flt, meta("*", ["hr"]))  # shared across tenants
    assert not matches(flt, meta("globex", ["hr"]))  # another tenant
    assert not matches(flt, meta("acme", ["finance"]))  # role not held


def test_untagged_chunks_are_invisible_to_everyone():
    flt = AccessPolicy(CFG).filter_for(RequestContext(tenant_id="acme", roles=["hr"]))
    assert not matches(flt, {})
    assert not matches(flt, {"tenant_id": "acme"})  # missing roles
    assert not matches(flt, {"allowed_roles": ["hr"]})  # missing tenant
    assert not matches(flt, {"tenant_id": "acme", "allowed_roles": []})


def test_a_user_with_no_roles_sees_only_public_content():
    flt = AccessPolicy(CFG).filter_for(RequestContext(tenant_id="acme"))
    assert matches(flt, meta("acme", ["*"])) and not matches(flt, meta("acme", ["hr"]))


def test_attribute_fields():
    policy = AccessPolicy(AccessConfig(attribute_fields={"region": "region"}))
    flt = policy.filter_for(RequestContext(attributes={"region": "EU"}))
    assert matches(flt, {"region": "EU"}) and matches(flt, {"region": "*"})
    assert not matches(flt, {"region": "US"}) and not matches(flt, {})
    none = policy.filter_for(RequestContext())
    assert matches(none, {"region": "*"}) and not matches(none, {"region": "EU"})


def test_fails_closed_without_a_context():
    with pytest.raises(AccessDeniedError, match="RequestContext"):
        AccessPolicy(CFG).filter_for(None)
    lenient = AccessPolicy(AccessConfig(roles_field="r", require_context=False))
    flt = lenient.filter_for(None)  # only shared content
    assert matches(flt, {"r": ["*"]}) and not matches(flt, {"r": ["hr"]})


def test_custom_shared_marker():
    policy = AccessPolicy(AccessConfig(roles_field="r", shared_value="public"))
    flt = policy.filter_for(RequestContext())
    assert matches(flt, {"r": ["public"]}) and not matches(flt, {"r": ["*"]})


def test_missing_fields_reports_untagged_content():
    policy = AccessPolicy(CFG)
    assert policy.missing_fields({}) == ["tenant_id", "allowed_roles"]
    assert policy.missing_fields(meta()) == []
    assert AccessPolicy(AccessConfig()).missing_fields({}) == []


# -- enforced end to end ------------------------------------------------------------------------


@pytest.fixture
def secured(make_config, docs_dir):
    """leave.md is public within acme; expenses.md is finance-only; a third file belongs to globex."""
    (docs_dir / "globex.md").write_text(
        "# Globex\n\n## Leave\n\nGlobex staff get 99 days of leave.\n", encoding="utf-8"
    )
    knowledge = make_config().to_dict()["knowledge"]
    knowledge["sources"] = [
        {
            "location": str(docs_dir / "leave.md"),
            "metadata": {"tenant_id": "acme", "allowed_roles": ["*"]},
        },
        {
            "location": str(docs_dir / "expenses.md"),
            "metadata": {"tenant_id": "acme", "allowed_roles": ["finance"]},
        },
        {
            "location": str(docs_dir / "globex.md"),
            "metadata": {"tenant_id": "globex", "allowed_roles": ["*"]},
        },
    ]
    return make_config(
        knowledge=knowledge,
        access={"tenant_field": "tenant_id", "roles_field": "allowed_roles"},
        retrieval={"mode": "hybrid", "final_k": 10, "top_k": 20},
    )


async def sources_for(kb, question, context, **kwargs):
    chunks = await kb.aretrieve(question, k=10, context=context, **kwargs)
    return {c.chunk.metadata["source"] for c in chunks}


async def test_retrieval_only_returns_what_the_caller_may_see(secured):
    kb = KnowledgeBase(secured)
    await kb.aingest()
    employee = RequestContext(tenant_id="acme", roles=["employee"])
    finance = RequestContext(tenant_id="acme", roles=["finance"])
    other = RequestContext(tenant_id="globex", roles=["employee"])
    question = "leave expense claims days"
    assert await sources_for(kb, question, employee) == {"leave.md"}
    assert await sources_for(kb, question, finance) == {"leave.md", "expenses.md"}
    assert await sources_for(kb, question, other) == {"globex.md"}


async def test_a_caller_supplied_filter_can_only_narrow_never_widen(secured):
    kb = KnowledgeBase(secured)
    await kb.aingest()
    employee = RequestContext(tenant_id="acme", roles=["employee"])
    # Asking for finance-only content, or another tenant's, through the caller filter finds nothing
    assert (
        await sources_for(kb, "expense claims", employee, filter={"source": "expenses.md"}) == set()
    )
    assert await sources_for(kb, "leave", employee, filter={"tenant_id": "globex"}) == set()
    widened = {"$or": [{"tenant_id": "globex"}, {"source": "expenses.md"}]}
    assert await sources_for(kb, "leave expense", employee, filter=widened) == set()


async def test_every_retrieval_mode_is_filtered(secured):
    for mode in ("dense", "sparse", "hybrid"):
        retrieval = secured.retrieval.model_copy(update={"mode": mode})
        kb = KnowledgeBase(secured.model_copy(update={"retrieval": retrieval}))
        await kb.aingest()
        context = RequestContext(tenant_id="acme", roles=["employee"])
        assert await sources_for(kb, "expense claims leave days", context) <= {"leave.md"}, mode


async def test_asking_without_identity_is_refused_before_anything_is_searched(secured):
    kb = KnowledgeBase(secured)
    await kb.aingest()
    with pytest.raises(AccessDeniedError):
        await kb.aretrieve("leave")
    llm = ScriptedLLM(grounded_reply)
    with pytest.raises(AccessDeniedError):
        await Agent(kb, llm=llm).aask("How many leave days?")
    assert llm.calls == []


async def test_the_model_never_sees_content_the_user_may_not_read(secured):
    kb = KnowledgeBase(secured)
    await kb.aingest()
    llm = ScriptedLLM(grounded_reply)
    answer = await Agent(kb, llm=llm).aask(
        "What is the expense claim deadline?",
        context=RequestContext(tenant_id="acme", roles=["employee"]),
    )
    sources_block = llm.calls[0][0][-1].content.split("Question:")[0]
    assert "30 days" not in sources_block and "expense claims" not in sources_block.lower()
    assert all(r.chunk.metadata["source"] == "leave.md" for r in answer.retrieved)


async def test_ingestion_warns_about_untagged_content(make_config):
    config = make_config(access={"tenant_field": "tenant_id", "roles_field": "allowed_roles"})
    kb = KnowledgeBase(config)
    report = await kb.aingest()
    assert any("leave.md" in w and "nobody can see it" in w and "'*'" in w for w in report.warnings)
    context = RequestContext(tenant_id="acme", roles=["hr"])
    assert await kb.aretrieve("leave", context=context) == []


async def test_eval_cases_carry_their_own_identity(secured):
    from heka.rag.eval import EvalCase, EvalDataset, SourceRef, run_retrieval_eval

    kb = KnowledgeBase(secured)
    await kb.aingest()
    gold = [SourceRef(source="expenses.md", quote="within 30 days")]
    dataset = EvalDataset(
        cases=[
            EvalCase(
                id="ok",
                question="expense claim deadline",
                gold_answer="30",
                gold_sources=gold,
                context=RequestContext(tenant_id="acme", roles=["finance"]),
            ),
            EvalCase(
                id="denied",
                question="expense claim deadline",
                gold_answer="30",
                gold_sources=gold,
                context=RequestContext(tenant_id="acme", roles=["employee"]),
            ),
        ]
    )
    report = await run_retrieval_eval(kb, dataset)
    by_id = {r.case_id: r.metrics["retrieval_hit"] for r in report.results}
    assert by_id == {"ok": 1.0, "denied": 0.0}  # same question, different answers per identity
