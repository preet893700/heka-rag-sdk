"""Qdrant-specific behaviour, and the differential test that keeps its filters identical to ours.

Access control is built on the SDK's filter dialect, so a store that interprets a filter differently
from the reference implementation (`heka.rag.filters.matches`) is a data-leak risk. This test runs every
filter through both and demands the same answer.
"""

import itertools

import pytest

pytest.importorskip("qdrant_client")

from heka.rag import ConfigError, KnowledgeBase, RequestContext  # noqa: E402
from heka.rag.access import AccessPolicy  # noqa: E402
from heka.rag.adapters.qdrant_store import (  # noqa: E402
    QdrantSettings,
    QdrantStore,
    point_id,
    to_qdrant_filter,
)
from heka.rag.adapters.stores import IndexMismatchError  # noqa: E402
from heka.rag.config import AccessConfig  # noqa: E402
from heka.rag.filters import matches  # noqa: E402
from heka.rag.types import Chunk  # noqa: E402

METADATA = {
    "a": {"tenant_id": "acme", "allowed_roles": ["hr"], "year": 2024, "grade": "L3", "dept": "hr"},
    "b": {"tenant_id": "acme", "allowed_roles": ["*"], "year": 2025, "grade": "L4"},
    "c": {"tenant_id": "globex", "allowed_roles": ["hr", "finance"], "year": 2025, "grade": "L3"},
    "d": {"tenant_id": "*", "allowed_roles": ["*"], "year": 2023, "dept": "legal"},
    "e": {"allowed_roles": ["hr"], "year": 2025},  # no tenant tag
    "f": {"tenant_id": "acme", "allowed_roles": [], "year": 2026, "grade": "L5"},  # empty roles
    "g": {"tenant_id": "acme"},  # no roles at all
    "h": {"tenant_id": "acme", "allowed_roles": ["finance"], "active": True, "dept": "hr"},
}

FILTERS = [
    {"tenant_id": "acme"},
    {"tenant_id": "nobody"},
    {"allowed_roles": "hr"},  # equality on a list-valued field = membership
    {"grade": {"$eq": "L3"}},
    {"grade": {"$ne": "L3"}},  # matches chunks that lack the field too
    {"grade": {"$in": ["L3", "L4"]}},
    {"grade": {"$nin": ["L3", "L4"]}},
    {"allowed_roles": {"$in": ["hr", "*"]}},  # overlap with a list
    {"allowed_roles": {"$nin": ["hr"]}},
    {"tenant_id": {"$in": ["acme", "*"]}, "allowed_roles": {"$in": ["finance", "*"]}},
    {"year": {"$gte": 2025}},
    {"year": {"$gt": 2024, "$lt": 2026}},
    {"year": {"$lte": 2024}},
    {"$and": [{"tenant_id": "acme"}, {"year": {"$gte": 2025}}]},
    {"$or": [{"tenant_id": "globex"}, {"dept": "legal"}]},
    {"$or": [{"grade": "L5"}, {"$and": [{"tenant_id": "acme"}, {"allowed_roles": "finance"}]}]},
    {"active": True},
    {"dept": "hr", "tenant_id": {"$ne": "globex"}},
    {"missing_field": "x"},
    {"missing_field": {"$ne": "x"}},
]


async def build_pair():
    from heka.rag.adapters.stores import LocalVectorStore

    local = LocalVectorStore()
    qdrant = QdrantStore(QdrantSettings(location=":memory:", collection="diff"))
    chunks = [Chunk(id=cid, doc_id="d", text=cid, metadata=meta) for cid, meta in METADATA.items()]
    vectors = [[1.0, 0.1 * index, 0.0] for index in range(len(chunks))]
    await local.upsert(chunks, vectors)
    await qdrant.upsert(chunks, vectors)
    return local, qdrant


@pytest.mark.parametrize("flt", FILTERS, ids=[str(f)[:60] for f in FILTERS])
async def test_qdrant_and_the_reference_filter_agree(flt):
    local, qdrant = await build_pair()
    expected = {cid for cid, meta in METADATA.items() if matches(flt, meta)}
    from_local = {r.chunk.id for r in await local.search([1.0, 0.0, 0.0], k=20, filter=flt)}
    from_qdrant = {r.chunk.id for r in await qdrant.search([1.0, 0.0, 0.0], k=20, filter=flt)}
    assert from_local == expected, f"local store disagrees with the reference for {flt}"
    assert from_qdrant == expected, f"qdrant disagrees with the reference for {flt}"


async def test_every_access_policy_shape_matches_across_stores():
    """The exact filters the access policy generates, for a spread of callers."""
    local, qdrant = await build_pair()
    policy = AccessPolicy(AccessConfig(tenant_field="tenant_id", roles_field="allowed_roles"))
    tenants, roles = ["acme", "globex", None], [[], ["hr"], ["finance"], ["hr", "finance"]]
    for tenant, held in itertools.product(tenants, roles):
        flt = policy.filter_for(RequestContext(tenant_id=tenant, roles=held))
        expected = {cid for cid, meta in METADATA.items() if matches(flt, meta)}
        got = {r.chunk.id for r in await qdrant.search([1.0, 0.0, 0.0], k=20, filter=flt)}
        assert got == expected, (tenant, held)


def test_unsupported_filter_shapes_fail_loudly_not_silently():
    from qdrant_client import models

    with pytest.raises(ConfigError, match="numbers or ISO dates"):
        to_qdrant_filter(models, {"name": {"$gt": "abc"}})
    with pytest.raises(ConfigError, match="non-empty list"):
        to_qdrant_filter(models, {"x": {"$in": []}})
    with pytest.raises(ConfigError, match="strings, integers and booleans"):
        to_qdrant_filter(models, {"x": 1.5})
    with pytest.raises(ConfigError, match="Unknown filter operator"):
        to_qdrant_filter(models, {"x": {"$regex": "a"}})
    assert to_qdrant_filter(models, None) is None and to_qdrant_filter(models, {}) is None


async def test_iso_date_ranges():
    _, qdrant = await build_pair()
    chunk = Chunk(id="dated", doc_id="d", text="t", metadata={"effective": "2025-06-01T00:00:00Z"})
    await qdrant.upsert([chunk], [[1.0, 0.0, 0.0]])
    hits = await qdrant.search(
        [1.0, 0.0, 0.0], k=20, filter={"effective": {"$gte": "2025-01-01T00:00:00Z"}}
    )
    assert {r.chunk.id for r in hits} == {"dated"}


async def test_chunks_round_trip_exactly_including_nested_metadata():
    store = QdrantStore(QdrantSettings(collection="rt"))
    original = Chunk(
        id="x:0001",
        doc_id="doc",
        text="Employees get 12 days.\n\n| a | b |",
        index=3,
        parent_id="p",
        metadata={"source": "a.md", "nested": {"k": [1, 2]}, "roles": ["hr"], "big": "y" * 2000},
    )
    await store.upsert([original], [[0.0, 1.0]])
    assert (await store.get(["x:0001"]))[0] == original
    assert (await store.search([0.0, 1.0], k=1))[0].chunk == original
    assert (await store.all_chunks()) == [original]
    assert (await store.search([0.0, 1.0], k=1, filter={"roles": "hr"}))[0].chunk.id == "x:0001"
    assert (
        await store.search([0.0, 1.0], k=1, filter={"big": "y" * 2000}) == []
    )  # long text isn't a field


async def test_embedded_store_persists_on_disk(tmp_path):
    settings = QdrantSettings(location=str(tmp_path / "qd"), collection="persist")
    first = QdrantStore(settings)
    await first.upsert([Chunk(id="a", doc_id="d", text="hello")], [[1.0, 0.0]])
    await first._client().close()
    second = QdrantStore(settings)
    assert await second.count() == 1 and (await second.get(["a"]))[0].text == "hello"


async def test_collections_isolate_agents_sharing_one_backend():
    one = QdrantStore(QdrantSettings(collection="agent-one"))
    await one.upsert([Chunk(id="a", doc_id="d", text="one")], [[1.0, 0.0]])
    assert await QdrantStore(QdrantSettings(collection="agent-two")).count() == 0


async def test_dimension_change_is_detected():
    store = QdrantStore(QdrantSettings(collection="dims"))
    await store.upsert([Chunk(id="a", doc_id="d", text="t")], [[1.0, 0.0]])
    with pytest.raises(IndexMismatchError, match="Embedding size changed"):
        await store.upsert([Chunk(id="b", doc_id="d", text="t")], [[1.0, 0.0, 0.0]])


def test_settings_reject_location_and_url_together():
    with pytest.raises(ValueError, match="either"):
        QdrantSettings(location=":memory:", url="http://localhost:6333")


def test_point_ids_are_stable_uuids():
    assert point_id("x") == point_id("x") != point_id("y") and len(point_id("x")) == 36


async def test_kb_end_to_end_with_qdrant_including_hybrid_and_access_control(make_config, docs_dir):
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
    ]
    config = make_config(
        store={"provider": "qdrant", "params": {"location": ":memory:"}},
        knowledge=knowledge,
        access={"tenant_field": "tenant_id", "roles_field": "allowed_roles"},
        retrieval={"mode": "hybrid", "final_k": 10, "top_k": 20},
    )
    kb = KnowledgeBase(config)
    report = await kb.aingest()
    assert not report.failed and await kb.acount() > 0
    employee = RequestContext(tenant_id="acme", roles=["employee"])
    finance = RequestContext(tenant_id="acme", roles=["finance"])
    seen = {
        c.chunk.metadata["source"]
        for c in await kb.aretrieve("leave expense claims", k=10, context=employee)
    }
    assert seen == {"leave.md"}
    seen = {
        c.chunk.metadata["source"]
        for c in await kb.aretrieve("leave expense claims", k=10, context=finance)
    }
    assert seen == {"leave.md", "expenses.md"}
    again = await kb.aingest()  # incremental works the same on this store
    assert again.files_unchanged == 2 and again.chunks_added == 0


def test_factory_defaults_give_each_agent_its_own_collection_and_folder(make_config, tmp_path):
    kb = KnowledgeBase(make_config(name="HR Assistant", store={"provider": "qdrant"}))
    assert kb.store.collection == "hr-assistant"
    assert kb.store.settings.location.endswith("qdrant")
