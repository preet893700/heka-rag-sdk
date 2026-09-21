"""Contract tests for the local vector store: what any `VectorStore` adapter must satisfy."""

import numpy as np
import pytest

from heka.rag.adapters.stores import IndexMismatchError, LocalStoreSettings, LocalVectorStore
from heka.rag.interfaces import VectorStore
from heka.rag.types import Chunk


def chunk(cid, **metadata):
    return Chunk(id=cid, doc_id="d", text=f"text {cid}", metadata=metadata)


def vec(*values):
    return list(values)


@pytest.fixture(params=["local", "qdrant"])
def store(request):
    """Every contract test below runs against each store implementation."""
    if request.param == "local":
        return LocalVectorStore()
    pytest.importorskip("qdrant_client")
    from heka.rag.adapters.qdrant_store import QdrantSettings, QdrantStore

    return QdrantStore(QdrantSettings(location=":memory:", collection="contract"))


async def seeded(store):
    await store.upsert(
        [chunk("a", region="EU", roles=["hr"]), chunk("b", region="US"), chunk("c", region="EU")],
        [vec(1, 0, 0), vec(0, 1, 0), vec(0.9, 0.1, 0)],
    )
    return store


def test_conforms_to_the_protocol(store):
    assert isinstance(store, VectorStore)


async def test_search_ranks_by_cosine_similarity(store):
    await seeded(store)
    results = await store.search(vec(1, 0, 0), k=3)
    assert [r.chunk.id for r in results] == ["a", "c", "b"]
    assert results[0].score == pytest.approx(1.0)
    assert results[0].score > results[1].score > results[2].score


async def test_vectors_are_normalised_so_scale_does_not_matter(store):
    await store.upsert([chunk("big"), chunk("small")], [vec(100, 0), vec(0.001, 0)])
    scores = {r.chunk.id: r.score for r in await store.search(vec(5, 0), k=2)}
    assert scores["big"] == pytest.approx(1.0)
    assert scores["small"] == pytest.approx(1.0)


async def test_k_larger_than_size_and_zero(store):
    await seeded(store)
    assert len(await store.search(vec(1, 0, 0), k=99)) == 3
    assert await store.search(vec(1, 0, 0), k=0) == []
    assert await LocalVectorStore().search(vec(1), k=3) == []


async def test_metadata_filters(store):
    await seeded(store)
    eu = await store.search(vec(1, 0, 0), k=5, filter={"region": "EU"})
    assert {r.chunk.id for r in eu} == {"a", "c"}
    hr = await store.search(vec(1, 0, 0), k=5, filter={"roles": "hr"})
    assert [r.chunk.id for r in hr] == ["a"]
    none = await store.search(vec(1, 0, 0), k=5, filter={"region": "APAC"})
    assert none == []


async def test_upsert_replaces_existing_ids(store):
    await seeded(store)
    await store.upsert([chunk("a", region="APAC")], [vec(0, 0, 1)])
    assert await store.count() == 3
    assert (await store.get(["a"]))[0].metadata["region"] == "APAC"
    top = (await store.search(vec(0, 0, 1), k=1))[0]
    assert top.chunk.id == "a"


async def test_delete_by_ids_filter_and_guardrail(store):
    await seeded(store)
    assert await store.delete(ids=["b"]) == 1
    assert await store.delete(filter={"region": "EU"}) == 2
    assert await store.count() == 0
    with pytest.raises(ValueError):
        await store.delete()


async def test_delete_ids_and_filter_intersect(store):
    await seeded(store)
    assert await store.delete(ids=["a", "b"], filter={"region": "EU"}) == 1
    assert {c.id for c in await store.get(["a", "b", "c"])} == {"b", "c"}


async def test_clear(store):
    await seeded(store)
    await store.clear()
    assert await store.count() == 0
    assert await store.search(vec(1, 0, 0), k=3) == []


async def test_get_ignores_unknown_ids(store):
    await seeded(store)
    assert [c.id for c in await store.get(["c", "nope"])] == ["c"]


async def test_dimension_mismatch_is_a_clear_error(store):
    await seeded(store)
    with pytest.raises(IndexMismatchError, match="embedder"):
        await store.upsert([chunk("x")], [vec(1, 0)])
    with pytest.raises(IndexMismatchError, match="embedder"):
        await store.search(vec(1, 0), k=1)


async def test_mismatched_lengths_rejected(store):
    with pytest.raises(ValueError):
        await store.upsert([chunk("a")], [])


async def test_persistence_round_trip(tmp_path):
    settings = LocalStoreSettings(path=str(tmp_path / "index"))
    first = LocalVectorStore(settings)
    await seeded(first)
    second = LocalVectorStore(settings)  # a fresh process reading the same folder
    assert await second.count() == 3
    results = await second.search(vec(1, 0, 0), k=1)
    assert results[0].chunk.id == "a"
    assert results[0].chunk.metadata["roles"] == ["hr"]
    await second.delete(ids=["a"])
    assert await LocalVectorStore(settings).count() == 2


async def test_clear_persists(tmp_path):
    settings = LocalStoreSettings(path=str(tmp_path / "index"))
    store = LocalVectorStore(settings)
    await seeded(store)
    await store.clear()
    assert await LocalVectorStore(settings).count() == 0


async def test_corrupt_index_is_detected(tmp_path):
    settings = LocalStoreSettings(path=str(tmp_path / "index"))
    await seeded(LocalVectorStore(settings))
    (tmp_path / "index" / "chunks.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(IndexMismatchError, match="inconsistent"):
        LocalVectorStore(settings)


async def test_keyword_search_not_native(store):
    assert store.supports_keyword is False
    with pytest.raises(NotImplementedError):
        await store.keyword_search("x", k=1)


async def test_scales_to_many_vectors(store):
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(5000, 64)).tolist()
    await store.upsert([chunk(f"c{i}") for i in range(5000)], vectors)
    results = await store.search(vectors[123], k=3)
    assert results[0].chunk.id == "c123"
