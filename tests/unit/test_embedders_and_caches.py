import numpy as np
import pytest
from pydantic import ValidationError

from heka.rag.adapters.caches import DiskCache, DiskCacheSettings, MemoryCache
from heka.rag.adapters.embedders import HashingEmbedder, HashingSettings, ResilientEmbedder
from heka.rag.interfaces import Cache, Embedder


class Flaky:
    """An embedder that fails with a 429 a few times, and counts what it embeds."""

    dimensions = 4
    model_id = "flaky"

    def __init__(self, failures=0):
        self.failures = failures
        self.doc_calls = []
        self.query_calls = []

    async def embed_documents(self, texts):
        if self.failures:
            self.failures -= 1
            error = RuntimeError("quota")
            error.status_code = 429
            raise error
        self.doc_calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]

    async def embed_query(self, text):
        self.query_calls.append(text)
        return [float(len(text)), 1.0, 0.0, 0.0]


async def test_hashing_embedder_is_deterministic_and_lexical():
    embedder = HashingEmbedder()
    assert isinstance(embedder, Embedder)
    a, b, c = await embedder.embed_documents(
        ["casual leave days", "casual leave policy", "quarterly revenue report"]
    )
    assert a == (await embedder.embed_documents(["casual leave days"]))[0]
    cosine = lambda x, y: float(np.dot(x, y))  # noqa: E731
    assert cosine(a, b) > cosine(a, c)
    assert len(a) == embedder.dimensions == 384
    assert cosine(a, a) == pytest.approx(1.0, abs=1e-5)


def test_settings_reject_unknown_keys():
    with pytest.raises(ValidationError):
        HashingSettings(dimension=10)


async def test_disk_cache_persists_and_expires(tmp_path):
    cache = DiskCache(DiskCacheSettings(path=str(tmp_path / "c" / "cache.sqlite")))
    assert isinstance(cache, Cache)
    await cache.set("k", {"a": [1, 2]})
    assert await cache.get("k") == {"a": [1, 2]}
    assert await DiskCache(DiskCacheSettings(path=str(tmp_path / "c" / "cache.sqlite"))).get("k")
    assert await cache.get("missing") is None
    await cache.set("short", "x", ttl_s=-1)
    assert await cache.get("short") is None
    await cache.set("k", "replaced")
    assert await cache.get("k") == "replaced"


async def test_memory_cache():
    cache = MemoryCache()
    await cache.set("a", 1)
    assert await cache.get("a") == 1
    await cache.set("b", 2, ttl_s=-1)
    assert await cache.get("b") is None


async def test_resilient_embedder_caches_per_text():
    inner = Flaky()
    embedder = ResilientEmbedder(inner, cache=MemoryCache(), batch_size=2)
    first = await embedder.embed_documents(["aa", "bbb", "cccc"])
    assert inner.doc_calls == [["aa", "bbb"], ["cccc"]]  # batched
    again = await embedder.embed_documents(["bbb", "new"])
    assert inner.doc_calls[-1] == ["new"]  # only the unseen text is embedded
    assert again[0] == pytest.approx(first[1])
    await embedder.embed_query("q")
    await embedder.embed_query("q")
    assert inner.query_calls == ["q"]


async def test_resilient_embedder_preserves_order_with_cache_gaps():
    inner = Flaky()
    embedder = ResilientEmbedder(inner, cache=MemoryCache(), batch_size=10)
    await embedder.embed_documents(["b"])
    vectors = await embedder.embed_documents(["a", "b", "c"])
    assert [v[0] for v in vectors] == [1.0, 1.0, 1.0]
    assert len(vectors) == 3


async def test_resilient_embedder_retries_rate_limits(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("heka.rag.resilience.asyncio.sleep", instant)
    inner = Flaky(failures=2)
    embedder = ResilientEmbedder(inner, max_retries=3)
    assert len(await embedder.embed_documents(["x"])) == 1
    assert inner.failures == 0
