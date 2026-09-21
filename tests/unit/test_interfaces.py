"""The Protocols are structural: any class with the right methods conforms, no base class needed."""

from heka.rag.interfaces import Cache, Embedder, Tracer


class FakeEmbedder:
    dimensions = 3

    async def embed_documents(self, texts):
        return [[0.0, 0.0, 0.0] for _ in texts]

    async def embed_query(self, text):
        return [0.0, 0.0, 0.0]


class FakeCache:
    def __init__(self):
        self.data = {}

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, ttl_s=None):
        self.data[key] = value


def test_fakes_conform():
    assert isinstance(FakeEmbedder(), Embedder)
    assert isinstance(FakeCache(), Cache)


def test_non_conforming_object_is_rejected():
    assert not isinstance(object(), Tracer)


async def test_fake_embedder_round_trip():
    assert await FakeEmbedder().embed_query("x") == [0.0, 0.0, 0.0]
