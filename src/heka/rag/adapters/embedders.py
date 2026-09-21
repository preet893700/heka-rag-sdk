"""Embedders, plus `ResilientEmbedder` (batching + cache + throttle + retry) that wraps any of them."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
from collections.abc import Sequence
from typing import Any

import numpy as np

from heka.rag.adapters._deps import Settings, api_key_from_env, require
from heka.rag.adapters.caches import cache_key
from heka.rag.interfaces import Cache, Embedder
from heka.rag.resilience import AsyncRateLimiter, with_retries

_WORD = re.compile(r"\w+")


class HashingSettings(Settings):
    dimensions: int = 384


class HashingEmbedder:
    """Zero-dependency lexical embedder (hashed bag of words).

    Similarity reflects shared words only - there is no semantic understanding - so use it for tests,
    demos and CI, not for real retrieval quality.
    """

    settings_model = HashingSettings

    def __init__(self, settings: HashingSettings | None = None) -> None:
        self.settings = settings or HashingSettings()
        self.model_id = f"hashing-{self.settings.dimensions}"

    @property
    def dimensions(self) -> int:
        return self.settings.dimensions

    def _embed(self, text: str) -> list[float]:
        vector = np.zeros(self.settings.dimensions, dtype=np.float32)
        for word in _WORD.findall(text.lower()):
            digest = hashlib.md5(word.encode("utf-8"), usedforsecurity=False).digest()
            slot = int.from_bytes(digest[:4], "little") % self.settings.dimensions
            vector[slot] += 1.0 if digest[4] & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return (vector / norm if norm else vector).tolist()

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LocalSettings(Settings):
    model: str = "BAAI/bge-small-en-v1.5"
    cache_dir: str | None = None  # where model files are stored; default is fastembed's own
    batch_size: int = 32


class LocalEmbedder:
    """On-device embeddings through fastembed (ONNX, no PyTorch). Nothing leaves the machine."""

    settings_model = LocalSettings

    def __init__(self, settings: LocalSettings | None = None) -> None:
        self.settings = settings or LocalSettings()
        self.model_id = f"local:{self.settings.model}"
        self._fastembed = require("fastembed", stage="embedder", provider="local", extra="local")
        self._model: Any = None
        self._dimensions: int | None = None

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            for info in self._fastembed.TextEmbedding.list_supported_models():
                if info["model"] == self.settings.model:
                    self._dimensions = int(info["dim"])
            if self._dimensions is None:
                self._dimensions = len(self._embed_sync(["dimension probe"], query=False)[0])
        return self._dimensions

    def _load(self) -> Any:
        if self._model is None:
            self._model = self._fastembed.TextEmbedding(
                model_name=self.settings.model, cache_dir=self.settings.cache_dir
            )
        return self._model

    def _embed_sync(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        model = self._load()
        embed = model.query_embed if query else model.embed
        kwargs = {} if query else {"batch_size": self.settings.batch_size}
        return [vector.tolist() for vector in embed(list(texts), **kwargs)]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._embed_sync, texts, query=False)

    async def embed_query(self, text: str) -> list[float]:
        return (await asyncio.to_thread(self._embed_sync, [text], query=True))[0]


class GeminiEmbedSettings(Settings):
    model: str = "models/gemini-embedding-001"
    api_key_env: str = "GOOGLE_API_KEY"
    dimensions: int | None = None  # None = the model's default output size


class GeminiEmbedder:
    settings_model = GeminiEmbedSettings

    def __init__(self, settings: GeminiEmbedSettings) -> None:
        self.settings = settings
        self.model_id = f"gemini:{settings.model}:{settings.dimensions or 'default'}"
        module = require(
            "langchain_google_genai", stage="embedder", provider="gemini", extra="gemini"
        )
        kwargs: dict[str, Any] = {}
        if settings.dimensions:
            kwargs["output_dimensionality"] = settings.dimensions
        self._model = module.GoogleGenerativeAIEmbeddings(
            model=settings.model,
            google_api_key=api_key_from_env(settings.api_key_env, "gemini"),
            **kwargs,
        )
        self._dimensions = settings.dimensions or 0  # learned from the first response if unset

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = await self._model.aembed_documents(list(texts))
        self._dimensions = self._dimensions or (len(vectors[0]) if vectors else 0)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vector: list[float] = await self._model.aembed_query(text)
        self._dimensions = self._dimensions or len(vector)
        return vector


class OpenAIEmbedSettings(Settings):
    model: str = "text-embedding-3-small"
    api_key_env: str = "OPENAI_API_KEY"
    dimensions: int | None = None  # text-embedding-3 models can return shorter vectors
    base_url: str | None = None


class OpenAIEmbedder:
    settings_model = OpenAIEmbedSettings

    def __init__(self, settings: OpenAIEmbedSettings) -> None:
        self.settings = settings
        self.model_id = f"openai:{settings.model}:{settings.dimensions or 'default'}"
        module = require("langchain_openai", stage="embedder", provider="openai", extra="openai")
        self._model = module.OpenAIEmbeddings(
            model=settings.model,
            api_key=api_key_from_env(settings.api_key_env, "openai"),
            dimensions=settings.dimensions,
            base_url=settings.base_url,
        )
        self._dimensions = settings.dimensions or 0  # learned from the first response if unset

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = await self._model.aembed_documents(list(texts))
        self._dimensions = self._dimensions or (len(vectors[0]) if vectors else 0)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vector: list[float] = await self._model.aembed_query(text)
        self._dimensions = self._dimensions or len(vector)
        return vector


class OllamaEmbedSettings(Settings):
    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"


class OllamaEmbedder:
    """Embeddings from a local Ollama server. Nothing leaves the machine."""

    settings_model = OllamaEmbedSettings

    def __init__(self, settings: OllamaEmbedSettings) -> None:
        self.settings = settings
        self.model_id = f"ollama:{settings.model}"
        module = require("langchain_ollama", stage="embedder", provider="ollama", extra="ollama")
        self._model = module.OllamaEmbeddings(model=settings.model, base_url=settings.base_url)
        self._dimensions = 0

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = await self._model.aembed_documents(list(texts))
        self._dimensions = self._dimensions or (len(vectors[0]) if vectors else 0)
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        vector: list[float] = await self._model.aembed_query(text)
        self._dimensions = self._dimensions or len(vector)
        return vector


def _encode(vector: Sequence[float]) -> str:
    return base64.b64encode(np.asarray(vector, dtype=np.float32).tobytes()).decode("ascii")


def _decode(blob: str) -> list[float]:
    return np.frombuffer(base64.b64decode(blob), dtype=np.float32).tolist()


class ResilientEmbedder:
    """Wraps an embedder with per-text caching, batching, a shared rate limiter and retries."""

    def __init__(
        self,
        inner: Embedder,
        *,
        cache: Cache | None = None,
        limiter: AsyncRateLimiter | None = None,
        max_retries: int = 3,
        batch_size: int = 64,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.limiter = limiter or AsyncRateLimiter(None)
        self.max_retries = max_retries
        self.batch_size = batch_size
        self.model_id: str = getattr(inner, "model_id", type(inner).__name__)

    @property
    def dimensions(self) -> int:
        return self.inner.dimensions

    def _key(self, kind: str, text: str) -> str:
        return cache_key("emb", self.model_id, kind, text)

    async def _embed_batch(self, texts: list[str], *, query: bool) -> list[list[float]]:
        async def call() -> list[list[float]]:
            if query:
                return [await self.inner.embed_query(texts[0])]
            return await self.inner.embed_documents(texts)

        return await with_retries(
            call, max_retries=self.max_retries, before_attempt=self.limiter.wait
        )

    async def _embed(self, texts: Sequence[str], *, query: bool) -> list[list[float]]:
        kind = "q" if query else "d"
        results: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for index, text in enumerate(texts):
            cached = await self.cache.get(self._key(kind, text)) if self.cache else None
            if cached is not None:
                results[index] = _decode(cached)
            else:
                missing.append(index)
        step = 1 if query else self.batch_size
        for start in range(0, len(missing), step):
            batch = missing[start : start + step]
            vectors = await self._embed_batch([texts[i] for i in batch], query=query)
            for index, vector in zip(batch, vectors, strict=True):
                results[index] = vector
                if self.cache:
                    await self.cache.set(self._key(kind, texts[index]), _encode(vector))
        return [vector for vector in results if vector is not None]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed(texts, query=False)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], query=True))[0]
