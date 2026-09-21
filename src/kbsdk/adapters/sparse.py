"""BM25 keyword search: the exact-term half of hybrid retrieval.

Dense embeddings capture meaning but blur exact tokens (form codes, names, numbers, "Germany" vs "France").
BM25 is the opposite, so combining them covers both. Pure Python + NumPy, no extra dependencies.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Sequence

import numpy as np

from kbsdk.filters import matches, validate_filter
from kbsdk.text import embedding_text
from kbsdk.types import Chunk, Filter, ScoredChunk

_STOPWORDS = frozenset(
    (  # noqa: SIM905 - a readable word list beats a 60-item literal
        "a an and are as at be been but by can could did do does for from had has have how i if in "
        "into is it its of on or so than that the their them then there these they this those to was "
        "were what when where which who whom why will with would you your"
    ).split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_SEPARATORS = re.compile(r"[-_./]")


def _stem(token: str) -> str:
    """Fold simple plurals only ("days" -> "day", "policies" -> "policy"); more would mangle words."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Identifiers like `hr-207` or `2.5` are kept whole *and* split, so both
    "HR-207" and "HR 207" find them."""
    tokens: list[str] = []
    for raw in _TOKEN.findall(text.lower().replace("'", "")):
        parts = _SEPARATORS.split(raw)
        if len(parts) > 1:
            tokens.append(raw)
        tokens.extend(_stem(p) for p in parts if p and p not in _STOPWORDS)
    return tokens


class Bm25Index:
    """Okapi BM25 over chunk text plus its heading context, with metadata filtering."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._chunks: list[Chunk] = []
        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._lengths = np.zeros(0)
        self._avg_length = 1.0

    def __len__(self) -> int:
        return len(self._chunks)

    def build(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        postings: dict[str, tuple[list[int], list[int]]] = {}
        lengths: list[int] = []
        for index, chunk in enumerate(self._chunks):
            counts = Counter(tokenize(embedding_text(chunk)))
            lengths.append(sum(counts.values()))
            for term, frequency in counts.items():
                docs, freqs = postings.setdefault(term, ([], []))
                docs.append(index)
                freqs.append(frequency)
        self._postings = {
            term: (np.asarray(docs, dtype=np.int64), np.asarray(freqs, dtype=np.float64))
            for term, (docs, freqs) in postings.items()
        }
        self._lengths = np.asarray(lengths, dtype=np.float64)
        self._avg_length = float(self._lengths.mean()) if len(lengths) else 1.0

    def search(self, query: str, *, k: int, filter: Filter | None = None) -> list[ScoredChunk]:
        if not self._chunks or k <= 0:
            return []
        validate_filter(filter)
        total = len(self._chunks)
        scores = np.zeros(total)
        norm = 1.0 - self.b + self.b * self._lengths / (self._avg_length or 1.0)
        for term in set(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            docs, freqs = posting
            idf = math.log(1.0 + (total - len(docs) + 0.5) / (len(docs) + 0.5))
            scores[docs] += idf * freqs * (self.k1 + 1.0) / (freqs + self.k1 * norm[docs])
        candidates = np.flatnonzero(scores > 0)
        if not len(candidates):
            return []
        order = candidates[np.argsort(-scores[candidates], kind="stable")]
        results: list[ScoredChunk] = []
        for index in order:
            chunk = self._chunks[int(index)]
            if filter and not matches(filter, chunk.metadata):
                continue
            score = float(scores[index])
            results.append(
                ScoredChunk(chunk=chunk, score=score, retriever="sparse", signals={"sparse": score})
            )
            if len(results) >= k:
                break
        return results


class SharedBm25:
    """A `Bm25Index` that rebuilds itself when the underlying store changes (thread-safe)."""

    def __init__(self) -> None:
        self._index = Bm25Index()
        self._key: object = None
        self._lock = threading.Lock()

    def ensure(self, key: object, chunks: Sequence[Chunk]) -> Bm25Index:
        with self._lock:
            if key != self._key:
                index = Bm25Index()
                index.build(chunks)
                self._index, self._key = index, key
            return self._index

    @property
    def key(self) -> object:
        return self._key
