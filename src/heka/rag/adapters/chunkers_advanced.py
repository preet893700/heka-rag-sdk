"""Advanced chunkers.

* `parent_child` - retrieval precision vs answer context. Small "child" chunks are what gets searched
  (a focused passage matches a focused question); each child carries its whole parent section, which
  is what the model reads. Children of one parent collapse to a single result.
* `semantic` - boundaries follow the meaning. Sentences are embedded and the section is cut where
  adjacent sentences are least alike, so a chunk tends to be about one thing.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import Field, model_validator

from heka.rag.adapters._deps import Settings
from heka.rag.adapters.chunkers import (
    _SENTENCE_BREAK,
    ChunkerSettings,
    StructureAwareChunker,
    _is_table,
    _make_chunk,
    _sections,
    _split_long,
    _tail,
)
from heka.rag.interfaces import Embedder
from heka.rag.types import Chunk, Document


def _pack(paragraphs: Sequence[str], size: int, overlap: int) -> list[str]:
    """Greedily pack paragraphs into pieces of at most ~`size` characters."""
    pieces: list[tuple[str, bool]] = []  # (text, came from splitting a long paragraph)
    for paragraph in paragraphs:
        if len(paragraph) <= size:
            pieces.append((paragraph, False))
        else:
            pieces.extend((part, True) for part in _split_long(paragraph, size, overlap))
    packed: list[str] = []
    current: list[str] = []
    length = 0
    for text, is_split in pieces:
        added = len(text) + (2 if current else 0)
        if current and length + added > size:
            body = "\n\n".join(current)
            packed.append(body)
            seed = "" if is_split else _tail(body, overlap)
            current, length = ([seed], len(seed)) if seed else ([], 0)
            added = len(text) + (2 if current else 0)
        current.append(text)
        length += added
    if current:
        packed.append("\n\n".join(current))
    return packed


class ParentChildSettings(Settings):
    parent_size: int = Field(
        default=2400, ge=400, description="Characters in the section the model reads"
    )
    child_size: int = Field(default=400, ge=100, description="Characters in a searchable chunk")
    child_overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def _sizes_make_sense(self) -> ParentChildSettings:
        if self.child_size >= self.parent_size:
            raise ValueError("child_size must be smaller than parent_size")
        if self.child_overlap >= self.child_size:
            raise ValueError("child_overlap must be smaller than child_size")
        return self


class ParentChildChunker:
    settings_model = ParentChildSettings

    def __init__(self, settings: ParentChildSettings | None = None) -> None:
        self.settings = settings or ParentChildSettings()
        self._parents = StructureAwareChunker(
            ChunkerSettings(chunk_size=self.settings.parent_size, chunk_overlap=0)
        )

    async def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for parent_number, parent in enumerate(await self._parents.chunk(document)):
            children = _pack(
                parent.text.split("\n\n"), self.settings.child_size, self.settings.child_overlap
            )
            for child_number, text in enumerate(children):
                chunks.append(
                    Chunk(
                        id=f"{parent.id}.{child_number:02d}",
                        doc_id=document.id,
                        text=text,
                        index=len(chunks),
                        parent_id=parent.id,
                        metadata={
                            **parent.metadata,
                            "parent_id": parent.id,
                            "parent_index": parent_number,
                            "parent_text": parent.text,
                        },
                    )
                )
        return chunks


class SemanticSettings(Settings):
    max_chunk_size: int = Field(default=1500, ge=200)
    min_chunk_size: int = Field(default=250, ge=50)
    breakpoint_percentile: float = Field(
        default=85.0, ge=50.0, le=99.0, description="Cut at the least-similar N% of sentence gaps"
    )
    window: int = Field(default=1, ge=1, le=5, description="Sentences per side when comparing")

    @model_validator(mode="after")
    def _sizes_make_sense(self) -> SemanticSettings:
        if self.min_chunk_size >= self.max_chunk_size:
            raise ValueError("min_chunk_size must be smaller than max_chunk_size")
        return self


def _cosine_distances(vectors: np.ndarray, window: int) -> np.ndarray:
    """Distance between the mean of the `window` sentences before each gap and the ones after it."""
    count = len(vectors)
    distances = np.zeros(count - 1)
    for gap in range(count - 1):
        before = vectors[max(0, gap - window + 1) : gap + 1].mean(axis=0)
        after = vectors[gap + 1 : gap + 1 + window].mean(axis=0)
        norm = float(np.linalg.norm(before) * np.linalg.norm(after))
        distances[gap] = 1.0 - (float(before @ after) / norm if norm else 0.0)
    return distances


class SemanticChunker:
    """Costs one embedding per sentence at ingest time (cached for hosted embedders)."""

    settings_model = SemanticSettings

    def __init__(self, settings: SemanticSettings | None = None, *, embedder: Embedder) -> None:
        self.settings = settings or SemanticSettings()
        self.embedder = embedder

    async def _groups(self, units: list[tuple[str, int]]) -> list[list[tuple[str, int]]]:
        settings = self.settings
        if len(units) < 3 or sum(len(t) for t, _ in units) <= settings.min_chunk_size:
            return [units]
        vectors = np.asarray(
            await self.embedder.embed_documents([text for text, _ in units]), dtype=np.float64
        )
        distances = _cosine_distances(vectors, settings.window)
        threshold = float(np.percentile(distances, settings.breakpoint_percentile))
        groups: list[list[tuple[str, int]]] = [[units[0]]]
        for gap, distance in enumerate(distances):
            current_size = sum(len(t) for t, _ in groups[-1])
            too_big = current_size + len(units[gap + 1][0]) > settings.max_chunk_size
            if (distance > threshold and current_size >= settings.min_chunk_size) or too_big:
                groups.append([])
            groups[-1].append(units[gap + 1])
        if len(groups) > 1 and sum(len(t) for t, _ in groups[-1]) < settings.min_chunk_size:
            tail = groups.pop()  # a short last group reads better attached to its neighbour
            groups[-1].extend(tail)
        return groups

    async def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in _sections(document.text):
            # Prose is grouped by meaning; a table is always a chunk of its own, never glued to prose.
            groups: list[list[tuple[str, int]]] = []
            prose: list[tuple[str, int]] = []
            for paragraph, page in section.paragraphs:
                if _is_table(paragraph):
                    if prose:
                        groups.extend(await self._groups(prose))
                        prose = []
                    groups.append([(paragraph, page)])
                else:
                    prose.extend(
                        (s.strip(), page) for s in _SENTENCE_BREAK.split(paragraph) if s.strip()
                    )
            if prose:
                groups.extend(await self._groups(prose))
            for group in groups:
                if not group:
                    continue
                body = " ".join(t for t, _ in group)
                for part in (
                    _split_long(body, self.settings.max_chunk_size, 0)
                    if len(body) > (self.settings.max_chunk_size)
                    else [body]
                ):
                    chunks.append(
                        _make_chunk(
                            document, len(chunks), part, section.path, group[0][1], group[-1][1]
                        )
                    )
        return chunks
