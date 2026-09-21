"""Chunkers. Both produce `Chunk`s whose `text` is the verbatim source text (so citations can be
verified against it) and whose metadata carries the document title, heading path and page.

* `structure_aware` splits on Markdown headings, keeps tables intact, and records the heading path
  ("Leave policy > Casual leave") that is embedded alongside the text and shown in citations.
* `fixed` slides a window of `chunk_size` characters with `chunk_overlap`, ignoring structure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import Field, model_validator

from heka.rag.adapters._deps import Settings
from heka.rag.types import Chunk, Document

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n")


class ChunkerSettings(Settings):
    chunk_size: int = Field(default=1200, ge=100, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=150, ge=0, description="Characters repeated between chunks")

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> ChunkerSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def _tail(text: str, size: int) -> str:
    """The last ~`size` characters of `text`, cut at a word boundary."""
    if size <= 0 or len(text) <= size:
        return text if size > 0 else ""
    cut = text[-size:]
    space = cut.find(" ")
    return cut[space + 1 :] if 0 <= space < len(cut) - 1 else cut


def _hard_split(text: str, size: int) -> list[str]:
    parts: list[str] = []
    while len(text) > size:
        cut = text.rfind(" ", 0, size)
        cut = cut if cut > size // 2 else size
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


def _is_table(text: str) -> bool:
    lines = text.split("\n")
    return len(lines) >= 3 and all(line.lstrip().startswith("|") for line in lines)


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """Split one oversized paragraph. Tables split by row, repeating the header row."""
    if _is_table(text):
        lines = text.split("\n")
        header, rows = "\n".join(lines[:2]), lines[2:]
        parts, current = [], header
        for row in rows:
            if len(current) + 1 + len(row) > size and current != header:
                parts.append(current)
                current = header
            current += "\n" + row
        if current != header:
            parts.append(current)
        return parts

    pieces: list[str] = []
    for sentence in (s for s in _SENTENCE_BREAK.split(text) if s.strip()):
        pieces.extend(_hard_split(sentence, size) if len(sentence) > size else [sentence])
    parts, current = [], ""
    for piece in pieces:
        joined = f"{current} {piece}".strip() if current else piece
        if current and len(joined) > size:
            parts.append(current)
            seed = _tail(current, overlap)
            current = f"{seed} {piece}".strip() if seed and len(seed) + len(piece) < size else piece
        else:
            current = joined
    if current:
        parts.append(current)
    return parts


@dataclass
class _Section:
    path: list[str]
    paragraphs: list[tuple[str, int]] = field(default_factory=list)  # (text, page)


def _sections(text: str) -> list[_Section]:
    """Split Markdown-flavoured text into sections, tracking heading path and page (\\f-separated)."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    current = _Section(path=[])
    buffer: list[str] = []
    in_fence = False

    def flush(page: int) -> None:
        if buffer:
            paragraph = "\n".join(buffer).strip()
            if paragraph:
                current.paragraphs.append((paragraph, page))
            buffer.clear()

    for page_number, page_text in enumerate(text.split("\f"), start=1):
        for line in page_text.split("\n"):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
            heading = None if in_fence else _HEADING.match(line)
            if heading:
                flush(page_number)
                if current.paragraphs:
                    sections.append(current)
                level, title = len(heading.group(1)), heading.group(2).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current = _Section(path=[t for _, t in stack])
            elif not line.strip() and not in_fence:
                flush(page_number)
            else:
                buffer.append(line.rstrip())
        flush(page_number)
    if current.paragraphs:
        sections.append(current)
    return sections


def _context(title: str, path: list[str]) -> str:
    if path and path[0].casefold() == title.casefold():
        return " > ".join(path)
    return " > ".join([p for p in [title, *path] if p])


def _make_chunk(
    document: Document, index: int, text: str, path: list[str], page: int, page_end: int
) -> Chunk:
    title = str(document.metadata.get("title") or document.source)
    metadata = {
        **document.metadata,
        "source": document.source,
        "title": title,
        "heading_path": " > ".join(path),
        "heading": path[-1] if path else "",
        "page": page,
        "page_end": page_end,
        "context": _context(title, path),
    }
    return Chunk(
        id=f"{document.id}:{index:04d}",
        doc_id=document.id,
        text=text,
        index=index,
        metadata=metadata,
    )


class StructureAwareChunker:
    settings_model = ChunkerSettings

    def __init__(self, settings: ChunkerSettings | None = None) -> None:
        self.settings = settings or ChunkerSettings()

    async def chunk(self, document: Document) -> list[Chunk]:
        size, overlap = self.settings.chunk_size, self.settings.chunk_overlap
        chunks: list[Chunk] = []
        for section in _sections(document.text):
            # (text, page, is_split_part): split parts already carry their own overlap
            pieces: list[tuple[str, int, bool]] = []
            for text, page in section.paragraphs:
                if len(text) <= size:
                    pieces.append((text, page, False))
                else:
                    pieces.extend((part, page, True) for part in _split_long(text, size, overlap))

            current: list[str] = []
            pages: list[int] = []
            length = 0
            for text, page, is_split in pieces:
                added = len(text) + (2 if current else 0)
                if current and length + added > size:
                    body = "\n\n".join(current)
                    chunks.append(
                        _make_chunk(document, len(chunks), body, section.path, pages[0], pages[-1])
                    )
                    seed = "" if is_split else _tail(body, overlap)
                    current, pages = ([seed], [pages[-1]]) if seed else ([], [])
                    length = len(seed)
                    added = len(text) + (2 if current else 0)
                current.append(text)
                pages.append(page)
                length += added
            if current:
                body = "\n\n".join(current)
                chunks.append(
                    _make_chunk(document, len(chunks), body, section.path, pages[0], pages[-1])
                )
        return chunks


class FixedChunker:
    settings_model = ChunkerSettings

    def __init__(self, settings: ChunkerSettings | None = None) -> None:
        self.settings = settings or ChunkerSettings()

    async def chunk(self, document: Document) -> list[Chunk]:
        size, overlap = self.settings.chunk_size, self.settings.chunk_overlap
        clean_parts, starts, offset = [], [], 0
        for part in document.text.split("\f"):
            starts.append(offset)
            clean_parts.append(part)
            offset += len(part)
        text = "".join(clean_parts)

        def page_at(position: int) -> int:
            page = 1
            for number, start in enumerate(starts, start=1):
                if position >= start:
                    page = number
            return page

        chunks: list[Chunk] = []
        start, length = 0, len(text)
        while start < length:
            end = min(length, start + size)
            if end < length:
                boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                end = boundary if boundary > start + size // 2 else end
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    _make_chunk(
                        document,
                        len(chunks),
                        piece,
                        [],
                        page_at(start),
                        page_at(max(start, end - 1)),
                    )
                )
            if end >= length:
                break
            start = max(end - overlap, start + 1)
        return chunks
