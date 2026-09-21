"""Document health: did each file actually turn into readable text?

Answer quality is capped by extraction quality, and the failures are silent: a scanned PDF becomes
zero characters, a two-column layout becomes interleaved sentences, a font problem becomes
"T h e  c o m p a n y". Nothing errors; the bot just cannot find the answer. This module inspects what
each loader produced, before any tuning, and flags files worth a human look. The checks are heuristics:
a flag means "look at this file", not "this file is broken".
"""

from __future__ import annotations

import re
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

from pydantic import BaseModel, Field

from kbsdk.text import stable_hash

TABULAR_LOADERS = {"csv", "xlsx"}  # legitimately full of numbers and short cells

MIN_CHARS = 300
MIN_CHARS_PER_PAGE = 150
LETTER_RATIO_FLOOR = 0.55
SPACED_LETTER_CEILING = 0.15  # single-letter tokens other than "a" and "I"
MEAN_WORD_LENGTH = (2.5, 12.0)
REPEATED_LINE_CEILING = 0.10  # headers/footers repeated on every page
NO_HEADINGS_CHARS = 4000
# Tuned on a clean 25-document corpus: 120 flagged seven healthy files whose sections are just short
# (~110 characters each, and retrieved perfectly). 80 still catches text cut into one-line pieces.
FRAGMENT_CHUNK_CHARS = 80
REPLACEMENT_CHAR = chr(0xFFFD)  # what decoders emit for bytes they could not read

_WORD = re.compile(r"[A-Za-z0-9']+")
_TABLE_SEPARATOR = re.compile(r"\|?[\s:|-]+\|?")
_VERSION_TOKENS = re.compile(
    r"(?:^|[\s_-])(?:v\d+(?:\.\d+)*|rev\d*|final|draft|copy|old|new|latest|superseded)(?=$|[\s_-])"
    r"|(?:^|[\s_-])(?:19|20)\d{2}(?:[-_]\d{1,2}){0,2}(?=$|[\s_-])|\(\d+\)",
    re.IGNORECASE,
)


class FileHealth(BaseModel):
    path: str
    loader: str
    characters: int = 0
    words: int = 0
    pages: int | None = None
    pages_without_text: int = 0
    ocr_pages: int = 0
    tables: int = 0
    headings: int = 0
    chunks: int = 0
    median_chunk_chars: int = 0
    flags: list[str] = Field(default_factory=list)  # human-readable, one problem each
    error: str | None = None


class HealthReport(BaseModel):
    files: list[FileHealth] = Field(default_factory=list)
    cross_document: list[str] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list)  # files that were not even attempted

    @property
    def flagged(self) -> list[FileHealth]:
        return [f for f in self.files if f.flags or f.error]

    def summary(self) -> str:
        if not self.files:
            return "no documents found"
        chars = sum(f.characters for f in self.files)
        chunks = sum(f.chunks for f in self.files)
        lines = [
            f"documents: {len(self.files)} files | {chunks} chunks | {chars:,} characters | "
            f"{len(self.flagged)} need a look",
            "",
            f"{'file':<38}{'type':<7}{'pages':>6}{'chars':>9}{'chunks':>7}{'tables':>7}{'heads':>6}  status",
        ]
        for f in self.files:
            status = "FAILED" if f.error else (f"{len(f.flags)} flag(s)" if f.flags else "ok")
            pages = "-" if f.pages is None else str(f.pages)
            name = f.path if len(f.path) <= 36 else "..." + f.path[-33:]
            lines.append(
                f"{name:<38}{f.loader:<7}{pages:>6}{f.characters:>9,}{f.chunks:>7}"
                f"{f.tables:>7}{f.headings:>6}  {status}"
            )
        details: list[str] = []
        for f in self.flagged:
            if f.error:
                details.append(f"  {f.path}: could not be read: {f.error}")
            details += [f"  {f.path}: {flag}" for flag in f.flags]
        if details:
            lines += ["", "look at:", *details]
        if self.cross_document:
            lines += ["", "across documents:", *[f"  {note}" for note in self.cross_document]]
        if self.skipped:
            lines += ["", "skipped:", *[f"  {note}" for note in self.skipped]]
        if not details and not self.cross_document and not self.skipped:
            lines += [
                "",
                "nothing suspicious found (heuristics only: spot-check a few documents yourself)",
            ]
        return "\n".join(lines)


# -- per-text statistics -------------------------------------------------------------------------


def count_tables(text: str) -> int:
    """Blocks of consecutive Markdown table rows (lines that start with `|`)."""
    tables, in_table = 0, False
    for line in text.splitlines():
        is_row = line.lstrip().startswith("|")
        if is_row and not in_table:
            tables += 1
        in_table = is_row
    return tables


def count_headings(text: str) -> int:
    return sum(1 for line in text.splitlines() if re.match(r"\s{0,3}#{1,6}\s+\S", line))


def _prose(text: str) -> str:
    """The text without table rows and page-break characters."""
    return "\n".join(
        line for line in text.replace("\f", "\n").splitlines() if not line.lstrip().startswith("|")
    )


def repeated_line_share(text: str) -> float:
    """Share of the characters that sit in lines seen three or more times (page headers/footers)."""
    lines = [line.strip() for line in text.splitlines()]
    # Markdown structure (table separators, headings) repeats by nature and is not page furniture.
    lines = [
        line
        for line in lines
        if len(line) >= 12 and not line.startswith("#") and not _TABLE_SEPARATOR.fullmatch(line)
    ]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(len(line) * n for line, n in counts.items() if n >= 3)
    return repeated / sum(len(line) for line in lines)


def quality_flags(text: str, *, loader: str) -> list[str]:
    """Signs that the text is noise rather than language."""
    prose = _prose(text)
    flags: list[str] = []
    visible = [c for c in prose if not c.isspace()]
    if len(visible) < 200:
        return flags  # too little to judge; the size checks report that
    if prose.count(REPLACEMENT_CHAR) / len(visible) > 0.002:
        flags.append("contains unreadable characters (encoding or font problem)")
    words = _WORD.findall(prose)
    if not words:
        return flags
    if loader not in TABULAR_LOADERS:
        letters = sum(1 for c in visible if c.isalpha())
        if letters / len(visible) < LETTER_RATIO_FLOOR:
            flags.append("mostly non-letters: extraction may have produced noise")
        singles = sum(1 for w in words if len(w) == 1 and w.isalpha() and w.lower() not in "ai")
        if singles / len(words) > SPACED_LETTER_CEILING:
            flags.append("letters appear spaced apart (text read letter by letter)")
        mean = statistics.fmean(len(w) for w in words)
        if not MEAN_WORD_LENGTH[0] <= mean <= MEAN_WORD_LENGTH[1]:
            flags.append(
                f"average word length {mean:.1f} is unusual (words split or glued together?)"
            )
    if repeated_line_share(prose) > REPEATED_LINE_CEILING:
        flags.append(
            "repeated header/footer or boilerplate text fills a large share of the document "
            "(copies of it compete for the same searches and waste context)"
        )
    return flags


def file_health(
    path: str,
    loader: str,
    text: str,
    metadata: Mapping[str, Any],
    chunk_sizes: list[int],
    *,
    ocr_configured: bool = False,
) -> FileHealth:
    """Everything the health report says about one file, from what its loader produced."""
    pages = metadata.get("pages")
    without_text = int(metadata.get("pages_without_text") or 0)
    ocr_pages = int(metadata.get("ocr_pages") or 0)
    health = FileHealth(
        path=path,
        loader=loader,
        characters=len(text.strip()),
        words=len(_WORD.findall(text)),
        pages=int(pages) if isinstance(pages, int) else None,
        pages_without_text=without_text,
        ocr_pages=ocr_pages,
        tables=count_tables(text),
        headings=count_headings(text),
        chunks=len(chunk_sizes),
        median_chunk_chars=int(statistics.median(chunk_sizes)) if chunk_sizes else 0,
    )
    flags = health.flags
    if health.characters == 0:
        flags.append("no text was extracted")
        return health
    if without_text:
        fix = "OCR found nothing on them" if ocr_configured else "set knowledge.ocr to read them"
        flags.append(f"{without_text} page(s) have no text layer (scanned); {fix}")
    if ocr_pages:
        flags.append(f"{ocr_pages} page(s) were read with OCR: check for misread digits and names")
    if loader not in TABULAR_LOADERS:
        if health.pages:
            per_page = health.characters / health.pages
            if per_page < MIN_CHARS_PER_PAGE and not without_text:
                flags.append(
                    f"only {per_page:.0f} characters per page: image-heavy or mostly empty?"
                )
        elif health.characters < MIN_CHARS and not health.tables:  # a short table is not "no text"
            flags.append(f"only {health.characters} characters of text")
    if (
        health.characters > NO_HEADINGS_CHARS
        and health.headings == 0
        and loader not in TABULAR_LOADERS
    ):
        flags.append(
            "no headings found in a long document: chunks carry no section context, so passages "
            "may be hard to find (contextual enrichment or fixed-size overlap can help)"
        )
    if len(chunk_sizes) >= 5 and health.median_chunk_chars < FRAGMENT_CHUNK_CHARS:
        flags.append(f"median chunk is only {health.median_chunk_chars} characters (fragmented)")
    flags += quality_flags(text, loader=loader)
    return health


# -- across files --------------------------------------------------------------------------------


def _family_key(path: str) -> str:
    """A file name with version-like parts (v2, final, 2024, (1)) removed."""
    stem = PurePosixPath(path).stem
    stripped = _VERSION_TOKENS.sub(" ", stem)
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


def cross_document_notes(items: Iterable[tuple[str, str]]) -> list[str]:
    """Notes about files that look like copies or versions of each other: `(path, text)` pairs.

    Conflicting versions are a common source of confident wrong answers, since retrieval cannot
    know which one is current.
    """
    notes: list[str] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    by_family: dict[str, list[str]] = defaultdict(list)
    for path, text in items:
        normalized = " ".join(text.split()).casefold()
        if normalized:
            by_hash[stable_hash(normalized)].append(path)
        key = _family_key(path)
        if key:
            by_family[key].append(path)
    duplicates = {p for group in by_hash.values() if len(group) > 1 for p in group}
    for group in by_hash.values():
        if len(group) > 1:
            notes.append(f"identical content: {', '.join(sorted(group))} (index one of them)")
    for group in by_family.values():
        if len(group) > 1 and not set(group) <= duplicates:
            notes.append(
                f"possible versions of one document: {', '.join(sorted(group))}; if they disagree, "
                "the bot may quote the outdated one (remove the old one or exclude it)"
            )
    return sorted(notes)
