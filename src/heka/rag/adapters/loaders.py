"""File loaders. Each turns one file into a single Markdown-flavoured `Document`.

Everything is normalised to lightweight Markdown (`#` headings, `|` tables, `-` lists) so the
structure-aware chunker treats every format the same way. PDF pages are separated by a form feed
(`\\f`), which the chunkers use to record page numbers.
"""

from __future__ import annotations

import asyncio
import csv
import os
import re
import statistics
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from heka.rag.adapters._deps import require
from heka.rag.errors import ConfigError
from heka.rag.interfaces import OCREngine
from heka.rag.text import stable_hash
from heka.rag.types import Document

EXTENSION_LOADERS: dict[str, str] = {
    ".txt": "text",
    ".md": "text",
    ".markdown": "text",
    ".rst": "text",
    ".pdf": "pdf",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".xlsm": "xlsx",
    ".csv": "csv",
    ".html": "html",
    ".htm": "html",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".tif": "image",
    ".tiff": "image",
    ".bmp": "image",
    ".webp": "image",
}


def _import(module: str, extra: str, provider: str) -> Any:
    return require(module, stage="loader", provider=provider, extra=extra)


def to_markdown_table(rows: list[list[str]]) -> str:
    rows = [
        [" ".join(cell.split()).replace("|", "\\|") for cell in row] for row in rows if any(row)
    ]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + " --- |" * width]
    lines += ["| " + " | ".join(row) + " |" for row in padded[1:]]
    return "\n".join(lines)


class _FileLoader:
    """Base: subclasses implement `_parse(path) -> (markdown, metadata)`.

    `extraction` and `ocr` come from the knowledge config (see `KnowledgeConfig`); loaders that have
    no use for them simply ignore them.
    """

    file_type = ""

    def __init__(self, *, extraction: str = "layout", ocr: OCREngine | None = None) -> None:
        self.extraction = extraction
        self.ocr = ocr

    def _parse(self, path: Path) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def _document(self, path: Path, text: str, meta: dict[str, Any]) -> Document:
        title = str(meta.pop("title", "") or path.stem)
        return Document(
            id=stable_hash(str(path.resolve()), length=16),
            source=path.name,
            text=text,
            metadata={"title": title, "file_type": self.file_type, **meta},
        )

    async def load(self, location: str) -> AsyncIterator[Document]:
        path = Path(location)
        text, meta = await asyncio.to_thread(self._parse, path)
        yield self._document(path, text, meta)


class TextLoader(_FileLoader):
    file_type = "text"

    def _parse(self, path: Path) -> tuple[str, dict[str, Any]]:
        text = path.read_text(encoding="utf-8", errors="replace")
        heading = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
        return text, {"title": heading.group(1) if heading else ""}


_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)+)[.)]?\s+[A-Z].{2,80}$")
_SECTION_HEADING = re.compile(
    r"^(?:section|article|chapter|part)\s+\d+[.:]?\s+.{2,80}$", re.IGNORECASE
)
_ALL_CAPS_HEADING = re.compile(r"^[A-Z][A-Z0-9 &/,\-']{3,59}$")


def promote_headings(text: str) -> str:
    """Turn heading-looking lines of plain text (PDF) into Markdown headings.

    Deliberately conservative: only dotted numbering ("3.2 Casual leave"), "Section 4 ..." lines and
    short ALL-CAPS lines qualify, so numbered list items ("1. Submit the form") are left alone.
    """
    # Pages are separated by a form feed, which would otherwise be glued to the first line of the
    # next page and hide a heading there. So each page is processed on its own.
    return "\f".join(_promote_page(page) for page in text.split("\f"))


def _promote_page(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        level = 0
        if stripped and len(stripped) <= 90 and not stripped.endswith((".", ":", ",")):
            if numbered := _NUMBERED_HEADING.match(stripped):
                level = 1 + numbered.group(1).count(".")
            elif _SECTION_HEADING.match(stripped) or (
                _ALL_CAPS_HEADING.match(stripped) and len(stripped.split()) >= 2
            ):
                level = 1
        out.append(f"{'#' * level} {stripped}" if level else line)
    return "\n".join(out)


_DIGIT_RUN = re.compile(r"\d+")
_DIGIT_HEAVY_SHARE = 0.2  # share of a line's letters+digits that must be digits to normalise them


def _boilerplate_key(line: str) -> str:
    """Loose fingerprint of a line, so "Page 1 of 59" and "Page 2 of 59" count as the same running
    footer even though the page number changes. Digits are only collapsed for lines that are mostly
    digits/punctuation to begin with (page numbers, dates, "3 of 59"); an ordinary sentence that
    happens to contain one number ("Section 1 covers...") is matched on its exact text only, so it
    is never confused with a different sentence on a different page."""
    text = " ".join(line.split())
    if len(text) < 4 or len(text) > 120:
        return ""
    alnum = [c for c in text if c.isalnum()]
    if alnum and sum(c.isdigit() for c in alnum) / len(alnum) >= _DIGIT_HEAVY_SHARE:
        return _DIGIT_RUN.sub("#", text)
    return text


def strip_repeated_lines(pages: list[str]) -> list[str]:
    """Drop lines that repeat, near-identically, across most pages: running headers and footers
    ("Acme Corp - Confidential", "Page 3 of 59"). Left in, they repeat in every chunk of a long
    document, competing with real content in retrieval and wasting context.

    Deliberately conservative: a document needs several pages, and a line must appear on at least
    half of them (never fewer than 3), before it is treated as page furniture rather than content
    that legitimately repeats a couple of times.
    """
    if len(pages) < 4:
        return pages
    page_lines = [page.split("\n") for page in pages]
    counts: Counter[str] = Counter()
    for lines in page_lines:
        for key in {_boilerplate_key(line.strip()) for line in lines} - {""}:
            counts[key] += 1
    threshold = max(3, len(pages) // 2)
    repeated = {key for key, n in counts.items() if n >= threshold}
    if not repeated:
        return pages
    return [
        "\n".join(line for line in lines if _boilerplate_key(line.strip()) not in repeated)
        for lines in page_lines
    ]


MIN_PAGE_CHARS = 20  # a page with less text than this is treated as scanned
RENDER_DPI = 250  # resolution pages are rendered at before OCR

_BOLD_FONT = re.compile(r"bold|black|heavy|semibold", re.IGNORECASE)
_HEADING_SIZE_RATIO = 1.15  # a line set this much larger than the page's body text is a heading
_BOLD_HEADING_SIZE_RATIO = 0.95  # bold text need not be larger, just not smaller, than body text
_TABULAR_MIN_GAP = 15  # points; how clear of its neighbour a word must sit to count as a "column"


def _line_record(line: dict[str, Any]) -> dict[str, Any]:
    """A pdfplumber text-line dict, reduced to what heading detection and column order need."""
    chars = line.get("chars") or []
    sizes = [c["size"] for c in chars if c.get("size")]
    size = statistics.median(sizes) if sizes else 0.0
    bold = bool(chars) and sum(1 for c in chars if _BOLD_FONT.search(c.get("fontname") or "")) > len(
        chars
    ) / 2
    return {
        "top": float(line["top"]),
        "x0": float(line["x0"]),
        "x1": float(line["x1"]),
        "text": str(line["text"]),
        "size": size,
        "bold": bold,
    }


def _chars_to_record(chars: list[dict[str, Any]]) -> dict[str, Any]:
    """Rebuild a line record straight from characters (used only for a piece split out of a merged
    line below, where pdfplumber's own ready-made `text` no longer applies to just that piece).

    Characters carry no explicit space glyphs, so a space is inserted wherever the gap to the next
    character is wider than ordinary letter spacing - the same signal `_split_line_at_column_gap`
    uses to find the column break in the first place.
    """
    ordered = sorted(chars, key=lambda c: c["x0"])
    sizes = [c["size"] for c in ordered if c.get("size")]
    size = statistics.median(sizes) if sizes else 0.0
    bold = sum(1 for c in ordered if _BOLD_FONT.search(c.get("fontname") or "")) > len(ordered) / 2
    parts = [ordered[0]["text"]]
    for prev, cur in zip(ordered, ordered[1:], strict=False):
        threshold = max(1.0, (cur.get("size") or size or 1.0) * 0.15)
        if cur["x0"] - prev["x1"] > threshold:
            parts.append(" ")
        parts.append(cur["text"])
    return {
        "top": min(c["top"] for c in ordered),
        "x0": ordered[0]["x0"],
        "x1": ordered[-1]["x1"],
        "text": "".join(parts),
        "size": size,
        "bold": bold,
    }


_COLUMN_GAP_MIN = 30.0  # points; a gap this wide within one "line" signals two side-by-side columns
_COLUMN_GAP_SLACK = 60.0  # points either side of the midline that a gap can straddle and still count


def _split_line_at_column_gap(line: dict[str, Any], mid: float) -> list[dict[str, Any]]:
    """`extract_text_lines()` merges same-row text across columns into a single line (it groups
    purely by vertical position), which would hide a two-column layout before `_reading_order` ever
    sees it. Split such a line back into a left and right piece when it has an unusually wide gap
    that straddles the page's midline; an ordinary line, with only normal word-spacing gaps, comes
    back as a single record, exactly as `_line_record` would have produced before this fix.
    """
    chars = line.get("chars") or []
    if len(chars) < 4:
        return [_line_record(line)]
    ordered = sorted(chars, key=lambda c: c["x0"])
    gap, split_at = max(
        ((ordered[i]["x0"] - ordered[i - 1]["x1"], i) for i in range(1, len(ordered))),
        key=lambda item: item[0],
    )
    boundary = (ordered[split_at - 1]["x1"] + ordered[split_at]["x0"]) / 2
    if gap < _COLUMN_GAP_MIN or not (mid - _COLUMN_GAP_SLACK <= boundary <= mid + _COLUMN_GAP_SLACK):
        return [_line_record(line)]
    return [_chars_to_record(ordered[:split_at]), _chars_to_record(ordered[split_at:])]


def _modal_size(sizes: list[float]) -> float:
    """The page's most common font size, i.e. its body text - the yardstick headings stand out from."""
    buckets = Counter(round(s * 2) / 2 for s in sizes if s > 0)
    return buckets.most_common(1)[0][0] if buckets else 0.0


def _is_font_heading(text: str, size: float, bold: bool, body_size: float) -> bool:
    """A line set in a larger or bold font than the surrounding body text - a heading regardless of
    wording or casing (unlike the regex patterns below, which only catch numbered/ALL-CAPS style)."""
    if not text or len(text) > 90 or text.endswith((".", ":", ",")) or len(text.split()) < 2:
        return False
    if body_size <= 0:
        return False
    return size >= body_size * _HEADING_SIZE_RATIO or (bold and size >= body_size * _BOLD_HEADING_SIZE_RATIO)


def _heading_line(record: dict[str, Any], body_size: float) -> str:
    text = str(record["text"])
    stripped = text.strip()
    if _is_font_heading(stripped, record["size"], record["bold"], body_size):
        return f"# {stripped}"
    return text


def _reading_order(records: list[dict[str, Any]], page_width: float) -> list[dict[str, Any]]:
    """Left-to-right, top-to-bottom order for a page that may be laid out in two columns.

    An ordinary single-column page is returned unchanged (sorted by vertical position); a page is
    only treated as two columns when there is clear, separable evidence for it - most lines sitting
    entirely left or entirely right of the page's midline - so this never reorders a normal page
    just because a few lines happen to start left of centre.
    """
    if len(records) < 6:
        return sorted(records, key=lambda r: r["top"])
    mid = page_width / 2
    margin = 18.0
    tagged: list[tuple[str, dict[str, Any]]] = []
    for record in records:
        if record["x1"] <= mid + margin:
            tagged.append(("left", record))
        elif record["x0"] >= mid - margin:
            tagged.append(("right", record))
        else:
            tagged.append(("span", record))
    left = [r for col, r in tagged if col == "left"]
    right = [r for col, r in tagged if col == "right"]
    spanning = [r for col, r in tagged if col == "span"]
    if len(left) < 3 or len(right) < 3 or len(spanning) > len(records) * 0.5:
        return sorted(records, key=lambda r: r["top"])
    # Full-width lines (a heading above two columns) reset the columns, so a title in between two
    # column blocks does not get stranded after both of them.
    ordered: list[dict[str, Any]] = []
    pending_left: list[dict[str, Any]] = []
    pending_right: list[dict[str, Any]] = []
    for col, record in sorted(tagged, key=lambda item: item[1]["top"]):
        if col == "span":
            ordered += sorted(pending_left, key=lambda r: r["top"])
            ordered += sorted(pending_right, key=lambda r: r["top"])
            pending_left, pending_right = [], []
            ordered.append(record)
        elif col == "left":
            pending_left.append(record)
        else:
            pending_right.append(record)
    ordered += sorted(pending_left, key=lambda r: r["top"])
    ordered += sorted(pending_right, key=lambda r: r["top"])
    return ordered


def _mentions_password(exc: BaseException) -> bool:
    """Whether a password problem appears anywhere in an exception, its cause chain, or its args.

    pdfplumber/pdfminer wrap the real `PDFPasswordIncorrect` inside a generic `PdfminerException`
    whose own type name and message say nothing about a password - the original error is only
    reachable via `.args` (`str(exc)` is empty) - so a plain `"password" in str(exc)` check misses it.
    """
    seen: set[int] = set()
    stack = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if "password" in type(current).__name__.lower() or "password" in str(current).lower():
            return True
        if current.__cause__ is not None:
            stack.append(current.__cause__)
        stack.extend(arg for arg in getattr(current, "args", ()) if isinstance(arg, BaseException))
    return False


def _word_rows(page: Any) -> list[list[Any]]:
    """Words on a page, grouped into visual rows by vertical position, each row left-to-right."""
    rows: dict[float, list[Any]] = defaultdict(list)
    for word in page.extract_words():
        rows[round(word["top"])].append(word)
    return [sorted(row, key=lambda w: w["x0"]) for _, row in sorted(rows.items())]


def _row_groups(row_words: list[Any]) -> list[list[Any]]:
    """A row's words split into column groups: a run of words with only ordinary word-spacing
    between them is one group (a cell can itself hold more than one word, e.g. "Monthly Rent"); a
    gap of `_TABULAR_MIN_GAP` or more starts a new group (a column boundary)."""
    groups = [[row_words[0]]]
    for prev, word in zip(row_words, row_words[1:], strict=False):
        if word["x0"] - prev["x1"] >= _TABULAR_MIN_GAP:
            groups.append([])
        groups[-1].append(word)
    return groups


_ALIGN_TOLERANCE = 6.0  # points; how far two cells' edges may differ and still be "the same column"
_MIN_TABLE_ROWS = 3
_MAX_MISSING_WORDS = 0.02  # a table rendering may lose at most this share of a page's words
_PROSE_CELL_WORDS = 5  # a cell this long is a line of running text, not a table cell
_PROSE_CELL_SHARE = 0.6  # ...and when most cells are, the "table" is multi-column prose


def _columns_line_up(run: list[list[list[Any]]]) -> bool:
    """Whether the rows in `run` (each a list of column groups) share three or more columns.

    A real table repeats the same column edges row after row (left-, right- or centre-aligned), while
    justified prose has wide gaps at different places on every line. Without this check a page of
    ordinary text with a few wide gaps looks like a table.
    """
    edges: list[Callable[[list[Any]], float]] = [
        lambda g: float(g[0]["x0"]),
        lambda g: float(g[-1]["x1"]),
        lambda g: float((g[0]["x0"] + g[-1]["x1"]) / 2),
    ]
    for edge in edges:
        buckets: Counter[int] = Counter()
        for groups in run:
            hit = {round(edge(g) / _ALIGN_TOLERANCE) for g in groups}
            buckets.update(hit | {b + 1 for b in hit})  # neighbours: an edge near a bucket boundary
        needed = max(_MIN_TABLE_ROWS, -(-len(run) * 6 // 10))  # in at least 60% of the rows
        if sum(1 for n in buckets.values() if n >= needed) >= 6:  # each real column counts twice
            return True
    return False


def _borderless_tables(
    page: Any,
) -> list[tuple[tuple[float, float, float, float], list[list[str]]]]:
    """Recover whitespace-aligned tables' rows and columns straight from word gaps.

    Each table is a *run of consecutive text rows* that split into three or more column groups and
    whose columns line up. Consecutive matters: a box drawn from the first to the last table-looking
    row on a page would also swallow the prose lines between them (and, since only table rows are
    written out, silently delete them).

    Deliberately not `page.find_tables(table_settings={"vertical_strategy": "text", ...})`:
    pdfplumber's own text-strategy table sizes each column from where words in it *usually* start,
    then clips extraction to that box - so a cell that runs wider than the others in its column (a
    real name like "Nakamura" under shorter ones like "Vacant") is silently truncated. Clustering by
    gaps reads every word in full, wherever it sits.
    """
    rows = _word_rows(page)
    grouped = [_row_groups(row) if len(row) >= 2 else None for row in rows]
    tables: list[tuple[tuple[float, float, float, float], list[list[str]]]] = []
    index = 0
    while index < len(rows):
        if grouped[index] is None or len(grouped[index] or []) < 3:
            index += 1
            continue
        end = index
        while end + 1 < len(rows) and len(grouped[end + 1] or []) >= 3:
            end += 1
        run = [g for g in grouped[index : end + 1] if g is not None]
        tables.extend(_table_from_run(part) for part in _aligned_parts(run))
        index = end + 1
    return tables


def _aligned_parts(run: list[list[list[Any]]]) -> list[list[list[list[Any]]]]:
    """The stretches of `run` whose rows share columns. A table sitting right under a paragraph of
    gappy prose forms one long run of "wide-gap" rows; only the table's own rows line up, so cut the
    run down to those (every window of three rows that lines up extends the current stretch)."""
    parts: list[list[list[list[Any]]]] = []
    start: int | None = None
    for i in range(len(run) - _MIN_TABLE_ROWS + 1):
        if _columns_line_up(run[i : i + _MIN_TABLE_ROWS]):
            start = i if start is None else start
            end = i + _MIN_TABLE_ROWS
            if i + 1 >= len(run) - _MIN_TABLE_ROWS + 1 or not _columns_line_up(
                run[i + 1 : i + 1 + _MIN_TABLE_ROWS]
            ):
                parts.append(run[start:end])
                start = None
    return [
        part
        for part in parts
        if len(part) >= _MIN_TABLE_ROWS and _columns_line_up(part) and not _looks_like_prose(part)
    ]


def _looks_like_prose(part: list[list[list[Any]]]) -> bool:
    """Columns of running text also line up (a newspaper-style or three-column page), but a table's
    cells are short and discrete while a text column's lines each run to five or more words. Reading
    such a page as a table would print unrelated sentence fragments side by side, so leave it as text."""
    cells = [group for groups in part for group in groups]
    long_cells = sum(1 for group in cells if len(group) >= _PROSE_CELL_WORDS)
    return long_cells / len(cells) >= _PROSE_CELL_SHARE


def _table_from_run(
    run: list[list[list[Any]]],
) -> tuple[tuple[float, float, float, float], list[list[str]]]:
    # Column positions come from whichever row split into the most columns (usually the header).
    reference = max(run, key=len)
    centers = [(group[0]["x0"] + group[-1]["x1"]) / 2 for group in reference]
    table: list[list[str]] = []
    all_words: list[Any] = []
    for groups in run:
        out_row = [""] * len(centers)
        for group in groups:
            all_words.extend(group)
            center = (group[0]["x0"] + group[-1]["x1"]) / 2
            index = min(range(len(centers)), key=lambda i: abs(center - centers[i]))
            text = " ".join(w["text"] for w in group)
            out_row[index] = f"{out_row[index]} {text}" if out_row[index] else text
        table.append(out_row)
    bbox = (
        min(w["x0"] for w in all_words),
        min(w["top"] for w in all_words),
        max(w["x1"] for w in all_words),
        max(w["bottom"] for w in all_words),
    )
    return bbox, table


def _missing_word_share(page: Any, rendered: str) -> float:
    """Share of the page's words that do not appear in `rendered` (Markdown decoration ignored)."""
    raw = Counter(w["text"] for w in page.extract_words())
    total = sum(raw.values())
    if not total:
        return 0.0
    have = Counter(re.findall(r"[^\s|]+", rendered))
    return sum((raw - have).values()) / total


class PdfLoader(_FileLoader):
    """PDFs.

    `extraction="layout"` (the default) reads tables as Markdown tables, detects headings by font
    size/weight as well as wording, and keeps two-column reading order; `"text"` is faster but
    flattens tables into running text and only catches numbered/ALL-CAPS style headings. Pages with
    no text layer are read with OCR when an OCR engine is configured; otherwise they are flagged
    (`needs_ocr`). Encrypted PDFs are read once a `password_env` naming the environment variable that
    holds the password is set (passwords are never accepted in config, like every other secret).
    """

    file_type = "pdf"

    def __init__(
        self,
        *,
        extraction: str = "layout",
        ocr: OCREngine | None = None,
        password_env: str | None = None,
    ) -> None:
        super().__init__(extraction=extraction, ocr=ocr)
        self.password_env = password_env

    def _password(self) -> str | None:
        if not self.password_env:
            return None
        value = os.environ.get(self.password_env, "").strip()
        if not value:
            raise ConfigError(
                f"password_env is set to '{self.password_env}' but that environment variable is "
                "empty or unset (PDF passwords are read from the environment, never from config)."
            )
        return value

    def _password_hint(self) -> str:
        return "" if self.password_env is None else f" (the password in {self.password_env} did not work)"

    def _text_pages(self, path: Path) -> tuple[list[str], dict[str, Any]]:
        pypdf = _import("pypdf", "pdf", "pdf")
        reader = pypdf.PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(self._password() or ""):
            raise ValueError(f"PDF is password-protected{self._password_hint()}")
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        return pages, {"title": reader.metadata.title if reader.metadata else ""}

    def _layout_pages(self, path: Path) -> tuple[list[str], dict[str, Any]]:
        pdfplumber = _import("pdfplumber", "pdf", "pdf")
        password = self._password() or ""  # outside the try: a ConfigError here is not "encrypted"
        try:
            pdf = pdfplumber.open(str(path), password=password)
        except Exception as exc:
            if _mentions_password(exc):
                raise ValueError(f"PDF is password-protected{self._password_hint()}") from exc
            raise
        with pdf:
            pages = [self._layout_page(page) for page in pdf.pages]
            title = str((pdf.metadata or {}).get("Title") or "")
        return pages, {"title": title}

    @staticmethod
    def _layout_page(page: Any) -> str:
        ruled = page.find_tables()
        if ruled:
            tables = [(t.bbox, [[cell or "" for cell in row] for row in t.extract()]) for t in ruled]
        else:
            tables = _borderless_tables(page)
        if tables:
            rendered = PdfLoader._render_page(page, tables)
            # Safety net: a table rendering must never lose the page's words. If it does (a table
            # heuristic misfired, a merged cell was dropped), keep the text and give up the structure.
            if _missing_word_share(page, rendered) <= _MAX_MISSING_WORDS:
                return rendered
        return PdfLoader._render_page(page, [])

    @staticmethod
    def _render_page(
        page: Any, tables: list[tuple[tuple[float, float, float, float], list[list[str]]]]
    ) -> str:
        boxes = [bbox for bbox, _ in tables]

        def outside_tables(obj: dict[str, Any]) -> bool:
            return not any(
                obj["x0"] >= x0 - 1
                and obj["x1"] <= x1 + 1
                and obj["top"] >= top - 1
                and obj["bottom"] <= bottom + 1
                for x0, top, x1, bottom in boxes
            )

        body = page.filter(outside_tables) if boxes else page
        lines = body.extract_text_lines()

        if tables:
            # A table anchors the page to a single reading column; keep the simple top-to-bottom
            # merge so the table's position relative to the surrounding text stays correct (a page
            # that mixes a ruled table with genuine two-column text is rare and out of scope here).
            records = [_line_record(line) for line in lines]
            body_size = _modal_size([r["size"] for r in records])
            items: list[tuple[float, str]] = [
                (r["top"], _heading_line(r, body_size)) for r in records
            ]
            for bbox, rows in tables:
                markdown = to_markdown_table(rows)
                if markdown:
                    items.append((float(bbox[1]), f"\n{markdown}\n"))
            items.sort(key=lambda item: item[0])
            return "\n".join(text for _, text in items).strip()

        mid = float(page.width) / 2
        records = [record for line in lines for record in _split_line_at_column_gap(line, mid)]
        body_size = _modal_size([r["size"] for r in records])
        ordered = _reading_order(records, float(page.width))
        return "\n".join(_heading_line(r, body_size) for r in ordered).strip()

    def _form_fields(self, path: Path) -> list[tuple[str, str]]:
        pypdf = _import("pypdf", "pdf", "pdf")
        reader = pypdf.PdfReader(str(path))
        if reader.is_encrypted and not reader.decrypt(self._password() or ""):
            return []  # already reported as an error by the main extraction path
        try:
            fields = reader.get_fields() or {}
        except Exception:
            return []
        out = []
        for name, field in fields.items():
            value = field.get("/V") if hasattr(field, "get") else None
            if value:
                out.append((str(name), str(value)))
        return out

    def _extract(self, path: Path) -> tuple[list[str], dict[str, Any]]:
        pages, meta = self._text_pages(path) if self.extraction == "text" else self._layout_pages(path)
        fields = self._form_fields(path)
        if fields:
            appendix = "\n".join(f"- **{name}:** {value}" for name, value in fields)
            if pages:
                pages = [*pages[:-1], f"{pages[-1]}\n\n## Form fields\n\n{appendix}"]
            else:
                pages = [f"## Form fields\n\n{appendix}"]
            meta["form_fields"] = len(fields)
        return pages, meta

    def _render(self, path: Path, indexes: list[int]) -> dict[int, Any]:
        pdfium = _import("pypdfium2", "pdf", "pdf")
        pdf = pdfium.PdfDocument(str(path))
        try:
            return {i: pdf[i].render(scale=RENDER_DPI / 72).to_pil() for i in indexes}
        finally:
            pdf.close()

    async def load(self, location: str) -> AsyncIterator[Document]:
        path = Path(location)
        pages, meta = await asyncio.to_thread(self._extract, path)
        pages = strip_repeated_lines(pages)
        meta["pages"] = len(pages)
        blank = [i for i, text in enumerate(pages) if len(text.strip()) < MIN_PAGE_CHARS]
        if blank and self.ocr is not None:
            images = await asyncio.to_thread(self._render, path, blank)
            recovered = 0
            for index in blank:
                text = (await self.ocr.recognize(images[index])).strip()
                if len(text) > len(pages[index].strip()):  # never replace real text with less
                    pages[index] = text
                    recovered += 1
            meta["ocr_pages"] = recovered
            blank = [i for i in blank if not pages[i].strip()]
        if blank:
            meta["needs_ocr"] = True
            meta["pages_without_text"] = len(blank)
        text = promote_headings("\f".join(pages))
        yield self._document(path, text, meta)


class ImageLoader(_FileLoader):
    """Photos and scans of documents (PNG, JPEG, TIFF, ...). Needs an OCR engine (`knowledge.ocr`)."""

    file_type = "image"

    async def load(self, location: str) -> AsyncIterator[Document]:
        path = Path(location)
        if self.ocr is None:
            raise ConfigError(
                f"{path.name}: reading images needs OCR. Set `knowledge.ocr` "
                "(e.g. {provider: rapidocr}) and pip install 'heka-rag-sdk[ocr]'."
            )
        frames = await asyncio.to_thread(self._frames, path)
        texts = [(await self.ocr.recognize(frame)).strip() for frame in frames]
        meta: dict[str, Any] = {"pages": len(frames), "ocr_pages": sum(1 for t in texts if t)}
        yield self._document(path, promote_headings("\f".join(texts)), meta)

    @staticmethod
    def _frames(path: Path) -> list[Any]:
        image_module = _import("PIL.Image", "ocr", "image")
        frames: list[Any] = []
        with image_module.open(path) as image:
            for index in range(getattr(image, "n_frames", 1)):  # multi-page TIFFs
                image.seek(index)
                frames.append(image.convert("RGB"))
        return frames


class DocxLoader(_FileLoader):
    file_type = "docx"

    def _parse(self, path: Path) -> tuple[str, dict[str, Any]]:
        docx = _import("docx", "office", "docx")
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        document = docx.Document(str(path))
        blocks: list[str] = []
        title = document.core_properties.title or ""
        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                text = item.text.strip()
                if not text:
                    continue
                style = (item.style.name if item.style is not None else "") or ""
                if style == "Title":
                    title = title or text
                    blocks.append(f"# {text}")
                elif style.startswith("Heading") and style.split()[-1].isdigit():
                    blocks.append(f"{'#' * min(6, int(style.split()[-1]))} {text}")
                elif style.startswith("List"):
                    blocks.append(f"- {text}")
                else:
                    blocks.append(text)
            elif isinstance(item, Table):
                rows = [[cell.text for cell in row.cells] for row in item.rows]
                blocks.append(to_markdown_table(rows))
        blocks.extend(self._page_furniture(document))
        return "\n\n".join(b for b in blocks if b), {"title": title}

    @staticmethod
    def _page_furniture(document: Any) -> list[str]:
        """Text in the page headers and footers (version, effective date, confidentiality...), once each
        even though every section and page type repeats it - body-only extraction would lose it."""
        found: dict[str, list[str]] = {"Header": [], "Footer": []}
        for section in document.sections:
            for label, parts in (
                ("Header", (section.header, section.first_page_header, section.even_page_header)),
                ("Footer", (section.footer, section.first_page_footer, section.even_page_footer)),
            ):
                for part in parts:
                    text = " ".join(p.text.strip() for p in part.paragraphs if p.text.strip())
                    if text and text not in found[label]:
                        found[label].append(text)
        return [f"{label}: {' | '.join(texts)}" for label, texts in found.items() if texts]


def _leaf_shapes(shapes: Any) -> Any:
    """Every shape on a slide in order, looking inside groups (diagrams and SmartArt-style layouts are
    groups; iterating only the top level skips all the text in them)."""
    for shape in shapes:
        inner = getattr(shape, "shapes", None)
        if inner is not None and getattr(shape, "shape_type", None) == 6:  # MSO_SHAPE_TYPE.GROUP
            yield from _leaf_shapes(inner)
        else:
            yield shape


class PptxLoader(_FileLoader):
    file_type = "pptx"

    def _parse(self, path: Path) -> tuple[str, dict[str, Any]]:
        pptx = _import("pptx", "office", "pptx")
        presentation = pptx.Presentation(str(path))
        blocks: list[str] = []
        for number, slide in enumerate(presentation.slides, start=1):
            title_shape = slide.shapes.title
            title = title_shape.text_frame.text.strip() if title_shape is not None else ""
            blocks.append(f"## Slide {number}: {title}" if title else f"## Slide {number}")
            for shape in _leaf_shapes(slide.shapes):
                if shape == title_shape:
                    continue
                if getattr(shape, "has_table", False) and shape.has_table:
                    rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                    blocks.append(to_markdown_table(rows))
                elif getattr(shape, "has_text_frame", False) and shape.has_text_frame:
                    text = "\n".join(
                        p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()
                    )
                    if text:
                        blocks.append(text)
            if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
                notes = slide.notes_slide.notes_text_frame.text.strip()
                if notes:
                    blocks.append(f"Speaker notes: {notes}")
        return "\n\n".join(blocks), {
            "title": presentation.core_properties.title or "",
            "slides": len(presentation.slides),
        }


class XlsxLoader(_FileLoader):
    file_type = "xlsx"

    def _parse(self, path: Path) -> tuple[str, dict[str, Any]]:
        openpyxl = _import("openpyxl", "office", "xlsx")
        workbook = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
        formulas = openpyxl.load_workbook(str(path), read_only=True, data_only=False)
        blocks: list[str] = []
        try:
            for sheet in workbook.worksheets:
                # A file written by a library (never opened in Excel) has formulas but no cached results, so
                # the value reads as empty; show the formula text rather than a blank that looks like "no data".
                raw = formulas[sheet.title].iter_rows(values_only=True)
                rows = [
                    [_cell_text(v, f) for v, f in zip(row, formula_row, strict=False)]
                    for row, formula_row in zip(sheet.iter_rows(values_only=True), raw, strict=False)
                ]
                table = to_markdown_table(rows)
                if table:
                    hidden = " (hidden)" if sheet.sheet_state != "visible" else ""
                    blocks.append(f"## Sheet: {sheet.title}{hidden}\n\n{table}")
            sheets = len(workbook.worksheets)
        finally:
            workbook.close()
            formulas.close()
        return "\n\n".join(blocks), {"sheets": sheets}


def _cell_text(value: Any, formula: Any) -> str:
    if value is not None:
        return str(value)
    if isinstance(formula, str) and formula.startswith("="):
        return formula
    return ""


class CsvLoader(_FileLoader):
    file_type = "csv"

    def _parse(self, path: Path) -> tuple[str, dict[str, Any]]:
        raw = path.read_text(encoding="utf-8-sig", errors="replace")
        try:
            dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(
                raw[:4096], delimiters=",;\t|"
            )
        except csv.Error:
            dialect = csv.excel
        rows = [list(row) for row in csv.reader(raw.splitlines(), dialect)]
        return f"## {path.stem}\n\n{to_markdown_table(rows)}", {}


_DROP_TAGS = ("script", "style", "noscript", "nav", "footer", "aside", "form", "svg", "template")
_HEADINGS = {f"h{n}": n for n in range(1, 7)}
_TEXT_BLOCKS = {"p", "pre", "blockquote", "dt", "dd", "summary", "figcaption"}
# Elements a browser lays out on their own line: text on either side is separate. Everything else
# (a, b, strong, em, i, span, sup, sub, code, ...) is inline and must not add or remove a space.
_BLOCK_TAGS = {
    "p", "div", "br", "hr", "ul", "ol", "li", "dl", "dt", "dd", "table", "thead", "tbody", "tfoot",
    "tr", "td", "th", "blockquote", "pre", "section", "article", "header", "main", "figure",
    "figcaption", "details", "summary", "address", "fieldset", "h1", "h2", "h3", "h4", "h5", "h6",
}


def _flow_text(node: Any, skip: frozenset[str] = frozenset()) -> str:
    """An element's text as a browser shows it: inline markup adds no space ("noncitizen" inside a link
    followed by ";" stays "noncitizen;"), block-level children and line breaks are separated by one.

    `get_text(" ")` puts a space around *every* element, so "eligible <a>noncitizen</a>;" became
    "eligible noncitizen ;" and "You<em>'</em>ll" became "You ' ll": the model then reads (and quotes)
    natural text that the indexed text does not contain.
    """
    parts: list[str] = []
    for child in node.children:
        name = getattr(child, "name", None)
        if name is None:
            if type(child).__name__ == "NavigableString":  # not comments, doctypes, CDATA
                parts.append(str(child))
        elif name in skip:
            continue
        elif name in _BLOCK_TAGS:
            parts.append(f" {_flow_text(child, skip)} ")
        else:
            parts.append(_flow_text(child, skip))
    return " ".join("".join(parts).split())


class HtmlLoader(_FileLoader):
    """HTML pages and wiki exports. Navigation, scripts and footers are stripped."""

    file_type = "html"

    def _parse(self, path: Path) -> tuple[str, dict[str, Any]]:
        bs4 = _import("bs4", "web", "html")
        soup = bs4.BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        for tag in soup(_DROP_TAGS):
            tag.decompose()
        blocks: list[str] = []
        self._walk(soup.body or soup, blocks)
        if not title:
            first = next((b for b in blocks if b.startswith("# ")), "")
            title = first[2:]
        return "\n\n".join(b for b in blocks if b), {"title": title}

    def _walk(self, node: Any, blocks: list[str]) -> None:
        # Text and inline elements that sit directly in a container (a bare <div>, <section>, <body>)
        # form a run of their own between its block children; without this, "<div>Fees are waived.</div>"
        # was skipped entirely because only <p>, headings, lists and tables were ever read.
        run: list[str] = []

        def flush() -> None:
            text = " ".join("".join(run).split())
            run.clear()
            if text:
                blocks.append(text)

        for child in node.children:
            name = getattr(child, "name", None)
            if name is None:
                if type(child).__name__ == "NavigableString":  # not comments or doctypes
                    run.append(str(child))
                continue
            if name not in _BLOCK_TAGS:  # inline element: part of the surrounding run
                run.append(_flow_text(child))
                continue
            flush()
            if name in _HEADINGS:
                text = _flow_text(child)
                if text:
                    blocks.append(f"{'#' * _HEADINGS[name]} {text}")
            elif name in _TEXT_BLOCKS:
                text = _flow_text(child)
                if text:
                    blocks.append(text)
            elif name == "li":
                own = _flow_text(child, skip=frozenset({"ul", "ol"}))
                if own:
                    blocks.append(f"- {own}")
                self._walk(child, blocks)
            elif name == "table":
                rows = [
                    [_flow_text(cell) for cell in row.find_all(["th", "td"])]
                    for row in child.find_all("tr")
                ]
                blocks.append(to_markdown_table(rows))
            else:
                self._walk(child, blocks)
        flush()
