"""Small text helpers: hashing, slugs, quote verification, JSON extraction."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from heka.rag.types import Chunk

_PUNCTUATION_FOLD = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
    }
)
_MARKDOWN_NOISE = re.compile(r"[*_`|#>]")
_ELLIPSIS = re.compile(r"\.\.\.|…")
_MIN_QUOTE_CHARS = 4


def stable_hash(*parts: object, length: int | None = None) -> str:
    digest = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return digest[:length] if length else digest


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "agent"


def normalize_ws(value: str) -> str:
    return " ".join(value.split())


def _canonical(value: str, *, strip_markdown: bool) -> str:
    value = value.translate(_PUNCTUATION_FOLD)
    if strip_markdown:
        value = _MARKDOWN_NOISE.sub(" ", value)
    return normalize_ws(value).casefold()


def _found_in_order(parts: list[str], haystack: str) -> bool:
    position = 0
    for part in parts:
        index = haystack.find(part, position)
        if index < 0:
            return False
        position = index + len(part)
    return True


def quote_in_text(quote: str, text: str) -> bool:
    """True if `quote` appears verbatim in `text`.

    Whitespace, letter case, curly-vs-straight punctuation and markdown decoration are ignored, and
    a quote may use "..." to skip material (each fragment must still appear, in order). This is what
    makes a citation *verified*: the model cannot cite words the source does not contain.
    """
    for strip_markdown in (False, True):
        haystack = _canonical(text, strip_markdown=strip_markdown)
        parts = [
            p
            for p in (
                _canonical(fragment, strip_markdown=strip_markdown).strip(" .,;:")
                for fragment in _ELLIPSIS.split(quote)
            )
            if p
        ]
        if (
            parts
            and sum(len(p) for p in parts) >= _MIN_QUOTE_CHARS
            and _found_in_order(parts, haystack)
        ):
            return True
    return False


def embedding_text(chunk: Chunk) -> str:
    """What gets embedded: the chunk text prefixed with its document and heading context."""
    context = chunk.metadata.get("context")
    return f"{context}\n\n{chunk.text}" if context else chunk.text


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model reply (tolerates code fences and chatter)."""
    candidate = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None
