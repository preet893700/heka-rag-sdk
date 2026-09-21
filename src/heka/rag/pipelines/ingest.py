"""Ingestion: discover files -> load -> chunk -> embed -> store, updating only what changed.

A manifest (`manifest.json` next to the index) records a content hash and the chunk ids for every file.
On each run new and changed files are re-processed, removed files are deleted from the index, and
unchanged files cost nothing. Changing the chunker or embedder invalidates everything, so a full rebuild
happens automatically.
"""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from heka.rag.adapters.loaders import EXTENSION_LOADERS
from heka.rag.aio import gather_limited
from heka.rag.config import SourceConfig
from heka.rag.errors import ConfigError
from heka.rag.pipelines.health import FileHealth, HealthReport, cross_document_notes, file_health
from heka.rag.registry import registry
from heka.rag.text import embedding_text, stable_hash
from heka.rag.types import Chunk, Document

if TYPE_CHECKING:
    from heka.rag.knowledge_base import KnowledgeBase

MANIFEST_VERSION = 1
_GLOB_CHARS = "*?["


class FailedFile(BaseModel):
    path: str
    error: str


class IngestReport(BaseModel):
    files_seen: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_removed: int = 0
    files_unchanged: int = 0
    failed: list[FailedFile] = Field(default_factory=list)
    chunks_added: int = 0
    chunks_removed: int = 0
    chunks_total: int = 0
    rebuilt: bool = False
    warnings: list[str] = Field(default_factory=list)
    seconds: float = 0.0

    def summary(self) -> str:
        lines = [
            f"files: {self.files_seen} seen | {self.files_new} new | {self.files_changed} changed | "
            f"{self.files_removed} removed | {self.files_unchanged} unchanged | {len(self.failed)} failed",
            f"chunks: +{self.chunks_added} / -{self.chunks_removed} -> {self.chunks_total} total"
            + ("  (full rebuild)" if self.rebuilt else ""),
            f"time: {self.seconds:.1f}s",
        ]
        lines += [f"warning: {w}" for w in self.warnings]
        lines += [f"FAILED {f.path}: {f.error}" for f in self.failed]
        return "\n".join(lines)


@dataclass
class SourceFile:
    path: Path
    rel: str  # name used in citations
    loader: str
    source: SourceConfig
    file_hash: str = field(default="")


def discover(
    sources: list[SourceConfig], *, ocr_enabled: bool = False
) -> tuple[list[SourceFile], list[str]]:
    """Expand configured sources (directory, file or glob) into loadable files.

    Image files are only loadable through OCR, so without it they are skipped (with a warning).
    """
    found: list[SourceFile] = []
    warnings: list[str] = []
    seen_rel: set[str] = set()
    claimed: dict[str, tuple[Path, SourceConfig]] = {}
    unsupported: list[str] = []
    images_without_ocr: list[str] = []

    for source in sources:
        location = source.location
        if "://" in location:
            raise ConfigError(
                f"Source {location!r}: URLs and connectors are not available yet; use a local path."
            )
        if any(ch in location for ch in _GLOB_CHARS):
            leading = Path(location).parts[: _first_glob_part(location)]
            root = Path(*leading) if leading else Path(".")
            candidates = sorted(Path(p) for p in glob.glob(location, recursive=True))
        else:
            path = Path(location)
            if path.is_dir():
                root, candidates = path, sorted(p for p in path.rglob("*"))
            elif path.is_file():
                root, candidates = path.parent, [path]
            else:
                raise ConfigError(f"Source not found: {location}")

        for path in candidates:
            if not path.is_file() or path.name.startswith(("~$", ".")):
                continue
            if any(part.startswith(".") for part in path.relative_to(root).parts[:-1]):
                continue
            loader = source.loader or EXTENSION_LOADERS.get(path.suffix.lower())
            if loader is None:
                unsupported.append(path.name)
                continue
            if loader == "image" and not ocr_enabled:
                images_without_ocr.append(path.name)
                continue
            rel = path.relative_to(root).as_posix()
            if any(fnmatch.fnmatch(rel, pattern) for pattern in source.exclude):
                continue
            resolved = str(path.resolve())
            previous = claimed.get(resolved)
            if previous is not None:
                # The same file reached through two sources. Silently indexing it twice would give
                # it two sets of tags (e.g. a public copy of a restricted document), so identical
                # settings are merged and different ones are an error.
                if _settings_of(previous[1], loader) != _settings_of(source, loader):
                    raise ConfigError(
                        f"{path} is matched by two sources with different settings "
                        f"({previous[1].location!r} and {location!r}). Make the sources disjoint, "
                        "e.g. with `exclude:` on the broader one."
                    )
                continue
            claimed[resolved] = (path, source)
            if rel in seen_rel:
                rel = f"{root.name}/{rel}"
            seen_rel.add(rel)
            found.append(SourceFile(path=path, rel=rel, loader=loader, source=source))

    if unsupported:
        sample = ", ".join(sorted(set(unsupported))[:5])
        warnings.append(
            f"skipped {len(unsupported)} file(s) with unsupported types (e.g. {sample})"
        )
    if images_without_ocr:
        sample = ", ".join(sorted(set(images_without_ocr))[:5])
        warnings.append(
            f"skipped {len(images_without_ocr)} image file(s) (e.g. {sample}): "
            "set knowledge.ocr to read them"
        )
    return found, warnings


def _settings_of(source: SourceConfig, loader: str) -> str:
    """Everything about a source that changes how (or who can see how) a file is indexed."""
    return json.dumps([loader, source.params, source.metadata], sort_keys=True, default=str)


def _first_glob_part(location: str) -> int:
    for index, part in enumerate(Path(location).parts):
        if any(ch in part for ch in _GLOB_CHARS):
            return index
    return len(Path(location).parts)


def _hash_file(item: SourceFile) -> str:
    digest = hashlib.sha256()
    with item.path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    settings = json.dumps(
        [item.loader, item.source.params, item.source.metadata], sort_keys=True, default=str
    )
    return stable_hash(digest.hexdigest(), settings)


def _fingerprint(kb: KnowledgeBase) -> str:
    config = kb.config
    embedder_id = getattr(kb.embedder, "model_id", config.embedder.provider)
    return stable_hash(
        MANIFEST_VERSION,
        json.dumps(config.chunker.model_dump(), sort_keys=True, default=str),
        embedder_id,
        config.knowledge.extraction,
        json.dumps(
            config.knowledge.ocr.model_dump() if config.knowledge.ocr else None, sort_keys=True
        ),
    )


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if data.get("version") == MANIFEST_VERSION else {}


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    os.replace(temp, path)


@dataclass
class _Processed:
    item: SourceFile
    chunks: list[Chunk] = field(default_factory=list)
    error: str | None = None
    notes: list[str] = field(default_factory=list)
    health: FileHealth | None = None  # only when asked for (see analyze_documents)
    text: str = ""


def _merged_metadata(parts: list[dict[str, Any]]) -> dict[str, Any]:
    """Loader metadata for a file that produced several documents: page counts add up."""
    merged: dict[str, Any] = {}
    for meta in parts:
        for key in ("pages", "pages_without_text", "ocr_pages"):
            if isinstance(meta.get(key), int):
                merged[key] = merged.get(key, 0) + meta[key]
    return merged


async def _process(kb: KnowledgeBase, item: SourceFile, *, health: bool = False) -> _Processed:
    """Load and chunk one file. Errors are captured so one bad file never aborts the run."""
    result = _Processed(item)
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    try:
        loader = registry.create(
            "loader",
            item.loader,
            item.source.params,
            deps={"extraction": kb.config.knowledge.extraction, "ocr": kb.ocr},
        )
        async for raw in loader.load(str(item.path)):
            document = Document(
                id=stable_hash(item.rel, length=16),
                source=item.rel,
                text=raw.text,
                metadata={**raw.metadata, **item.source.metadata},
            )
            missing = kb.access.missing_fields(document.metadata)
            if missing:
                result.notes.append(
                    f"{item.rel}: no {', '.join(missing)} metadata, so under the access policy "
                    "nobody can see it. Set it in knowledge.sources[].metadata "
                    f"(use '{kb.config.access.shared_value}' for everyone)"
                )
            if raw.metadata.get("ocr_pages"):
                result.notes.append(
                    f"{item.rel}: {raw.metadata['ocr_pages']} scanned page(s) were read with OCR; "
                    "OCR can misread characters, so spot-check the answers that cite it"
                )
            if raw.metadata.get("needs_ocr"):
                hint = (
                    "OCR found no text on them"
                    if kb.ocr is not None
                    else "set knowledge.ocr (pip install 'heka-rag-sdk[ocr]') to read them"
                )
                result.notes.append(
                    f"{item.rel}: {raw.metadata.get('pages_without_text')} page(s) have no text "
                    f"layer (scanned); {hint}"
                )
            result.chunks.extend(await kb.chunker.chunk(document))
            if health:
                texts.append(raw.text)
                metas.append(dict(raw.metadata))
    except Exception as exc:
        result.chunks, result.error = [], f"{type(exc).__name__}: {exc}"
    if health:
        if result.error is not None:
            result.health = FileHealth(path=item.rel, loader=item.loader, error=result.error)
        else:
            result.text = "\n\n".join(texts)
            result.health = file_health(
                item.rel,
                item.loader,
                result.text,
                _merged_metadata(metas),
                [len(c.text) for c in result.chunks],
                ocr_configured=kb.ocr is not None,
            )
    return result


async def analyze_documents(kb: KnowledgeBase) -> HealthReport:
    """Load and chunk every configured file exactly as ingestion would, and report how each fared.

    Nothing is embedded or written to the index, so this needs no model and no API key.
    """
    files, skipped = discover(kb.config.knowledge.sources, ocr_enabled=kb.ocr is not None)
    results = await gather_limited(files, lambda item: _process(kb, item, health=True), limit=4)
    report = HealthReport(files=[r.health for r in results if r.health is not None])
    report.skipped = skipped
    report.cross_document = cross_document_notes((r.item.rel, r.text) for r in results if r.text)
    return report


async def run_ingest(kb: KnowledgeBase) -> IngestReport:
    started = time.perf_counter()
    config = kb.config
    report = IngestReport()

    files, report.warnings = discover(config.knowledge.sources, ocr_enabled=kb.ocr is not None)
    report.files_seen = len(files)
    for item in files:
        item.file_hash = _hash_file(item)

    manifest_path = kb.dir / "manifest.json"
    manifest = _read_manifest(manifest_path)
    fingerprint = _fingerprint(kb)
    indexed = await kb.store.count()
    rebuild = (
        config.knowledge.update_mode == "rebuild"
        or manifest.get("fingerprint") != fingerprint
        or (bool(manifest.get("files")) and indexed == 0)
    )
    known: dict[str, dict[str, Any]] = {} if rebuild else dict(manifest.get("files", {}))
    if rebuild and (manifest or indexed):
        await kb.store.clear()
        report.rebuilt = True
        report.chunks_removed += indexed

    current = {item.rel: item for item in files}
    removed = [rel for rel in known if rel not in current]
    to_process = [item for item in files if known.get(item.rel, {}).get("hash") != item.file_hash]
    report.files_removed = len(removed)
    report.files_unchanged = len(files) - len(to_process)
    report.files_new = sum(1 for item in to_process if item.rel not in known)
    report.files_changed = len(to_process) - report.files_new

    results = await gather_limited(to_process, lambda item: _process(kb, item), limit=4)

    stale_ids: list[str] = [cid for rel in removed for cid in known[rel]["chunk_ids"]]
    new_chunks: list[Chunk] = []
    files_state = {rel: entry for rel, entry in known.items() if rel not in removed}
    for result in results:
        item = result.item
        if result.error is not None:
            report.failed.append(FailedFile(path=item.rel, error=result.error))
            continue
        if item.rel in known:
            stale_ids.extend(known[item.rel]["chunk_ids"])
        files_state[item.rel] = {"hash": item.file_hash, "chunk_ids": [c.id for c in result.chunks]}
        new_chunks.extend(result.chunks)
        report.warnings.extend(result.notes)
        if not result.chunks:
            report.warnings.append(f"{item.rel}: no text could be extracted")

    if stale_ids:
        report.chunks_removed += await kb.store.delete(ids=stale_ids)
    if new_chunks:
        vectors = await kb.embedder.embed_documents([embedding_text(c) for c in new_chunks])
        await kb.store.upsert(new_chunks, vectors)
    report.chunks_added = len(new_chunks)
    report.chunks_total = await kb.store.count()

    _write_manifest(
        manifest_path,
        {"version": MANIFEST_VERSION, "fingerprint": fingerprint, "files": files_state},
    )
    report.seconds = time.perf_counter() - started
    return report
