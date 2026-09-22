"""Build the PDF resolver's "torture-test" corpus: a mix of real, public-domain PDFs chosen for
specific failure modes, plus a handful of small PDFs it is impractical to reliably find in the wild
(a password-protected file with a known password, a filled AcroForm) and so are constructed by hand.

    python build_pdf_torture_corpus.py --out <folder>

Real files (fetched politely: an identifying User-Agent, one request per second, a stop on HTTP
403/429, direct file URLs only - never a disallowed search/listing page):
  - `federal-register-2024-01-02-excerpt.pdf`: the first 15 pages of a real Federal Register daily
    issue from govinfo.gov (a work of the US federal government, 17 U.S.C. 105: no copyright in the
    US). Federal Register issues are laid out in two columns throughout - a genuine multi-column
    document, not a constructed one.
  - `irs-p15-2026-repeated-headers-excerpt.pdf`: the first 10 pages of IRS Publication 15, fetched
    fresh from irs.gov (also a US federal government work). Every page repeats an identical running
    header/footer ("Page N of 59 ... 12:33 - 15-Dec-2025") - the literal real-world case that
    motivated `strip_repeated_lines`.

Constructed files (built here, not downloaded - see the module docstring above for why):
  - `borderless-roster.pdf`: a whitespace-aligned table with no ruled lines, the case
    `find_tables()`'s default ruled-line strategy misses and the `text`-strategy fallback recovers.
  - `locked-memo.pdf`: encrypted with the password in `TORTURE_PDF_PASSWORD` below (pypdf's own
    writer; needs no invented cryptography).
  - `benefits-application.pdf`: a one-page PDF with a filled AcroForm text field
    (`TORTURE_FORM_FIELD`/`TORTURE_FORM_VALUE` below).

`manifest.json` records, per file, where it came from: a source URL + sha256 of the original download
for the real files, or "constructed" + what it demonstrates for the built ones. These are all public
findings (fetch results, chosen test-relevant properties, generated fixture provenance) - already
either public government data or content this script itself produced - not private information, so
this manifest is safe to keep, and the whole `docs/` folder is git-ignored like the rest of `data/`
regardless.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

USER_AGENT = "heka-rag-research/0.1 (documentation-testing; respects robots.txt, 1 request/second)"
DELAY_S = 1.0

# Shared with tests/integration/test_pdf_torture_corpus.py - keep both in sync if these change.
TORTURE_PDF_PASSWORD = "torture-test-2026"
TORTURE_FORM_FIELD = "ApplicantName"
TORTURE_FORM_VALUE = "Jordan Alvarez"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.read()  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise SystemExit(f"stopped: {url} answered HTTP {exc.code}") from exc
            raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def _trim(data: bytes, pages: int) -> bytes:
    import io

    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(data))
    writer = pypdf.PdfWriter()
    for page in reader.pages[:pages]:
        writer.add_page(page)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _fetch_real(docs: Path, manifest: list[dict[str, Any]]) -> None:
    fr_url = "https://www.govinfo.gov/content/pkg/FR-2024-01-02/pdf/FR-2024-01-02.pdf"
    print(f"fetching {fr_url} ...")
    full = _fetch(fr_url)
    time.sleep(DELAY_S)
    excerpt = _trim(full, 15)
    name = "federal-register-2024-01-02-excerpt.pdf"
    (docs / name).write_bytes(excerpt)
    manifest.append(
        {
            "file": name,
            "source_url": fr_url,
            "original_bytes": len(full),
            "original_sha256": hashlib.sha256(full).hexdigest(),
            "kept_pages": "1-15 of the full issue (trimmed locally to keep the corpus small)",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "licence": "US federal government work (public domain in the US)",
            "demonstrates": "genuine two-column layout throughout",
        }
    )
    print(f"  saved {name} ({len(excerpt):,} bytes, trimmed from {len(full):,})")

    irs_url = "https://www.irs.gov/pub/irs-pdf/p15.pdf"
    print(f"fetching {irs_url} ...")
    full = _fetch(irs_url)
    time.sleep(DELAY_S)
    excerpt = _trim(full, 10)
    name = "irs-p15-2026-repeated-headers-excerpt.pdf"
    (docs / name).write_bytes(excerpt)
    manifest.append(
        {
            "file": name,
            "source_url": irs_url,
            "original_bytes": len(full),
            "original_sha256": hashlib.sha256(full).hexdigest(),
            "kept_pages": "1-10 of Publication 15 (trimmed locally to keep the corpus small)",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "licence": "US federal government work (public domain in the US)",
            "demonstrates": "an identical running header/footer repeated on every page",
        }
    )
    print(f"  saved {name} ({len(excerpt):,} bytes, trimmed from {len(full):,})")


def _pdf_from_streams(streams: list[bytes]) -> bytes:
    """Minimal hand-built single/multi-page PDF (Helvetica as /F1). Mirrors the same tiny PDF
    writer `tests/pdf_helpers.py` uses for unit tests, kept separate since this script must run
    standalone (it is not part of the installed package or the test suite)."""

    def assemble(objects: list[bytes]) -> bytes:
        out = b"%PDF-1.4\n"
        offsets = []
        for index, obj in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
        out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
        out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        return out

    n = len(streams)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for content in streams:
        # The content stream is appended right after this page object, so its 1-based object
        # number is two past the current length (one for the page object about to be appended,
        # one more for the stream after it) - not one past, which would self-reference the page.
        objects.append(
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Contents %d 0 R /Resources << /Font << /F1 3 0 R >> >> >>" % (len(objects) + 2)
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    return assemble(objects)


def _build_borderless_table(docs: Path) -> None:
    rows = [
        ("Unit", "Monthly Rent", "Occupant"),
        ("101", "1450", "Vacant"),
        ("102", "1525", "M. Alvarez"),
        ("201", "1610", "Vacant"),
        ("202", "1590", "R. Nakamura"),
    ]
    ops, y = [], 720
    x_positions = (72, 220, 380)
    for row in rows:
        for cell, x in zip(row, x_positions, strict=False):
            ops.append(f"BT /F1 11 Tf {x} {y} Td ({cell}) Tj ET")
        y -= 20
    (docs / "borderless-roster.pdf").write_bytes(_pdf_from_streams(["\n".join(ops).encode()]))
    print("  built borderless-roster.pdf (whitespace-aligned table, no ruled lines)")


def _build_password_protected(docs: Path) -> None:
    import io

    import pypdf

    content = (
        b"BT /F1 12 Tf 72 720 Td (Internal memo: renewal terms are confidential until signed.) "
        b"Tj ET"
    )
    reader = pypdf.PdfReader(io.BytesIO(_pdf_from_streams([content])))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password=TORTURE_PDF_PASSWORD)
    buf = io.BytesIO()
    writer.write(buf)
    (docs / "locked-memo.pdf").write_bytes(buf.getvalue())
    print(f"  built locked-memo.pdf (password: {TORTURE_PDF_PASSWORD!r})")


def _build_acroform(docs: Path) -> None:
    content = b"BT /F1 12 Tf 72 720 Td (Benefits enrollment application) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /AcroForm 6 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R "
        b"/Resources << /Font << /F1 3 0 R >> >> /Annots [7 0 R] >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Fields [7 0 R] >>",
        f"<< /Type /Annot /Subtype /Widget /FT /Tx /T ({TORTURE_FORM_FIELD}) "
        f"/V ({TORTURE_FORM_VALUE}) /Rect [100 700 300 720] /P 4 0 R >>".encode(),
    ]

    def assemble(objs: list[bytes]) -> bytes:
        out = b"%PDF-1.4\n"
        offsets = []
        for index, obj in enumerate(objs, start=1):
            offsets.append(len(out))
            out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
        xref = len(out)
        out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
        out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
        out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
        return out

    (docs / "benefits-application.pdf").write_bytes(assemble(objects))
    print(f"  built benefits-application.pdf (field {TORTURE_FORM_FIELD!r}={TORTURE_FORM_VALUE!r})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--skip-fetch",
        action="store_true",
        help="only (re)build the constructed files; skip the real, downloaded ones",
    )
    args = parser.parse_args()
    out = Path(args.out)
    docs = out / "docs"
    docs.mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    if not args.skip_fetch:
        _fetch_real(docs, manifest)
    elif (out / "manifest.json").exists():
        # Keep whatever entries a previous full run already recorded (e.g. the real, fetched
        # files) instead of silently dropping them just because this run only rebuilds the
        # constructed ones.
        previous = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        manifest.extend(entry for entry in previous if entry.get("source") != "constructed")
    for name, entry in (
        ("borderless-roster.pdf", "a whitespace-aligned table with no ruled lines"),
        ("locked-memo.pdf", f"encrypted with a known password ({TORTURE_PDF_PASSWORD!r})"),
        ("benefits-application.pdf", "a filled AcroForm text field"),
    ):
        manifest.append({"file": name, "source": "constructed", "demonstrates": entry})
    _build_borderless_table(docs)
    _build_password_protected(docs)
    _build_acroform(docs)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"\nwrote manifest.json ({len(manifest)} files) to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
