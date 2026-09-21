"""Fetch a set of IRS publications and form instructions (real, long, table-heavy PDFs).

These are works of the US federal government (17 U.S.C. 105: no copyright in the US); they may still
embed third-party material, so keep them for local testing and do not redistribute or commit them.
Keep `manifest.json` (URL, size, hash, fetch time) as the provenance record.

    python fetch_irs.py --out <folder>

Direct file URLs under /pub/irs-pdf/ are fetched; the IRS search and listing pages, which its robots.txt
disallows, are never touched. One request per second, an identifying User-Agent, a per-file and a total
size cap, and a stop on HTTP 403/429.
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

BASE = "https://www.irs.gov/pub/irs-pdf/"
USER_AGENT = "kbsdk-research/0.1 (documentation-testing; respects robots.txt, 1 request/second)"
FILES = [
    # publications
    "p15",
    "p15a",
    "p15b",
    "p17",
    "p54",
    "p225",
    "p334",
    "p463",
    "p501",
    "p502",
    "p503",
    "p504",
    "p505",
    "p510",
    "p515",
    "p516",
    "p519",
    "p521",
    "p523",
    "p524",
    "p525",
    "p526",
    "p527",
    "p529",
    "p530",
    "p535",
    "p536",
    "p537",
    "p541",
    "p544",
    "p547",
    "p550",
    "p551",
    "p554",
    "p555",
    "p559",
    "p575",
    "p590a",
    "p590b",
    "p596",
    "p598",
    "p925",
    "p926",
    "p936",
    "p946",
    "p970",
    "p972",
    "p978",
    # form instructions
    "i1040gi",
    "i1040sc",
    "i1040sd",
    "i1040se",
    "i1040sse",
    "i8829",
    "i8949",
    "i941",
    "i1099gi",
    "iw4",
]
MAX_FILE_BYTES = 9_000_000
MAX_TOTAL_BYTES = 220_000_000
DELAY_S = 1.0


def request(url: str, method: str) -> urllib.request.addinfourl:
    req = urllib.request.Request(url, method=method, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            return urllib.request.urlopen(req, timeout=60)  # type: ignore[no-any-return]
        except urllib.error.HTTPError as exc:
            if exc.code in (403, 429):
                raise SystemExit(f"stopped: {url} answered HTTP {exc.code}") from exc
            raise
        except urllib.error.URLError:
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)

    manifest, total = [], 0
    try:
        for stem in FILES:
            url = f"{BASE}{stem}.pdf"
            try:
                with request(url, "HEAD") as head:
                    size = int(head.headers.get("Content-Length") or 0)
                    kind = head.headers.get("Content-Type", "")
            except urllib.error.HTTPError as exc:
                print(f"  skip {stem}: HTTP {exc.code}")
                time.sleep(DELAY_S)
                continue
            time.sleep(DELAY_S)
            if (
                "pdf" not in kind.lower()
                or size == 0
                or size > MAX_FILE_BYTES
                or total + size > MAX_TOTAL_BYTES
            ):
                print(f"  skip {stem}: type={kind!r} size={size}")
                continue
            with request(url, "GET") as response:
                data = response.read()
            time.sleep(DELAY_S)
            if not data.startswith(b"%PDF"):
                print(f"  skip {stem}: not a PDF")
                continue
            (out / "docs" / f"{stem}.pdf").write_bytes(data)
            total += len(data)
            manifest.append(
                {
                    "file": f"{stem}.pdf",
                    "url": url,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "licence": "US federal government work (public domain in the US)",
                }
            )
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"saved {len(manifest)} PDFs, {total / 1e6:.0f} MB, to {out / 'docs'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
