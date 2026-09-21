"""Fetch a set of GOV.UK employment guides through the public GOV.UK Content and Search APIs.

Content is published under the Open Government Licence v3.0
(https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/): free to copy, adapt and use
with attribution. Keep the generated `manifest.json` (URL, title, last update, fetch time) as the
attribution record, and do not commit the downloaded pages.

    python fetch_govuk.py --out <folder> [--limit 80]

Politeness: an identifying User-Agent, at most one request per second, and a stop on HTTP 403/429.
Only the documented APIs are used (never the disallowed `/search/all` pages).
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE = "https://www.gov.uk"
USER_AGENT = "heka-rag-research/0.1 (documentation-testing; respects robots.txt, 1 request/second)"
QUERIES = [
    "employment rights",
    "employing staff",
    "redundancy",
    "statutory sick pay",
    "maternity paternity leave",
    "working hours holiday pay",
    "dismissal and disciplinary",
    "minimum wage",
    "pensions employer",
    "workplace health and safety",
]
DELAY_S = 1.0


class Blocked(Exception):
    """The site asked us to stop (403/429); do not retry."""


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data: dict[str, Any] = json.load(response)
            break
        except urllib.error.HTTPError as exc:  # the site answered: never retry a refusal
            if exc.code in (403, 429):
                raise Blocked(f"{url}: HTTP {exc.code}") from exc
            raise
        except urllib.error.URLError:  # DNS or connection trouble: back off and try again
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))
    time.sleep(DELAY_S)
    return data


def discover(limit: int) -> dict[str, str]:
    links: dict[str, str] = {}
    for query in QUERIES:
        url = f"{BASE}/api/search.json?q={urllib.parse.quote(query)}&filter_format=guide&count=20&fields=title&fields=link"
        for result in get_json(url)["results"]:
            links.setdefault(result["link"], result["title"])
    return dict(list(links.items())[:limit])


def render(content: dict[str, Any]) -> str:
    """A simple HTML page from the guide's parts (each part is a section of the guide)."""
    title = html.escape(content.get("title", ""))
    details = content.get("details", {})
    body = [f"<h1>{title}</h1>"]
    if content.get("description"):
        body.append(f"<p>{html.escape(content['description'])}</p>")
    parts = details.get("parts") or []
    if parts:
        for part in parts:
            body.append(f"<h2>{html.escape(part.get('title', ''))}</h2>\n{part.get('body', '')}")
    elif details.get("body"):
        body.append(details["body"])
    return (
        f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{title}</title></head><body>\n'
        + "\n".join(body)
        + "\n</body></html>\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=80)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)

    manifest: list[dict[str, Any]] = []
    try:
        links = discover(args.limit)
        print(f"discovered {len(links)} guides")
        for link in links:
            try:
                content = get_json(f"{BASE}/api/content{link}")
            except urllib.error.HTTPError as exc:
                print(f"  skip {link}: HTTP {exc.code}")
                continue
            page = render(content)
            if len(re.sub(r"<[^>]+>", " ", page).split()) < 80:
                print(f"  skip {link}: almost no text")
                continue
            name = re.sub(r"[^a-z0-9]+", "-", link.lower()).strip("-")[:90] + ".html"
            (out / "docs" / name).write_text(page, encoding="utf-8")
            manifest.append(
                {
                    "file": name,
                    "url": BASE + link,
                    "title": content.get("title"),
                    "public_updated_at": content.get("public_updated_at"),
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "sha256": hashlib.sha256(page.encode("utf-8")).hexdigest(),
                    "licence": "Open Government Licence v3.0",
                }
            )
    except Blocked as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"saved {len(manifest)} pages to {out / 'docs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
