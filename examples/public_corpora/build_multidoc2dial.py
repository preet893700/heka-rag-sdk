"""Turn the public MultiDoc2Dial dataset into a kbsdk test bed: documents plus human-written questions.

MultiDoc2Dial (Feng et al., EMNLP 2021, CC BY 3.0) holds real US-government web documents (SSA, VA, DMV,
Student Aid) and dialogues in which people ask about them. Each answerable user question is labelled
with the document spans that answer it, and the next human agent turn is a written reference answer.
Cite the paper if you use the data:

    Feng, Patel, Wan, Joshi. MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents. EMNLP 2021.

    python build_multidoc2dial.py --src <folder with multidoc2dial_*.json> --out <target folder>

Per domain it writes `<out>/<domain>/docs/*.html` (the raw HTML, so the HTML loader and chunker are
exercised) and `<out>/<domain>/questions.jsonl`.

Design decisions (so the numbers can be trusted):
* Only `query_solution` user turns whose next agent turn is `respond_solution` with grounding spans are
  used. Every such question has earlier turns (the dialogue opens with a general request), so each case
  carries its history: this is a conversational test.
* "dev" cases come from the dataset's train file and "test" cases from its validation file (the official
  test file is a dummy). Tune on dev; look at test once.
* The dataset's "no relevant information" turns mean "not answerable from the dialogue's *own* document",
  which is not the same as "not in the corpus", so they are NOT used as unanswerable questions.
* A gold source is a passage of the text our HTML loader extracts (found by matching the dataset span's
  words, since the dataset text is tokenized), cut to its first characters so a chunk boundary inside a
  long span does not count as a miss.
* The corpus holds exact copies of some pages; one copy of each is kept, and any other page containing the
  same passage is added as an alternate gold source. `retrieval_recall` is therefore diluted on this test
  bed; use `retrieval_hit` and `retrieval_mrr`.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from kbsdk import registry

MIN_QUESTION_WORDS = 3
QUOTE_CHARS = 160
MIN_ALTERNATE_CHARS = 50
MAX_ALTERNATES = 3
HISTORY_TURNS = 6


def slug(text: str, limit: int = 70) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "doc"


def file_name(domain: str, doc_id: str, title: str) -> str:
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()[:6]
    return f"{domain}__{slug(title)}__{digest}.html"


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def html_page(title: str, body: str) -> str:
    safe = title.replace("<", "&lt;")
    return f'<!DOCTYPE html>\n<html><head><meta charset="utf-8"><title>{safe}</title></head><body>\n{body}\n</body></html>\n'


def clean(text: str) -> str:
    return " ".join(text.split())


_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text.lower())]


def anchor(span_text: str, doc_tokens: list[tuple[str, int, int]], doc_text: str) -> str | None:
    """The passage of the *extracted* document that best matches a dataset span.

    The dataset's text is tokenized ("you ll", "non - driver", dropped parentheses), so it rarely matches
    what an HTML loader extracts character for character. Comparing word tokens only, take the longest
    run of words the two share and quote the real text for it. None when too little of the span matches.
    """
    wanted = [t[0] for t in tokens(span_text)]
    if not wanted:
        return None
    matcher = difflib.SequenceMatcher(None, wanted, [t[0] for t in doc_tokens], autojunk=False)
    match = matcher.find_longest_match(0, len(wanted), 0, len(doc_tokens))
    if match.size < min(len(wanted), 4) or match.size / len(wanted) < 0.5:
        return None
    start = doc_tokens[match.b][1]
    end = doc_tokens[match.b + match.size - 1][2]
    quote = clean(doc_text[start:end])
    if len(quote) > QUOTE_CHARS:
        quote = quote[:QUOTE_CHARS].rsplit(" ", 1)[0]
    return quote if len(quote) >= 12 else None


async def extract_texts(docs_dir: Path) -> dict[str, str]:
    """What the SDK's own HTML loader reads from each file (the text the gold quotes must exist in)."""
    loader = registry.create("loader", "html", {})
    texts: dict[str, str] = {}
    for path in sorted(docs_dir.glob("*.html")):
        parts = [doc.text async for doc in loader.load(str(path))]
        texts[path.name] = "\n\n".join(parts)
    return texts


def cases_from(
    dialogues: list[dict[str, Any]],
    docs: dict[str, dict[str, Any]],
    names: dict[str, str],
    split: str,
    extracted: dict[str, str],
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    doc_tokens = {name: tokens(text) for name, text in extracted.items()}
    flat = {name: clean(text).casefold() for name, text in extracted.items()}
    cases: list[dict[str, Any]] = []
    for dialogue in dialogues:
        turns = dialogue["turns"]
        for i, turn in enumerate(turns):
            if turn["role"] != "user" or turn["da"] != "query_solution" or not turn["references"]:
                continue
            reply = turns[i + 1] if i + 1 < len(turns) else None
            if not reply or reply["role"] != "agent" or reply["da"] != "respond_solution":
                continue
            question = clean(turn["utterance"])
            answer = clean(reply["utterance"])
            if len(question.split()) < MIN_QUESTION_WORDS or not answer:
                continue
            sources, seen = [], set()
            for ref in reply["references"]:
                doc = docs.get(ref["doc_id"])
                span = doc["spans"].get(str(ref["id_sp"])) if doc else None
                if span is None:
                    continue
                name = names[ref["doc_id"]]
                stats["spans"] += 1
                quote = anchor(span["text_sp"], doc_tokens[name], extracted[name])
                if quote is None:
                    stats["spans_unaligned"] += 1
                    continue
                key = (name, quote)
                if key in seen:
                    continue
                seen.add(key)
                source: dict[str, str] = {"source": name, "quote": quote}
                if span.get("title"):
                    source["section"] = clean(span["title"])[:120]
                sources.append(source)
                # Other documents that contain the same passage (near-duplicate pages) answer it just
                # as well; retrieving one of them must not count as a miss.
                # Only for specific passages: a short generic phrase ("you can apply for") occurs in
                # many pages and accepting them all would make retrieval look better than it is.
                needle = clean(quote).casefold()
                alternates = 0
                for other, haystack in flat.items():
                    if len(needle) < MIN_ALTERNATE_CHARS or alternates >= MAX_ALTERNATES:
                        break
                    if other != name and needle in haystack and (other, quote) not in seen:
                        seen.add((other, quote))
                        alternate = dict(source)
                        alternate["source"] = other
                        sources.append(alternate)
                        alternates += 1
                        stats["alternate_sources"] += 1
            if not sources:
                stats["questions_dropped_unaligned"] += 1
                continue
            stats["questions"] += 1
            history = [
                {
                    "role": "user" if t["role"] == "user" else "assistant",
                    "content": clean(t["utterance"]),
                }
                for t in turns[max(0, i - HISTORY_TURNS) : i]
                if clean(t["utterance"])
            ]
            cases.append(
                {
                    "id": f"md2d-{split}-{dialogue['dial_id'][:8]}-{turn['turn_id']}",
                    "question": question,
                    "gold_answer": answer,
                    "gold_sources": sources,
                    "answerable": True,
                    "split": split,
                    "tags": [
                        "followup" if history else "first-turn",
                        "multi-span" if len(sources) > 1 else "single-span",
                    ],
                    "history": history,
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--src", required=True, help="folder with multidoc2dial_doc.json and the dialogue files"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--domains", nargs="*", default=["ssa", "va", "dmv", "studentaid"])
    parser.add_argument(
        "--dev-n", type=int, default=150, help="dev questions per domain (from the train file)"
    )
    parser.add_argument(
        "--test-n",
        type=int,
        default=250,
        help="test questions per domain (from the validation file)",
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    src, out = Path(args.src), Path(args.out)
    doc_data = load(src / "multidoc2dial_doc.json")["doc_data"]
    train = load(src / "multidoc2dial_dial_train.json")["dial_data"]
    valid = load(src / "multidoc2dial_dial_validation.json")["dial_data"]

    for domain in args.domains:
        docs = doc_data[domain]
        names = {doc_id: file_name(domain, doc_id, d["title"]) for doc_id, d in docs.items()}
        target = out / domain / "docs"
        target.mkdir(parents=True, exist_ok=True)
        for doc_id, d in docs.items():
            (target / names[doc_id]).write_text(
                html_page(d["title"], d["doc_html_raw"]), encoding="utf-8"
            )

        extracted = asyncio.run(extract_texts(target))

        # The corpus contains byte-for-byte copies of the same page. Index one of each (what
        # `kbsdk ingest --report` advises) and point every gold label at that one.
        canonical: dict[str, str] = {}
        first_seen: dict[str, str] = {}
        for name in sorted(extracted):
            fingerprint = hashlib.sha1(
                clean(extracted[name]).casefold().encode("utf-8")
            ).hexdigest()
            canonical[name] = first_seen.setdefault(fingerprint, name)
        for name, keep in canonical.items():
            if name != keep:
                (target / name).unlink()
                del extracted[name]
        names = {doc_id: canonical[name] for doc_id, name in names.items()}
        print(f"{domain}: {len(canonical)} pages, {len(extracted)} after removing exact duplicates")
        rng = random.Random(f"{args.seed}-{domain}")
        chosen: list[dict[str, Any]] = []
        stats: dict[str, int] = defaultdict(int)
        for split, dialogues, limit in (
            ("dev", train[domain], args.dev_n),
            ("test", valid[domain], args.test_n),
        ):
            cases = cases_from(dialogues, docs, names, split, extracted, stats)
            rng.shuffle(cases)
            chosen += sorted(cases[:limit], key=lambda c: c["id"])
        print(
            f"  alignment: {stats['spans'] - stats['spans_unaligned']}/{stats['spans']} spans anchored; "
            f"{stats['questions_dropped_unaligned']} of "
            f"{stats['questions'] + stats['questions_dropped_unaligned']} candidate questions dropped "
            f"because none of their spans could be anchored; {stats['alternate_sources']} alternate "
            "gold sources added for near-duplicate pages"
        )
        with (out / domain / "questions.jsonl").open("w", encoding="utf-8") as handle:
            for case in chosen:
                handle.write(json.dumps(case, ensure_ascii=False) + "\n")
        per_split = defaultdict(int)
        for case in chosen:
            per_split[case["split"]] += 1
        print(f"{domain}: {len(docs)} documents, questions {dict(per_split)}")


if __name__ == "__main__":
    main()
