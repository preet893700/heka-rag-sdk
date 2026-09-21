"""How does accuracy change with the number of documents? A study on the MultiDoc2Dial test bed.

    python scaling_study.py retrieval --root <md2d folder> --out <work folder>
    python scaling_study.py e2e       --root <md2d folder> --out <work folder>

`retrieval`: for each real domain (coherent government topic area) draw random subsets of N documents
(default 12 and 45, plus the whole domain), keep the questions whose answer passages lie in the subset, and
score retrieval. Two configurations are fixed in advance, not tuned on these results:
  A  the shipped defaults (dense search, k=5, no follow-up handling);
  B  A plus LLM follow-up condensation (chosen because the earlier synthetic benchmark showed it was the only
     lever that helped; needs a model key).
`e2e`: a sample of those questions is answered end to end with B and graded by a *different* model than the
one answering, and the same questions are also asked with NO documents (a control: how much of the
"correctness" is just the model already knowing these public topics?).

Read the output with these limits in mind: subsets are random documents from one domain, so with few
documents the distractors are fewer than in a coherent real collection; questions come from a public
dataset an LLM may have seen; dev and test questions are pooled because nothing is tuned here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import statistics
from pathlib import Path
from typing import Any

from heka.rag import Agent, KnowledgeBase, RAGConfig, factory
from heka.rag.eval import EvalCase, LLMJudge, load_dataset, run_retrieval_eval
from heka.rag.eval.analysis import wilson_interval
from heka.rag.types import Answer, Message

ANSWER_LLM = {
    "provider": "groq",
    "params": {"model": "openai/gpt-oss-120b", "max_output_tokens": 4096},
}
# A different model family from the answerer (OpenAI-family gpt-oss answers, Alibaba's Qwen grades), so the
# grader is not marking its own homework. Grading prompts carry no retrieved text, so they fit the free
# tier's 8,000 tokens/minute. (Gemini 2.5 Flash was tried first: its free quota is 20 requests/day.)
JUDGE_LLM = {
    "provider": "groq",
    "params": {"model": "qwen/qwen3.8-27b", "max_output_tokens": 4096},
}
DOMAINS = ["studentaid", "va"]


def config_for(sub: Path, name: str, *, condense: bool) -> RAGConfig:
    return RAGConfig.preset(
        "free-tier-dev",
        name=name,
        knowledge={"sources": [{"location": str(sub / "docs")}], "persist_dir": str(sub / "index")},
        retrieval={"mode": "dense", "top_k": 20, "final_k": 5, "condense_followups": condense},
        generation={"llm": ANSWER_LLM},
        reliability={"max_retries": 6, "requests_per_minute": 9},  # under both free tiers' limits
        evaluation={"judge_llm": JUDGE_LLM},
    )


def make_subset(
    root: Path, domain: str, n: int | None, seed: int, out: Path
) -> tuple[Path, list[dict[str, Any]]]:
    docs = sorted((root / domain / "docs").glob("*.html"))
    size = len(docs) if n is None else min(n, len(docs))
    chosen = docs if n is None else random.Random(f"{domain}-{n}-{seed}").sample(docs, size)
    label = f"{domain}-{'all' if n is None else n}-s{seed}"
    sub = out / label
    if sub.exists():
        shutil.rmtree(sub)
    (sub / "docs").mkdir(parents=True)
    names = {p.name for p in chosen}
    for path in chosen:
        shutil.copy(path, sub / "docs" / path.name)
    questions: list[dict[str, Any]] = []
    for line in (root / domain / "questions.jsonl").read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        kept = [s for s in case["gold_sources"] if s["source"] in names]
        if kept:  # answerable from this subset
            case["gold_sources"] = kept
            case["tags"] = [*case["tags"], domain]
            questions.append(case)
    (sub / "questions.jsonl").write_text(
        "\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n", encoding="utf-8"
    )
    return sub, questions


async def score(sub: Path, condense: bool) -> dict[str, dict[str, float]]:
    config = config_for(sub, "study", condense=condense)  # same name: both variants share one index
    kb = KnowledgeBase(config)
    await kb.aingest()
    report = await run_retrieval_eval(kb, load_dataset(sub / "questions.jsonl"), k=5)
    return {r.case_id: r.metrics for r in report.results}


def summarize(rows: list[dict[str, float]]) -> str:
    if not rows:
        return "-"
    hits = [r["retrieval_hit"] for r in rows]
    mrr = statistics.fmean(r["retrieval_mrr"] for r in rows)
    low, high = wilson_interval(sum(hits), len(hits))
    return f"hit {sum(hits) / len(hits):.2f} [{low:.2f}-{high:.2f}]  mrr {mrr:.2f}  (n={len(hits)})"


async def retrieval(args: argparse.Namespace) -> None:
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for n in [*args.sizes, None]:
        for domain in DOMAINS:
            for seed in range(args.seeds if n is not None else 1):
                sub, questions = make_subset(root, domain, n, seed, out)
                label = sub.name
                a = await score(sub, condense=False)
                entry: dict[str, Any] = {"n_questions": len(questions), "A": a}
                if (
                    args.condense and seed == 0 and n is not None
                ):  # whole domains would need ~800 calls
                    entry["B"] = await score(sub, condense=True)
                results[label] = entry
                print(
                    f"{label:<22} questions={len(questions):>3}  A: {summarize(list(a.values()))}"
                    + (
                        f"\n{'':<22} {'':>14}  B: {summarize(list(entry['B'].values()))}"
                        if "B" in entry
                        else ""
                    ),
                    flush=True,
                )
                (out / "retrieval-results.json").write_text(json.dumps(results), encoding="utf-8")
    print("\n=== pooled by corpus size (variant A = shipped defaults)")
    for n in [*args.sizes, None]:
        tag = "all" if n is None else str(n)
        pooled = [
            m for label, e in results.items() if f"-{tag}-s" in label for m in e["A"].values()
        ]
        per_subset = [
            statistics.fmean(m["retrieval_hit"] for m in e["A"].values())
            for label, e in results.items()
            if f"-{tag}-s" in label and e["A"]
        ]
        spread = f"subset range {min(per_subset):.2f}-{max(per_subset):.2f}" if per_subset else ""
        print(f"docs={tag:<4} A: {summarize(pooled)}  {spread}")
    if args.condense:
        print("=== variant B (with follow-up condensation), seed-0 subsets only")
        for n in [*args.sizes, None]:
            tag = "all" if n is None else str(n)
            pooled = [
                m
                for label, e in results.items()
                if f"-{tag}-s0" in label and "B" in e
                for m in e["B"].values()
            ]
            base = [
                m
                for label, e in results.items()
                if f"-{tag}-s0" in label and "B" in e
                for m in e["A"].values()
            ]
            print(f"docs={tag:<4} A: {summarize(base)}\n{'':<9} B: {summarize(pooled)}")


async def e2e(args: argparse.Namespace) -> None:
    root, out = Path(args.root), Path(args.out)
    checkpoint = out / "e2e-results.json"
    done: dict[str, Any] = (
        json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
    )
    for n, per_domain in ((12, args.per_domain_small), (45, args.per_domain_large)):
        for domain in DOMAINS:
            sub, questions = make_subset(root, domain, n, 0, out)
            sample = random.Random(f"e2e-{domain}-{n}").sample(
                questions, min(per_domain, len(questions))
            )
            config = config_for(sub, "study", condense=True)
            kb = KnowledgeBase(config)
            await kb.aingest()
            agent = Agent(kb, config)
            judge = LLMJudge(
                factory.build_llm(
                    config, config.evaluation.judge_llm, kb.cache, defaults={"temperature": 0.0}
                )
            )
            control = factory.build_llm(
                config, config.generation.llm, kb.cache, defaults={"temperature": 0.0}
            )
            for raw in sample:
                key = f"{domain}-{n}-{raw['id']}"
                if key in done:
                    continue
                case = EvalCase.model_validate(raw)
                history = list(case.history)
                answer = await agent.aask(case.question, history=history)
                right, right_note = await judge.correctness(case, answer)
                grounded = None  # correctness only: saves a judge call per question on free tiers
                blind = await control.generate(
                    [*history, Message(role="user", content=case.question)],
                    system="You answer questions about US government services (Social Security, veterans "
                    "benefits, student aid, motor vehicles). Answer briefly from your own knowledge.",
                )
                blind_right, _ = await judge.correctness(case, Answer(text=blind.text))
                done[key] = {
                    "docs": n,
                    "domain": domain,
                    "abstained": answer.abstained,
                    "correct": right,
                    "grounded": grounded,
                    "no_docs_correct": blind_right,
                    "cites": [c.source for c in answer.citations],
                    "gold": [s["source"] for s in raw["gold_sources"]],
                    "note": right_note[:200],
                }
                checkpoint.write_text(json.dumps(done), encoding="utf-8")
                print(
                    f"{key}: correct={right} no-docs={blind_right} abstained={answer.abstained}",
                    flush=True,
                )
    report(done)


def report(done: dict[str, Any]) -> None:
    print("\n=== end to end (answered with documents + condensation; graded by a different model)")
    for n in (12, 45):
        rows = [r for r in done.values() if r["docs"] == n]
        scored = [r for r in rows if r["correct"] is not None]
        if not scored:
            continue
        full = [1.0 if r["correct"] == 1.0 else 0.0 for r in scored]
        credit = statistics.fmean(r["correct"] for r in scored)
        blind = [r["no_docs_correct"] for r in scored if r["no_docs_correct"] is not None]
        low, high = wilson_interval(sum(full), len(full))
        cited_gold = [
            any(c in r["gold"] for c in r["cites"])
            for r in scored
            if not r["abstained"] and r["cites"]
        ]
        print(
            f"docs={n:<3} n={len(scored):<3} fully correct {sum(full) / len(full):.2f} [{low:.2f}-{high:.2f}] | "
            f"mean credit {credit:.2f} | declined {sum(r['abstained'] for r in scored)}/{len(scored)} | "
            f"NO-DOCS control credit {statistics.fmean(blind):.2f} | cited a gold page in "
            f"{sum(cited_gold)}/{len(cited_gold)} answers"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=["retrieval", "e2e", "report"])
    parser.add_argument(
        "--root", required=True, help="the md2d folder built by build_multidoc2dial.py"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", type=int, nargs="*", default=[12, 45])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument(
        "--condense", action="store_true", help="also run variant B (needs a model key)"
    )
    parser.add_argument("--per-domain-small", type=int, default=20)
    parser.add_argument("--per-domain-large", type=int, default=30)
    args = parser.parse_args()
    if args.command == "retrieval":
        asyncio.run(retrieval(args))
    elif args.command == "e2e":
        asyncio.run(e2e(args))
    else:
        report(json.loads((Path(args.out) / "e2e-results.json").read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
