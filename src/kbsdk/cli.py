"""Command line: kbsdk ingest | ask | inspect | eval run | eval check | eval gen | eval ablate | serve | presets."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from kbsdk import factory
from kbsdk.agent import Agent
from kbsdk.aio import run_sync
from kbsdk.config import RAGConfig, available_presets
from kbsdk.env import load_env_file
from kbsdk.errors import KbsdkError
from kbsdk.eval.ablation import run_ablation
from kbsdk.eval.dataset import load_dataset
from kbsdk.eval.judge import LLMJudge
from kbsdk.eval.report import EvalReport
from kbsdk.eval.runner import EvalRunner, run_retrieval_eval
from kbsdk.eval.synth import SynthOptions, generate_dataset
from kbsdk.knowledge_base import KnowledgeBase
from kbsdk.text import slugify
from kbsdk.types import RequestContext
from kbsdk.usage import MeteredLLM, track_usage


def _err(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _sync_index(kb: KnowledgeBase, *, quiet: bool = True) -> None:
    """Bring the index up to date before answering, so results always reflect the current files."""
    report = kb.ingest()
    changed = report.files_new or report.files_changed or report.files_removed or report.failed
    if changed or not quiet:
        _err(report.summary())
    if kb.count() == 0:
        raise KbsdkError(
            "The index is empty. Check that knowledge.sources points at your documents."
        )


def _cmd_presets(args: argparse.Namespace) -> int:
    for name in available_presets():
        print(name)
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(RAGConfig.from_file(args.config))
    report = kb.ingest()
    print(report.summary())
    return 1 if report.failed else 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    kb = KnowledgeBase(RAGConfig.from_file(args.config))
    if not args.no_ingest:
        _sync_index(kb)
    for rank, item in enumerate(kb.retrieve(args.query, k=args.k), start=1):
        metadata = item.chunk.metadata
        print(
            f"[{rank}] score {item.score:.3f}  {metadata.get('source')}  |  {metadata.get('context')}"
        )
        print("    " + " ".join(item.chunk.text.split())[:240])
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    config = RAGConfig.from_file(args.config)
    kb = KnowledgeBase(config)
    if not args.no_ingest:
        _sync_index(kb)
    answer = Agent(kb, config).ask(args.question)
    if args.json:
        print(answer.model_dump_json(indent=2))
        return 0
    print(answer.text)
    if answer.abstained:
        print(f"\n[abstained: {answer.abstain_reason}]")
    for number, citation in enumerate(answer.citations, start=1):
        where = f" ({citation.location})" if citation.location else ""
        print(f'\n  [{number}] {citation.source}{where}\n      "{citation.quote}"')
    usage = answer.usage
    print(f"\n[{answer.model or '-'} | {usage.input_tokens} in / {usage.output_tokens} out tokens]")
    return 0


def _cmd_eval_check(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    answerable = sum(c.answerable for c in dataset.cases)
    print(
        f"{len(dataset.cases)} cases ({answerable} answerable, {len(dataset.cases) - answerable} unanswerable)"
    )
    for warning in dataset.warnings():
        print(f"warning: {warning}")
    return 0


def _cmd_eval_run(args: argparse.Namespace) -> int:
    config = RAGConfig.from_file(args.config)
    kb = KnowledgeBase(config)
    if not args.no_ingest:
        _sync_index(kb)

    dataset = load_dataset(args.dataset)
    if args.split or args.tag:
        dataset = dataset.subset(split=args.split, tags=args.tag)
    if args.limit:
        dataset = dataset.model_copy(update={"cases": dataset.cases[: args.limit]})
    if not dataset.cases:
        raise KbsdkError("No cases to run after applying --split/--tag/--limit.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) if args.out else kb.dir / "runs" / f"{stamp}-{slugify(dataset.name)}"
    if args.resume and not args.out:
        raise KbsdkError("--resume needs --out pointing at the earlier run's folder.")

    if args.retrieval_only:  # no LLM, no API key: scores whether the right passage is retrieved
        report = run_sync(run_retrieval_eval(kb, dataset, thresholds=config.evaluation.thresholds))
    else:
        judge = None
        if not args.no_judge:
            component = config.evaluation.judge_llm or config.generation.llm
            judge = LLMJudge(
                factory.build_llm(config, component, kb.cache, defaults={"temperature": 0.0})
            )
        runner = EvalRunner(
            Agent(kb, config),
            judge=judge,
            concurrency=args.concurrency,
            sample_size=args.sample or config.evaluation.sample_size,
        )
        report = runner.run(
            dataset,
            run_dir=out_dir,
            resume=args.resume,
            progress=lambda done, total, case_id: _err(f"  [{done}/{total}] {case_id}"),
        )
    report.save(out_dir / "report.json")
    (out_dir / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    baseline = EvalReport.load(args.baseline) if args.baseline else None
    print(report.to_text(baseline))
    print(f"\nsaved: {out_dir}")
    if report.passed is False:
        return 1
    return 1 if report.counts.get("errors") else 0


def _cmd_eval_gen(args: argparse.Namespace) -> int:
    config = RAGConfig.from_file(args.config)
    kb = KnowledgeBase(config)
    out = Path(args.out)
    if out.exists() and not args.force:
        raise KbsdkError(f"{out} already exists; pass --force to overwrite it.")
    context = None
    if args.context:
        try:
            context = RequestContext.model_validate(json.loads(args.context))
        except (ValueError, TypeError) as exc:
            raise KbsdkError(
                f'--context must be a JSON object like {{"roles": ["hr"]}}: {exc}'
            ) from exc
    if kb.access.enabled and context is None:
        raise KbsdkError(
            "Access control is on, so generated questions need an identity: pass --context "
            '\'{"tenant_id": ..., "roles": [...]}\'. Only passages that identity may see are used.'
        )
    if not args.no_ingest:
        _sync_index(kb)

    component = (
        config.evaluation.generator_llm or config.evaluation.judge_llm or config.generation.llm
    )
    llm = MeteredLLM(
        factory.build_llm(config, component, kb.cache, defaults={"temperature": 0.0}),
        pricing=config.pricing,
    )
    options = SynthOptions(
        answerable=args.count,
        followups=args.followups,
        unanswerable=args.unanswerable,
        test_fraction=args.test_fraction,
        seed=args.seed,
        concurrency=args.concurrency,
        min_chunk_chars=args.min_chars,
        context=context,
    )
    with track_usage() as meter:
        result = run_sync(generate_dataset(kb, llm, options, name=out.stem))
    if not result.dataset.cases:
        raise KbsdkError("No questions passed the checks. " + " ".join(result.warnings))

    out.parent.mkdir(parents=True, exist_ok=True)
    lines = (
        json.dumps(case.model_dump(mode="json", exclude_none=True), ensure_ascii=False)
        for case in result.dataset.cases
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    kept = ", ".join(f"{n} {kind}" for kind, n in result.kept.items())
    print(f"wrote {len(result.dataset.cases)} questions to {out} ({kept})")
    if result.dropped:
        print("dropped by the checks:")
        for reason, n in sorted(result.dropped.items()):
            print(f"  {n:>3}  {reason}")
    unanswerable = [c for c in result.dataset.cases if not c.answerable]
    if unanswerable:
        print(
            "review these: they are labelled unanswerable by a model check, and a wrong label "
            "punishes a correct answer. Delete any the documents actually cover:"
        )
        for case in unanswerable:
            print(f"  {case.id}: {case.question}")
    usage = meter.total
    cost = f" | cost: ${usage.cost_usd:.4f}" if usage.cost_usd is not None else ""
    print(
        f"model: {llm.name} | llm calls: {usage.llm_calls} | cache hits: {usage.cache_hits}{cost}"
    )
    for warning in [*result.warnings, *result.dataset.warnings()]:
        print(f"warning: {warning}")
    if component == config.generation.llm:
        print(
            "warning: the model that wrote these questions is the model being tested, which flatters "
            "it; set evaluation.generator_llm to a different model if you can."
        )
    print(
        "note: synthetic questions echo the documents' wording more than real ones do, so scores on "
        "them run optimistic. Read a sample, fix or delete weak ones, and add real questions."
    )
    return 0


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _required_env(name: str, why: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise KbsdkError(
            f"{why}: set the {name} environment variable (secrets are never taken from flags)."
        )
    return value


def _build_auth(args: argparse.Namespace) -> Any:
    from kbsdk.server import ApiKeyAuth, JwtAuth, NoAuth

    if args.auth == "none":
        if args.host not in LOOPBACK_HOSTS:
            raise KbsdkError(
                f"--auth none is for local development only; refusing to listen on {args.host}. "
                "Use --auth apikey or --auth jwt."
            )
        return NoAuth()
    if args.auth == "apikey":
        keys = _required_env("KBSDK_API_KEYS", "--auth apikey needs the allowed keys").split(",")
        return ApiKeyAuth(keys, trust_caller_context=args.trust_caller_context)
    if args.trust_caller_context:
        raise KbsdkError("--trust-caller-context only applies to --auth apikey.")
    hmac_alg = args.jwt_algorithm.upper().startswith("HS")
    key = (
        _required_env(
            "KBSDK_JWT_SECRET", "--auth jwt with an HS* algorithm needs the shared secret"
        )
        if hmac_alg
        else _required_env("KBSDK_JWT_PUBLIC_KEY", "--auth jwt needs the issuer's public key (PEM)")
    )
    return JwtAuth(
        key,
        algorithm=args.jwt_algorithm,
        audience=args.jwt_audience,
        issuer=args.jwt_issuer,
        tenant_claim=args.jwt_tenant_claim,
        roles_claim=args.jwt_roles_claim,
        user_claim=args.jwt_user_claim,
    )


def _cmd_serve(args: argparse.Namespace) -> int:
    from kbsdk import server

    if args.debug_responses and args.host not in LOOPBACK_HOSTS:
        raise KbsdkError("--debug-responses exposes retrieved passages; only use it on localhost.")
    configs = [RAGConfig.from_file(path) for path in args.config]
    admin_key = os.environ.get("KBSDK_ADMIN_KEY", "").strip() or None
    app = server.create_app(
        configs,
        auth=_build_auth(args),
        admin_key=admin_key,
        cors_origins=args.cors_origin or (),
        ingest_on_startup=not args.no_ingest,
        request_timeout_s=args.timeout,
        max_concurrent_requests=args.max_concurrency,
        expose_debug=args.debug_responses,
        enable_docs=args.docs,
    )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _err(
        f"serving {len(configs)} agent(s) on http://{args.host}:{args.port} "
        f"(auth: {args.auth}, admin endpoints: {'on' if admin_key else 'off'})"
    )
    server.serve(app, host=args.host, port=args.port)
    return 0


def _load_variants(path: str) -> dict[str, dict[str, Any]]:
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise KbsdkError(f"Cannot read variants file {path}: {exc.strerror or exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("variants"), dict):
        data = data["variants"]
    if not isinstance(data, dict) or not data:
        raise KbsdkError("The variants file must map variant names to config overrides.")
    for name, overrides in data.items():
        if overrides is None:
            data[name] = {}
        elif not isinstance(overrides, dict):
            raise KbsdkError(f"Variant {name!r} must be a mapping of config overrides.")
    return {str(name): overrides for name, overrides in data.items()}


def _cmd_eval_ablate(args: argparse.Namespace) -> int:
    base = RAGConfig.from_file(args.config)
    dataset = load_dataset(args.dataset)
    if args.split or args.tag:
        dataset = dataset.subset(split=args.split, tags=args.tag)
    if args.limit:
        dataset = dataset.model_copy(update={"cases": dataset.cases[: args.limit]})
    if not dataset.cases:
        raise KbsdkError("No cases to run after applying --split/--tag/--limit.")
    variants = _load_variants(args.variants)

    report = run_sync(
        run_ablation(
            base,
            variants,
            dataset,
            retrieval_only=args.retrieval_only,
            k=args.k,
            concurrency=args.concurrency,
            sample_size=args.sample,
            no_judge=args.no_judge,
            progress=_err,
        )
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        Path(args.out)
        if args.out
        else Path(base.knowledge.persist_dir) / "ablations" / f"{stamp}-{slugify(dataset.name)}"
    )
    report.save(out_dir / "ablation.json")
    (out_dir / "ablation.md").write_text(report.to_markdown(), encoding="utf-8")
    print(report.to_text())
    print(f"\nsaved: {out_dir}")
    return 0 if any(v.report is not None for v in report.variants) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kbsdk", description="Configurable RAG SDK")
    commands = parser.add_subparsers(dest="command", required=True)

    # Opt-in: nothing reads a .env file unless this flag is given. Existing variables win.
    env = argparse.ArgumentParser(add_help=False)
    env.add_argument(
        "--env-file", help="load API keys from this .env file (real environment variables win)"
    )

    commands.add_parser("presets", help="list built-in config presets", parents=[env]).set_defaults(
        func=_cmd_presets
    )

    def add_config(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "-c", "--config", required=True, help="path to a config file (.yaml/.toml/.json)"
        )

    ingest = commands.add_parser(
        "ingest", help="index the configured documents (incremental)", parents=[env]
    )
    add_config(ingest)
    ingest.set_defaults(func=_cmd_ingest)

    inspect = commands.add_parser(
        "inspect", help="show what retrieval returns for a query", parents=[env]
    )
    add_config(inspect)
    inspect.add_argument("query")
    inspect.add_argument("-k", type=int, default=5)
    inspect.add_argument("--no-ingest", action="store_true", help="skip the automatic index update")
    inspect.set_defaults(func=_cmd_inspect)

    ask = commands.add_parser("ask", help="ask the agent a question", parents=[env])
    add_config(ask)
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true", help="print the full Answer as JSON")
    ask.add_argument("--no-ingest", action="store_true", help="skip the automatic index update")
    ask.set_defaults(func=_cmd_ask)

    serve = commands.add_parser(
        "serve", help="serve the agents over HTTP (needs kbsdk[server])", parents=[env]
    )
    serve.add_argument(
        "-c",
        "--config",
        action="append",
        required=True,
        help="config file (repeat to host several agents)",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--auth",
        required=True,
        choices=["apikey", "jwt", "none"],
        help="how callers prove who they are (secrets come from KBSDK_API_KEYS / KBSDK_JWT_SECRET / "
        "KBSDK_JWT_PUBLIC_KEY; KBSDK_ADMIN_KEY enables POST .../ingest)",
    )
    serve.add_argument(
        "--trust-caller-context",
        action="store_true",
        help="apikey only: let the calling backend send the end user's identity in the body",
    )
    serve.add_argument("--jwt-algorithm", default="HS256")
    serve.add_argument("--jwt-audience")
    serve.add_argument("--jwt-issuer")
    serve.add_argument("--jwt-tenant-claim", default="tenant_id")
    serve.add_argument("--jwt-roles-claim", default="roles")
    serve.add_argument("--jwt-user-claim", default="sub")
    serve.add_argument("--cors-origin", action="append", help="allowed browser origin (repeatable)")
    serve.add_argument("--no-ingest", action="store_true", help="do not index at startup")
    serve.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    serve.add_argument(
        "--max-concurrency", type=int, default=16, help="questions in flight per agent"
    )
    serve.add_argument("--docs", action="store_true", help="serve /docs and /openapi.json")
    serve.add_argument(
        "--debug-responses", action="store_true", help="include retrieved passages (localhost only)"
    )
    serve.set_defaults(func=_cmd_serve)

    evaluation = commands.add_parser("eval", help="measure accuracy on a question set")
    eval_commands = evaluation.add_subparsers(dest="eval_command", required=True)

    check = eval_commands.add_parser("check", help="validate a dataset and list its weaknesses")
    check.add_argument("dataset")
    check.set_defaults(func=_cmd_eval_check)

    gen = eval_commands.add_parser(
        "gen",
        help="draft a starter question set from the indexed documents (needs an LLM)",
        parents=[env],
    )
    add_config(gen)
    gen.add_argument("-o", "--out", required=True, help="where to write the .jsonl dataset")
    gen.add_argument("--count", type=int, default=30, help="answerable questions (default 30)")
    gen.add_argument("--followups", type=int, default=6, help="follow-up questions (default 6)")
    gen.add_argument(
        "--unanswerable",
        type=int,
        default=6,
        help="questions the documents cannot answer (default 6)",
    )
    gen.add_argument("--test-fraction", type=float, default=0.3, help="share held out as 'test'")
    gen.add_argument("--seed", type=int, default=0, help="changes which passages are sampled")
    gen.add_argument("--concurrency", type=int, default=3, help="model calls in flight")
    gen.add_argument(
        "--min-chars", type=int, default=200, help="skip passages shorter than this (default 200)"
    )
    gen.add_argument("--context", help="JSON RequestContext (needed when access control is on)")
    gen.add_argument("--force", action="store_true", help="overwrite an existing output file")
    gen.add_argument("--no-ingest", action="store_true", help="skip the automatic index update")
    gen.set_defaults(func=_cmd_eval_gen)

    run = eval_commands.add_parser(
        "run", help="run the agent over a dataset and score it", parents=[env]
    )
    add_config(run)
    run.add_argument("-d", "--dataset", required=True)
    run.add_argument("--split", choices=["dev", "test"], help="only cases in this split")
    run.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    run.add_argument("--limit", type=int, help="only the first N cases")
    run.add_argument(
        "--concurrency", type=int, default=2, help="cases in flight (keep low on free tiers)"
    )
    run.add_argument("--sample", type=int, help="LLM-judge only N cases (deterministic sample)")
    run.add_argument(
        "--no-judge", action="store_true", help="deterministic metrics only, no judge calls"
    )
    run.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score retrieval only: needs no LLM or API key",
    )
    run.add_argument("--baseline", help="an earlier report.json to compare against")
    run.add_argument("--out", help="folder for results (default: inside the agent's index folder)")
    run.add_argument("--resume", action="store_true", help="continue an interrupted run in --out")
    run.add_argument("--no-ingest", action="store_true", help="skip the automatic index update")
    run.set_defaults(func=_cmd_eval_run)

    ablate = eval_commands.add_parser(
        "ablate",
        help="compare several config variants on the same questions",
        parents=[env],
    )
    add_config(ablate)
    ablate.add_argument("-d", "--dataset", required=True)
    ablate.add_argument("--variants", required=True, help="YAML: variant name -> config overrides")
    ablate.add_argument(
        "--retrieval-only",
        action="store_true",
        help="score retrieval only: needs no LLM or API key",
    )
    ablate.add_argument(
        "-k", type=int, help="chunks scored per question (default: retrieval.final_k)"
    )
    ablate.add_argument("--split", choices=["dev", "test"], help="only cases in this split")
    ablate.add_argument("--tag", action="append", help="only cases with this tag (repeatable)")
    ablate.add_argument("--limit", type=int, help="only the first N cases")
    ablate.add_argument("--concurrency", type=int, default=2)
    ablate.add_argument("--sample", type=int, help="LLM-judge only N cases per variant")
    ablate.add_argument("--no-judge", action="store_true")
    ablate.add_argument("--out", help="folder for the report")
    ablate.set_defaults(func=_cmd_eval_ablate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "env_file", None):
            _err(load_env_file(args.env_file).summary())  # names only, never values
        code: int = args.func(args)
    except KbsdkError as exc:
        _err(f"error: {exc}")
        return 2
    except KeyboardInterrupt:
        _err("interrupted")
        return 130
    return code


if __name__ == "__main__":
    sys.exit(main())
