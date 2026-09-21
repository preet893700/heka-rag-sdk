# heka-rag-sdk

A configurable, plug-and-play RAG SDK for Python. An **agent is a folder of documents plus a config**;
a new agent is a different folder and config, with no code changes. Every answer carries citations that
are checked against the source text, the agent declines when the documents don't contain the answer, who
may see which document is enforced inside retrieval, and accuracy is measured with a built-in evaluation
runner.

> **Status: 0.1.0, a pre-release.** Install `heka-rag-sdk`, import `heka.rag`, run `heka-rag`
> (naming and the `heka` namespace: see `docs/RELEASING.md`).
> Works today: ingestion (PDF with tables + OCR, Office, HTML, text), four chunkers, dense / sparse /
> hybrid / agentic retrieval, rerankers, query rewriting and follow-up condensation, grounded answers
> with verified citations and abstention, **access control and multi-tenancy, guardrails (PII, prompt
> injection, topic routing), answer verification, tracing, cost tracking and budgets**, Gemini / Groq /
> Claude / OpenAI / Ollama models, local and Qdrant stores, the CLI, a REST server, and evaluation, ablation
> and starter-question-generation runners.
> **Not built:** pgvector and other database stores except Qdrant, provider-native citations, more
> connectors. See *Roadmap*.
>
> **Verified vs. not.** Everything is covered by 617 offline tests. **Run live, on Groq
> (`openai/gpt-oss-120b`, free tier), once:** grounded generation with verified citations, abstention, the
> LLM judge, follow-up condensation, rewrite, multi-query, HyDE, answer verification, the agentic loop
> (an HR eval of 26 questions and a 71-question retrieval benchmark, numbers below), `heka-rag eval gen`, and
> the REST server (a real uvicorn process with JWT auth, access control and guardrails, called over HTTP).
> **Run live on Gemini, more lightly:** `gemini-2.5-flash` generation (single questions) and
> `gemini-embedding-001` embeddings (retrieval scores below); a full Gemini answer-and-judge evaluation was not
> completed. **Never run against a live service:** Claude, OpenAI and Ollama calls, the LLM reranker, a remote
> Qdrant server (embedded mode is tested), and a real OpenTelemetry backend. One model and one small
> synthetic set prove the plumbing, not general accuracy: measure on your own questions.

## Install

The package is not on a package index yet, so install it from this repository's release. Both commands
below were tested in clean environments with no GitHub login:

```bash
# From the git tag:
pip install "heka-rag-sdk[pdf,local,gemini] @ git+https://github.com/preet893700/heka-rag-sdk.git@v0.1.0"

# Or from the wheel attached to the release (check it against the SHA-256 in the release notes):
pip install "heka-rag-sdk[pdf,local,gemini] @ https://github.com/preet893700/heka-rag-sdk/releases/download/v0.1.0/heka_rag_sdk-0.1.0-py3-none-any.whl"
```

Use `[all]` in place of the extras list for everything below. Once the package is published to a private
package index (see `docs/RELEASING.md`), the install becomes simply `pip install "heka-rag-sdk[pdf,local,gemini]"`.

The source is publicly readable, but the licence is proprietary: reading it grants no right to use, copy,
modify or distribute it (see `LICENSE`).

| Extra       | Adds                                                                  |
| ----------- | --------------------------------------------------------------------- |
| `pdf`       | PDFs: text and table extraction (`pypdf`, `pdfplumber`)               |
| `ocr`       | scanned PDFs and images (RapidOCR on ONNX; models ship in the package)|
| `office`    | DOCX, PPTX, XLSX                                                      |
| `web`       | HTML pages and wiki exports                                           |
| `local`     | on-device embeddings and rerankers (`fastembed`, no PyTorch)          |
| `gemini` `groq` `anthropic` `openai` `ollama` | model providers (via LangChain)     |
| `qdrant`    | Qdrant vector store (embedded or server)                              |
| `agentic`   | multi-step retrieval (LangGraph)                                      |
| `otel`      | OpenTelemetry tracing                                                 |
| `server`    | REST server (`fastapi`, `uvicorn`, `pyjwt`)                           |

The core install needs only `pydantic`, `pyyaml` and `numpy`. A feature whose extra is missing fails with
the exact `pip install` command to run.

## Quick start

```python
from heka.rag import Agent, KnowledgeBase, RAGConfig

config = RAGConfig.preset(
    "free-tier-dev",  # local embeddings + Gemini; also: high-accuracy, low-cost, on-prem
    name="HR Assistant",
    instructions="Answer employee questions about company policy. Escalate to hr@example.com if unsure.",
    knowledge={"sources": [{"location": "./docs/hr", "metadata": {"department": "hr"}}]},
)

kb = KnowledgeBase(config)
kb.ingest()  # incremental: only new, changed or removed files are processed

agent = Agent(kb, config)
answer = agent.ask("How many casual leave days do I get?")

print(answer.text)
for c in answer.citations:  # each quote was verified to appear verbatim in the cited source
    print(c.source, c.location, "-", c.quote)
print(answer.abstained, answer.abstain_reason, answer.escalation, answer.warnings, answer.usage)
```

`ask`/`ingest` are sync wrappers over `aask`/`aingest` (async is the core). Pass `history=[...]` for
follow-up questions and `context=RequestContext(...)` to say who is asking.

**A second agent is a second folder and config.** `examples/hr_agent` and `examples/onboarding_agent`
use identical code; `examples/hr_agent/hr-secure.yaml` turns on every Phase 3 control from config alone.

### API keys

Keys are read from the environment (`GOOGLE_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`), never from config files (a param named like `api_key` is rejected). Loading a `.env` file
is **opt-in** - nothing is read unless you ask:

```
heka-rag ask -c hr.yaml "..." --env-file .env    # CLI (every command that may need a key accepts it)
```
```python
from heka.rag import load_env_file

load_env_file(".env")  # in code
```
```yaml
env_file: .env                                # in a config (relative to the config file)
```

Variables already set in the real environment win; only variable *names* are ever reported, never values.

## Command line

```
heka-rag ingest  -c examples/hr_agent/hr.yaml
heka-rag ingest  -c examples/hr_agent/hr.yaml --report-only                   # how was each document read?
heka-rag inspect -c examples/hr_agent/hr.yaml "can I work from home?" -k 3   # what the model would read
heka-rag ask     -c examples/hr_agent/hr.yaml "How many casual leave days do I get?"
heka-rag eval check examples/hr_agent/questions.jsonl                         # dataset health
heka-rag eval run -c hr.yaml -d questions.jsonl --retrieval-only              # no LLM, no key
heka-rag eval run -c hr.yaml -d questions.jsonl --baseline old/report.json    # full run, compare
heka-rag eval ablate -c bench.yaml -d questions.jsonl --variants v.yaml --retrieval-only -k 3
heka-rag eval gen -c hr.yaml -o questions.jsonl --count 30      # draft a starter question set
heka-rag serve -c hr.yaml --auth jwt                             # REST server (heka-rag-sdk[server])
```

## The pipeline

```
question ─► input guardrails ─► condense follow-up ─► transform ─► retrieve (access-filtered)
        ─► fuse ─► rerank ─► expand to parent section ─► context guardrails ─► [abstain?]
        ─► answer + verified citations ─► answer verification ─► output guardrails
```

Every step except retrieval and answering is optional and off by default; every step leaves a `TraceEvent`
in `answer.trace`, and all model calls for one question are summed (and priced) into `answer.usage`.

```yaml
retrieval:
  mode: hybrid                   # dense | sparse | hybrid
  fusion: weighted               # rrf | weighted
  dense_weight: 0.7
  top_k: 20                      # candidates fetched      final_k: 5   # chunks the model reads
  reranker: {provider: local_cross_encoder, params: {model: BAAI/bge-reranker-base}}
  query_transforms: [{provider: multi_query, params: {n: 3}}]   # +1 model call per question
  condense_followups: true       # +1 model call, only when there is history
  agentic: {max_steps: 3}        # multi-document questions: search, judge, search again (+1 call per step)
chunker: {provider: parent_child, params: {parent_size: 3200, child_size: 400}}
knowledge:
  extraction: layout             # PDF tables become Markdown tables
  ocr: {provider: rapidocr}      # read scanned pages and image files
```

## Production controls

### Access control and multi-tenancy

Who may see which chunk is decided by tags on your sources and by the `RequestContext` your application
passes, and enforced **inside retrieval**: the model never sees content the caller may not read, and a
caller-supplied `filter` can only narrow results, never widen them.

```yaml
access:
  tenant_field: tenant_id        # chunk tag matched to context.tenant_id
  roles_field: allowed_roles     # chunk tag; needs an overlap with context.roles
  attribute_fields: {region: region}   # optional: context.attributes["region"] must match the chunk's
knowledge:
  sources:
    - location: ./docs
      exclude: [code-of-conduct.md]
      metadata: {tenant_id: acme, allowed_roles: ["*"]}          # "*" = everyone
    - location: ./docs/code-of-conduct.md
      metadata: {tenant_id: acme, allowed_roles: [hr, manager]}
```
```python
agent.ask("...", context=RequestContext(tenant_id="acme", roles=["employee"]))
```

* **Fail-closed.** Asking without a context raises `AccessDeniedError`; a chunk missing a tag is visible to
  nobody (ingestion warns per file); the same file reached through two sources with *different* tags is a
  config error (use `exclude:`), so a restricted document can't be indexed a second time as public.
* The filter dialect behaves identically on every store; a differential test runs 20 filter shapes and
  every access-policy combination through the built-in store and Qdrant and requires the same answers.

### Guardrails

```yaml
guardrails:
  - provider: pii            # email / phone / card (Luhn-checked) / SSN / IPv4: redact or block
    params: {stages: [input], allow: ['@acme\.example$']}
  - provider: injection      # "ignore your instructions", prompt-delimiter breakouts: block the question,
                             #   and drop any retrieved document that contains them
  - provider: scope          # topics that must go to a person
    params:
      rules: [{name: emergency, keywords: [chest pain], action: escalate, message: "Call emergency services."}]
```

Guardrails run on the question, on each retrieved chunk and on the drafted answer. `escalate` sets
`answer.escalation` (the rule name) so your application can route to a human. **They are deterministic
pattern matching: a cheap first filter and safety net, not a security boundary** (they can be evaded, and
PII detection is regex-based: expect misses and some false positives). Write your own `Guardrail` for more.

### Answer verification, tracing, cost

* `generation: {verify_answer: true, verify_action: abstain|flag}`: a second model pass lists claims the
  sources do not support (citation checking proves the *quotes* are real, not that the *claims* follow).
  +1 model call per answer; being a model, it can err either way.
* `tracing: [{provider: jsonl|logging|otel|memory}]` ships each step as an event tagged with a `run_id`.
  **Text derived from the question is stripped by default** (the question, the answer, condensed and rewritten
  queries, the model's abstention note; `include_text: true` keeps it); a failing tracer
  never breaks answering.
* `pricing: {"gemini:gemini-2.5-flash": {input_per_mtok: 0.3, output_per_mtok: 2.5}}` (prices are yours to
  supply) makes `answer.usage.cost_usd` and the eval report show cost. `budget: {max_total_usd,
  max_llm_calls_per_question}` stops spending; a dollar budget without a price is a config error.

### Stores and providers

| Stage | Providers |
| ----- | --------- |
| store | `local` (NumPy, on disk, exact; tens of thousands of chunks), `qdrant` (embedded in-memory / on-disk, or a server; needs `url` and `api_key_env` for a server) |
| llm | `gemini`, `groq`, `claude`, `openai` (any OpenAI-compatible `base_url`), `ollama` |
| embedder | `local`, `gemini`, `openai`, `ollama`, `hashing` (tests/demos only) |

Claude note: current Claude models reject sampling parameters, so the Claude adapter has **no
`temperature` setting** and never sends one (`max_output_tokens` defaults to 2048). OpenAI reasoning models
may need `temperature: null`.

## REST server

`pip install "heka-rag-sdk[server]"` adds an HTTP layer over your agents, for callers that are not Python:

```
export HEKA_RAG_JWT_SECRET=...            # HS* secret (or HEKA_RAG_JWT_PUBLIC_KEY for RS*/ES*); >= 32 bytes
export HEKA_RAG_ADMIN_KEY=...             # optional: enables POST .../ingest
heka-rag serve -c hr.yaml -c onboarding.yaml --auth jwt --jwt-audience kb --cors-origin https://intranet.example
```
```
POST /v1/agents/{id}/ask      {"question": "...", "history": [{"role": "user", "content": "..."}, ...]}
POST /v1/agents/{id}/ingest   re-index (admin key in X-Admin-Key; disabled unless configured)
GET  /v1/agents   /healthz   /readyz
```
`{id}` is the slug of the config's `name`. An answer returns `text`, verified `citations`, `abstained` /
`abstain_reason`, `escalation`, `warnings`, token `usage` and a `request_id` (also the `X-Request-ID` header).

Security decisions, since access control is only as good as the identity it is given:

* **Identity comes from the credential, not the body.** `--auth jwt` verifies the signature (only the
  configured algorithm; `exp` required; `aud`/`iss` checked when set) and maps claims to tenant, user and roles
  (`--jwt-tenant-claim`, `--jwt-roles-claim`). A `context` in the body is rejected with 400. `--auth apikey`
  proves the *application*, not the user; with `--trust-caller-context` your own backend may then vouch for the
  end user in the body: use that only when the key never reaches a browser. In Python, `CustomAuth(fn)` plugs
  in your own session/SSO check. `create_app` lets you pass all of this programmatically.
* **`--auth` is required**, `--auth none` only binds to localhost, and the server refuses to start with no
  authentication for an agent that has access control. Secrets come from environment variables, never flags.
* **Errors are generic** (no policy details, paths or provider messages); details go to the server log under
  the `request_id`. Question and answer text are not logged. Answers are sent `Cache-Control: no-store`.
* Requests have a timeout (`--timeout`) and a per-agent concurrency cap (`--max-concurrency`); questions are
  limited to 4000 characters and history to 20 messages. `/docs` is off unless `--docs`; CORS is off unless
  `--cors-origin`.

Not included: streaming responses, per-user rate limiting and TLS (terminate it at your proxy). Run a single
worker process: with the local store each process holds its own in-memory copy of the index, so an ingest
would reach only the worker that received it. A shared store such as a Qdrant server is the way to scale
out, but that combination has not been tested.

## Measuring accuracy

Put real questions in a JSONL file (`heka.rag.eval.EvalCase`): the question, the answer a human expects, where
it lives (a document, optionally a `section` and an exact `quote`), whether the documents can answer it at
all, a `dev`/`test` split, optional `tags`, `history` and `context` (who is asking). Tune on `dev`; treat
`test` as the honest number.

| Metric | LLM? | Meaning |
| ------ | ---- | ------- |
| `retrieval_hit` / `_recall` / `_mrr` | no | did the right passage come back in the top k, and how high |
| `citation_valid_rate`, `citation_source_match` | no | proposed quotes that were real; citations pointing at the right documents |
| `abstain_correct`, `unanswerable_abstain_rate`, `answerable_abstain_rate` | no | declines exactly when it should |
| `correctness`, `groundedness` | judge | answer matches the reference; every claim follows from the sources |

With a `quote` in a gold reference, a "hit" means the returned chunk *contains* that text. An LLM judge is a
fast proxy, not ground truth: read a sample of its reasons before trusting a number.

### Reading a report honestly

Every report (`heka-rag eval run`) also shows:

* **95% intervals.** With 50-100 questions a score of 0.90 is really "somewhere around 0.82-0.96", so a
  difference of a few points between two configs is usually noise. Yes/no metrics use a Wilson interval; graded
  ones (judge scores, reciprocal rank) a rougher normal approximation.
* **Outcomes.** Each case is put in one bucket saying where it failed: `retrieval_miss` (the expected passage was
  not retrieved), `wrong_with_right_context` (right passage, wrong answer), `wrong_abstention` (declined although
  the passage was there), `answered_unanswerable` (hallucinated), `citation_problem`, `error`. Each has a
  different fix, and the report says where to look. A declined answer whose passage was never retrieved counts as
  a retrieval miss, since that is the root cause. `correctness=1` with a retrieval miss counts as ok: the answer
  came from another passage, so the gold label may be incomplete.
* **By question type.** The same metrics per case `tag`. This is where averages hide problems: on the synthetic
  benchmark the overall retrieval hit rate was 0.97 (interval 0.90-0.99), yet follow-up questions scored 0.29
  MRR and answers buried in long documents 0.61.

### Checking how documents were read

Answer quality is capped by extraction quality, and extraction fails silently: a scan becomes zero characters,
letters come out spaced apart, a header repeated on every page fills the chunks. Before tuning anything:

```
heka-rag ingest -c hr.yaml --report-only     # no embedding, no index changes, no API key
heka-rag ingest -c hr.yaml --report          # index, then report
```

It loads and chunks every file exactly as ingestion does, and lists per file: characters, pages, tables,
headings, chunks and flags (no text extracted, scanned pages and whether OCR is set, OCR misreads to spot-check,
very little text per page, no headings in a long document, unreadable or spaced-out text, repeated page
furniture, fragmented chunks). Across files it notes identical content and files that look like versions of one
another (`policy_v1.pdf`, `policy_v2.pdf`, `holidays-2025.md`, `holidays-2026.md`): conflicting versions are a
classic source of confident wrong answers. From Python: `kb.analyze()`.

The checks are heuristics: a flag means "look at this file". They were tuned so that two clean corpora (the
25-document benchmark and the HR example) report nothing, and are unit-tested against synthetic bad inputs; they
have **not** been run on real messy PDFs yet.

### Drafting a starter question set

No real questions yet? `heka-rag eval gen` drafts a set from your indexed documents (needs a model):

```
heka-rag eval gen -c hr.yaml -o questions.jsonl --count 30 --followups 6 --unanswerable 6
```

A model writes each question, reference answer and supporting quote from one passage; the SDK then rejects
drafts whose quote is not verbatim in the passage, whose numbers are not in the passage, that copy six or
more consecutive words of it, that point at "the passage", or that duplicate another question. Follow-ups
come with the earlier turns as `history`. Unanswerable questions are proposed from a document outline and
checked against what retrieval returns. Output is deterministic per `--seed`, split into `dev`/`test`,
and every case is tagged `synthetic`. With access control on, pass `--context` and only passages that
identity may see are used. Set `evaluation.generator_llm` to a model *other than* the one being tested.

Read the result before trusting it. On the HR example (live, `qwen/qwen3.8-27b` writing) the answerable and
follow-up questions were sound, but the model-based "unanswerable" check is imperfect: it labelled "what is
the deadline to submit expense reports?" unanswerable although the policy states a 30-day deadline, and a
wrong label makes a correct answer count as a hallucination. The command therefore prints every
unanswerable question for you to review. Synthetic questions also echo the documents' wording more than real
ones do, so scores on them run optimistic: use them to get going and replace them with real questions.

`heka-rag eval ablate` scores the same questions under several configs (each with its own index) and prints
deltas plus a breakdown by question `tag`; a variant that cannot run is reported as skipped.

### What was measured (and what it does not show)

On `examples/benchmark` (25 documents, 71 synthetic questions built to punish specific weaknesses; retrieval
only, local models, 66 answerable questions):

| Setup (k = chunks the model reads) | hit | MRR |
| ---------------------------------- | --- | --- |
| dense (k=5) | 0.97 | 0.85 |
| dense with Gemini embeddings (`gemini-embedding-001`, k=5) | 0.91 | 0.89 |
| sparse BM25 only | 0.73 | 0.72 |
| hybrid, equal-weight RRF | 0.85 | 0.81 |
| hybrid, weighted (dense 0.7) | 0.92 | 0.86 |
| dense + MiniLM reranker | 0.94 | 0.87 |
| dense (k=3) | 0.94 | 0.84 |
| dense + bge-reranker-base (k=3) | 0.94 | **0.92** |
| structure-aware chunks 1200 / 600 chars (k=3) | 0.94 / 0.95 | 0.84 / 0.88 |
| parent-child / semantic chunks (k=3) | 0.95 / 0.95 | 0.87 / 0.87 |
| fixed-size chunks, no headings (k=3) | 0.85 | 0.71 |

Dense retrieval with heading-aware chunks was already strong and the extras were trade-offs, not free wins:
equal-weight hybrid *hurt* everyday-wording questions (vocabulary MRR 0.93 to 0.68) while lifting exact-ID and
buried-answer ones; weighting toward dense recovered most of the loss but did not beat plain dense on hit rate; only the large reranker improved ordering (+0.08 MRR,
about 2 s CPU per question, 1 GB); gains did not stack. Defaults therefore stay `dense` + `structure_aware`.
**The LLM-based levers** (same benchmark, retrieval only, base = hybrid + small reranker, Groq
`gpt-oss-120b`; `examples/benchmark/llm-variants.yaml`):

| Variant | hit | MRR | follow-up MRR | extra model calls |
| ------- | --- | --- | ------------- | ----------------- |
| base | 0.94 | 0.87 | 0.36 | none |
| + follow-up condensation | 0.95 | **0.92** | **0.81** | one per follow-up question |
| + rewrite / multi-query / HyDE | 0.94 | 0.87 | 0.36 | one per question (minutes instead of seconds at a 25 requests/min limit) |

Condensation is the only one that paid off: it fixes questions like "and for contractors?" that cannot be
searched without the conversation. Rewrite, multi-query and HyDE changed nothing here (they do fire: rewrite
turned "temporary assignment with a partner organisation" into "secondment application form"), because the
dense retriever already found those passages; they may matter on messier real questions, so try them on
yours before paying for the extra calls.

**HR example** (26 questions, live Groq generation and judge; `examples/hr_agent`): correctness 0.98,
groundedness 1.00, all 5 unanswerable questions declined, no answerable question declined. Turning on
`verify_answer` left every score unchanged and cost about two-thirds more tokens (34k vs 21k). It did catch one real
hallucination in an ad-hoc test (the model claimed "manager approval over $100"; the policy says managers
approve up to $1,000), and it flagged one honest "the policy does not address this" sentence, so it can
reject answers that were fine: prefer `verify_action: flag` until you have measured it on your questions.
An LLM judge grading its own family of model is a fast proxy, not ground truth.

Licences: the default reranker is Apache-2.0 and `bge-reranker-base` is MIT. `jina-reranker-v2-base-multilingual`
(also in fastembed) is CC-BY-NC, non-commercial: do not use it.

## Configuration

Every stage is a `provider` + `params` pair, so a config is plain data (YAML, TOML, JSON or Python). Secrets
never live in configs; unknown keys are errors; switching a component's `provider` replaces its params;
`RAGConfig.json_schema()` and `registry.settings_schema(stage, name)` emit JSON Schemas for a future admin
UI. Metadata filters use one dialect on every store: `{"department": "hr"}`, `{"year": {"$gte": 2025}}`,
`{"$or": [...]}`.

## Built-in components

| Stage | Providers |
| ----- | --------- |
| loader | `text`, `pdf` (tables, OCR), `docx`, `pptx`, `xlsx`, `csv`, `html`, `image` |
| chunker | `structure_aware`, `fixed`, `parent_child`, `semantic` |
| retrieval | dense, sparse (BM25), hybrid (RRF or weighted), agentic wrapper |
| reranker | `local_cross_encoder`, `llm` |
| query_transform | `rewrite`, `multi_query`, `hyde` |
| guardrail | `pii`, `injection`, `scope` |
| tracer | `jsonl`, `logging`, `otel`, `memory` |
| ocr | `rapidocr` |
| cache | `disk` (SQLite), `memory` |

Every LLM is wrapped with the response cache, a shared rate limiter, retries and optional
`reliability.fallback_llms`; every model call is metered.

## Plug-ins

Every stage is a small `Protocol` in `heka.rag.interfaces`. Register an adapter and use it by name in any
config; runtime collaborators (an LLM, an embedder) are injected only if the constructor asks for them:

```python
from heka.rag import register
from heka.rag.adapters._deps import Settings


class MyGuardrailSettings(Settings):  # validated; unknown keys are errors
    banned: list[str] = []


@register("guardrail", "banned_words")
class BannedWords:
    settings_model = MyGuardrailSettings

    def __init__(self, settings):
        self.banned = settings.banned

    async def check(self, text, *, stage, context=None):
        from heka.rag.types import GuardrailResult

        hit = next((w for w in self.banned if w in text.lower()), None)
        return (
            GuardrailResult(action="block", reason=hit, message="Please rephrase.")
            if hit
            else GuardrailResult()
        )
```

Third-party packages can ship adapters through the `heka.rag.plugins` entry-point group.

## Known limitations

* **Only Groq was exercised live** - see the box at the top. Treat your first run on another provider as a
  test of that adapter.
* **pgvector is not built.** No Postgres was available to test it, and database filter translation sits on the
  access-control path, so it was left out rather than shipped unverified. The store interface and the shared
  contract + differential tests are ready for it.
* **Guardrails are heuristic** (see above). Guardrail redaction covers the question, retrieved chunks and the
  answer, but not `history` you pass in.
* **Tracing counts** `embedding_calls` once per query, so agentic retrieval undercounts it.
* **OCR is imperfect** and PDF tables are detected only when ruled or clearly aligned.
* **The local store is exact and in-memory** (loaded at start-up); BM25 is rebuilt in memory when the index
  changes. Fine for tens of thousands of chunks.
* Model names in presets change; override `model` if one is retired.

## Roadmap

1. **Next:** run the LLM-dependent features against a real model and benchmark them (`llm-variants.yaml`,
   verification, agentic); agree accuracy targets from real numbers; pgvector once a Postgres is available.
2. **Later, by demand:** more stores and connectors (SharePoint, Drive, Confluence), multimodal, GraphRAG,
   provider-native citations, streaming answers, an admin app built on the SDK.

## Development

```
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest          # no network or API keys needed
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mypy
python examples/benchmark/build_benchmark.py   # regenerate the benchmark (deterministic)
```

Real company documents and question sets belong in `data/`, which is git-ignored, as are the `.heka-rag/`
index folders. Building, versioning and publishing to a private index are in `docs/RELEASING.md`; changes
are recorded in `CHANGELOG.md`. The licence is proprietary (see `LICENSE`).
