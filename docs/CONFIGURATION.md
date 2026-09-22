# Configuration reference

[← back to README](../README.md)

Every stage is a `provider` + `params` pair, so a config is plain data (YAML, TOML, JSON or Python). Secrets
never live in configs (see [Access control & guardrails](ACCESS_CONTROL_AND_GUARDRAILS.md) for the same rule
applied to API keys); unknown keys are errors; switching a component's `provider` replaces its params;
`RAGConfig.json_schema()` and `registry.settings_schema(stage, name)` emit JSON Schemas for a future admin UI.
Metadata filters use one dialect on every store: `{"department": "hr"}`, `{"year": {"$gte": 2025}}`,
`{"$or": [...]}`.

## Install extras

```bash
pip install "heka-rag-sdk[pdf,local,gemini]"   # pick only what you need
pip install "heka-rag-sdk[all]"                # or everything
```

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

## A fuller retrieval config

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

## Stores and providers

| Stage | Providers |
| ----- | --------- |
| store | `local` (NumPy, on disk, exact; tens of thousands of chunks), `qdrant` (embedded in-memory / on-disk, or a server; needs `url` and `api_key_env` for a server) |
| llm | `gemini`, `groq`, `claude`, `openai` (any OpenAI-compatible `base_url`), `ollama` |
| embedder | `local`, `gemini`, `openai`, `ollama`, `hashing` (tests/demos only) |

Claude note: current Claude models reject sampling parameters, so the Claude adapter has **no
`temperature` setting** and never sends one (`max_output_tokens` defaults to 2048). OpenAI reasoning models
may need `temperature: null`.

## API keys

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

## Extending it

Every stage is a small `Protocol` in `heka.rag.interfaces`. Register an adapter and use it by name in any
config; runtime collaborators (an LLM, an embedder) are injected only if the constructor asks for them. See
[Plug-ins](PLUGINS.md) for a full worked example (a custom guardrail, registered and used from YAML).

Third-party packages can ship adapters through the `heka.rag.plugins` entry-point group.
