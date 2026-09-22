# Known limitations and roadmap

[← back to README](../README.md)

## Known limitations

* **Only Groq was exercised live** - see the verification box in the README. Treat your first run on another
  provider as a test of that adapter.
* **pgvector is not built.** No Postgres was available to test it, and database filter translation sits on the
  access-control path, so it was left out rather than shipped unverified. The store interface and the shared
  contract + differential tests are ready for it.
* **Guardrails are heuristic** (see [Access control & guardrails](ACCESS_CONTROL_AND_GUARDRAILS.md)). Guardrail
  redaction covers the question, retrieved chunks and the answer, but not `history` you pass in.
* **Tracing counts** `embedding_calls` once per query, so agentic retrieval undercounts it.
* **OCR is imperfect**, and PDF tables/headings are read by heuristics: battle-tested against a real,
  provenance-logged corpus (see [Evaluation](EVALUATION.md)), but a genuinely new kind of malformed PDF can
  still surface a new edge case - the health report (`heka-rag ingest --report-only`) is the safety net for
  that, not a guarantee that none exists.
* **The local store is exact and in-memory** (loaded at start-up); BM25 is rebuilt in memory when the index
  changes. Fine for tens of thousands of chunks.
* Model names in presets change; override `model` if one is retired.

## Roadmap

1. **Next:** run the LLM-dependent features against a real model and benchmark them (`llm-variants.yaml`,
   verification, agentic); agree accuracy targets from real numbers; pgvector once a Postgres is available.
2. **Later, by demand:** more stores and connectors (SharePoint, Drive, Confluence), multimodal, GraphRAG,
   provider-native citations, streaming answers, an admin app built on the SDK.
