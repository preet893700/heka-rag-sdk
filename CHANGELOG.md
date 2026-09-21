# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic versioning once it reaches 1.0;
until then any minor version may change the public API.

## [Unreleased]

### Added
- `kbsdk ingest --report` / `--report-only` and `KnowledgeBase.analyze()`: a per-document health report
  (characters, pages, tables, headings, chunks, scanned/OCR pages, noise heuristics) plus notes on identical
  or version-like files. `--report-only` needs no embedding model or API key.
- Evaluation reports now include 95% confidence intervals, a per-case outcome (retrieval miss, wrong answer
  with the right context, wrong abstention, answered unanswerable, citation problem, error) with hints on
  where to look, and a breakdown by question type (case tags).

### Fixed
- The Gemini adapter no longer prints Google's spurious "automatic function calling" notice on every call.

## [0.1.0] - 2026-09-21

First tagged version. Built in four phases; see the README for what was measured and what was not.

### Added
- Core: validated configuration (Python objects, YAML/TOML/JSON, presets), adapter registry with lazy
  loading and install hints, shared metadata-filter dialect, async core with sync wrappers.
- Ingestion: text, PDF (tables, OCR), DOCX, PPTX, XLSX, CSV, HTML and image loaders; four chunkers;
  incremental indexing with a manifest; local and Qdrant vector stores.
- Retrieval: dense, sparse (BM25) and hybrid search, rerankers, query rewrite / multi-query / HyDE,
  follow-up condensation, agentic (LangGraph) multi-step retrieval.
- Generation: grounded answers with SDK-verified citations, abstention, optional answer verification;
  Gemini, Groq, Claude, OpenAI and Ollama providers with caching, retries, rate limiting and fallbacks.
- Production controls: access control and multi-tenancy enforced inside retrieval (fail-closed),
  PII / prompt-injection / scope guardrails, tracing (JSONL, logging, OpenTelemetry), cost tracking, budgets.
- Evaluation: dataset format, retrieval and answer metrics, LLM judge, resumable runs, threshold gating,
  ablation runner, and `kbsdk eval gen` for drafting a starter question set.
- Interfaces: `kbsdk` command line and a REST server (`kbsdk[server]`, `kbsdk serve`) with JWT / API-key
  authentication.

### Known limitations
- Only Groq (`openai/gpt-oss-120b`) has been exercised against a live model; Gemini, Claude, OpenAI and
  Ollama adapters are covered by offline tests only.
- pgvector and other database stores, provider-native citations, streaming answers and enterprise
  connectors are not built.
