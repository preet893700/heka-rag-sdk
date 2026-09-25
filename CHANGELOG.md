# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project uses semantic versioning once it reaches 1.0;
until then any minor version may change the public API.

## [Unreleased]

### Fixed
- Office files: text inside grouped PowerPoint shapes is no longer skipped; Excel formulas without a saved result show the formula instead of a blank cell and hidden sheets are labelled `(hidden)`; Word page headers and footers are kept (once each). Answers now carry a warning when some citations were dropped for failing verification.
- HTML: text sitting directly inside a `<div>`, `<section>` or `<body>` (not wrapped in a paragraph) was skipped, and inline markup such as `<b>` or `<a>` inserted stray spaces ("Wednes day", "word ,") that could make a correct quote fail the citation check. Both are fixed.
- PDF: the borderless-table fallback could mistake multi-column or gappy prose for a table and silently drop
  most of the page (up to 22% of a real Federal Register excerpt, 5-10% of several IRS publications). A table
  is now accepted only as a run of consecutive rows whose columns line up, text-like cells (running prose,
  including aligned three-column pages) are never read as a table, and a safety net keeps a page's plain text
  whenever a table rendering would lose more than 2% of its words.
- Citation checking rejected correct quotes that differ from the source only by look-alike characters:
  non-breaking and other Unicode hyphens, minus signs, soft hyphens, zero-width characters, composed vs
  decomposed accents, and by extraction spacing around punctuation ("noncitizen ;", "You ' ll") or a "[10]"
  link marker the model left out. Quotes with a different word, number, punctuation mark or meaning are still
  rejected.

### Added
- PDF (`extraction: layout`): headings from font size and weight, running headers and footers stripped,
  two-column reading order, borderless tables, AcroForm fields, and `password_env` for encrypted PDFs.

## [0.1.0] - 2026-09-22

First tagged version, by Hekaos. See the README for what was measured and what was not.

### Names
- Install `heka-rag-sdk`, import `heka.rag` (a PEP 420 namespace shared by future `heka.*` packages),
  command `heka-rag`, environment variables `HEKA_RAG_*`, error base class `HekaRagError`, state folder
  `.heka-rag/`, plug-in group `heka.rag.plugins`. Renamed from the internal placeholder `kbsdk` before release.
- Proprietary licence; Hekaos is the copyright holder, author and maintainer.

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
  ablation runner, and `heka-rag eval gen` for drafting a starter question set. Reports include 95%
  confidence intervals, a per-case outcome (retrieval miss, wrong answer with the right context, wrong
  abstention, answered unanswerable, citation problem, error) with hints on where to look, and a breakdown
  by question type.
- `heka-rag ingest --report` / `--report-only` and `KnowledgeBase.analyze()`: a per-document health report
  (characters, pages, tables, headings, chunks, scanned/OCR pages, noise heuristics) plus notes on identical
  or version-like files. `--report-only` needs no embedding model or API key.
- Interfaces: the `heka-rag` command line and a REST server (`heka-rag-sdk[server]`, `heka-rag serve`)
  with JWT / API-key authentication.
- `examples/public_corpora`: a test bed on public data (MultiDoc2Dial, GOV.UK guides, IRS PDFs) with
  polite, provenance-logged fetchers and an accuracy-versus-collection-size study.

### Known limitations
- **Accuracy on real documents is not established.** On public government documents with real
  conversational questions, retrieval alone put the right passage in the top 5 for 75% of questions at 12
  documents, 64% at 45 and 54% for a whole 92-138-document domain (a conservative floor: labels name one
  passage). End-to-end answer accuracy on real documents has not been measured; free-tier limits blocked it.
  The earlier 0.97 figure came from a synthetic benchmark and was too optimistic.
- PDF section headings are detected poorly on multi-column PDFs (structure-aware chunking then degrades to
  size-based chunks); a font-size-based detector is planned, not built.
- Live model runs used Groq (`openai/gpt-oss-120b`, with Qwen as an independent grader) and, more lightly,
  Gemini generation and embeddings. Claude, OpenAI and Ollama adapters and the LLM reranker are covered by
  offline tests only.
- pgvector and other database stores, provider-native citations, streaming answers and enterprise
  connectors are not built.
