# Measuring accuracy

[← back to README](../README.md)

## Verified vs. not, in full

The README's "We measured it, honestly" section is the short version; this is the complete picture, covered
by 644 offline tests plus what has and hasn't been run against a live service:

* **Run live, on Groq (`openai/gpt-oss-120b`, free tier), once:** grounded generation with verified
  citations, abstention, the LLM judge, follow-up condensation, rewrite, multi-query, HyDE, answer
  verification, the agentic loop (the HR eval and the retrieval benchmark below), `heka-rag eval gen`, and
  the REST server (a real uvicorn process with JWT auth, access control and guardrails, called over HTTP).
* **Run live on Gemini, more lightly:** `gemini-2.5-flash` generation (single questions) and
  `gemini-embedding-001` embeddings (retrieval scores below); a full Gemini answer-and-judge evaluation was
  not completed.
* **Never run against a live service:** Claude, OpenAI and Ollama calls, the LLM reranker, a remote Qdrant
  server (embedded mode is tested), and a real OpenTelemetry backend.

One model and one small synthetic set prove the plumbing, not general accuracy: measure on your own
questions.

## Setting up a question set

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

## Reading a report honestly

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

## Checking how documents were read

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
25-document benchmark and the HR example) report nothing, and are unit-tested against synthetic bad inputs, and
against a real, provenance-logged corpus of public-domain PDFs (`examples/public_corpora/`,
`data/public-corpus/pdf-torture/`) chosen for specific known failure modes (multi-column layout, running
headers, borderless tables).

## Drafting a starter question set

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

## What was measured (and what it does not show)

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

**Real IRS PDFs** (`data/public-corpus/irs-small/`, government publications, not written for this SDK):
before the font-size/layout-aware heading detector was added, a 59-page real publication produced only 1
detected heading; the same document now produces 296, with zero new health-report flags.

Licences: the default reranker is Apache-2.0 and `bge-reranker-base` is MIT. `jina-reranker-v2-base-multilingual`
(also in fastembed) is CC-BY-NC, non-commercial: do not use it.
