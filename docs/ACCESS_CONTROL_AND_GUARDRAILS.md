# Access control, guardrails, verification, tracing and cost

[← back to README](../README.md)

## Access control and multi-tenancy

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

## Guardrails

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
PII detection is regex-based: expect misses and some false positives). Write your own `Guardrail` for more
(see [Plug-ins](PLUGINS.md)). Guardrail redaction covers the question, retrieved chunks and the answer, but
not `history` you pass in.

## Answer verification, tracing, cost

* `generation: {verify_answer: true, verify_action: abstain|flag}`: a second model pass lists claims the
  sources do not support (citation checking proves the *quotes* are real, not that the *claims* follow).
  +1 model call per answer; being a model, it can err either way.
* `tracing: [{provider: jsonl|logging|otel|memory}]` ships each step as an event tagged with a `run_id`.
  **Text derived from the question is stripped by default** (the question, the answer, condensed and rewritten
  queries, the model's abstention note; `include_text: true` keeps it); a failing tracer
  never breaks answering. Tracing counts `embedding_calls` once per query, so agentic retrieval undercounts it.
* `pricing: {"gemini:gemini-2.5-flash": {input_per_mtok: 0.3, output_per_mtok: 2.5}}` (prices are yours to
  supply) makes `answer.usage.cost_usd` and the eval report show cost. `budget: {max_total_usd,
  max_llm_calls_per_question}` stops spending; a dollar budget without a price is a config error.
