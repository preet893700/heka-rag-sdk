# REST server

[← back to README](../README.md)

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

## Security decisions

Access control is only as good as the identity it is given:

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

## Not included

Streaming responses, per-user rate limiting and TLS (terminate it at your proxy). Run a single worker
process: with the local store each process holds its own in-memory copy of the index, so an ingest would
reach only the worker that received it. A shared store such as a Qdrant server is the way to scale out, but
that combination has not been tested.
