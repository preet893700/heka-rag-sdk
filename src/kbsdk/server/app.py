"""The REST server: a thin, authenticated HTTP layer over `Agent`. Needs `pip install 'kbsdk[server]'`.

    GET  /healthz                      liveness (no auth)
    GET  /readyz                       every agent has an index (no auth, no details)
    GET  /v1/agents                    the hosted agents (authenticated)
    POST /v1/agents/{id}/ask           {"question", "history"?} -> answer with verified citations
    POST /v1/agents/{id}/ingest        re-index the documents (admin key only; disabled without one)

Design choices that matter:
* Identity comes from the credential (see `kbsdk.server.auth`), never from the request body unless the
  operator explicitly trusts the caller. The SDK's access control is only as good as that identity.
* Errors are generic: no policy details, stack traces or provider messages reach the caller; they go to
  the server log tagged with the request id, which is also returned in `X-Request-ID`.
* Question and answer text are not logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from kbsdk.agent import Agent
from kbsdk.config import RAGConfig
from kbsdk.errors import AccessDeniedError, BudgetExceededError, ConfigError, KbsdkError
from kbsdk.knowledge_base import KnowledgeBase
from kbsdk.server.auth import Authenticator, AuthError, NoAuth, secrets_match
from kbsdk.text import slugify
from kbsdk.types import Citation, Message, RequestContext

log = logging.getLogger("kbsdk.server")

MAX_QUESTION_CHARS = 4000
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_CHARS = 20000
ADMIN_HEADER = "x-admin-key"


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    history: list[Message] = Field(default_factory=list, max_length=MAX_HISTORY_MESSAGES)
    # Only accepted when the operator trusts the caller (ApiKeyAuth(trust_caller_context=True)).
    context: RequestContext | None = None


class UsageOut(BaseModel):
    input_tokens: int
    output_tokens: int
    llm_calls: int


class AskResponse(BaseModel):
    request_id: str
    text: str
    citations: list[Citation]
    abstained: bool
    abstain_reason: str | None = None
    escalation: str | None = None
    warnings: list[str] = Field(default_factory=list)
    confidence: float | None = None
    usage: UsageOut
    model: str | None = None
    debug: dict[str, Any] | None = None  # only when the server is started with expose_debug


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, headers: Mapping[str, str] | None = None
    ) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.headers = dict(headers or {})


@dataclass
class HostedAgent:
    id: str
    agent: Agent
    ingest_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    slots: asyncio.Semaphore | None = None

    @property
    def kb(self) -> KnowledgeBase:
        return self.agent.kb

    @property
    def name(self) -> str:
        return self.agent.config.name


def _host(item: Agent | RAGConfig) -> Agent:
    if isinstance(item, Agent):
        return item
    return Agent(KnowledgeBase(item), item)


def create_app(
    agents: Agent | RAGConfig | Sequence[Agent | RAGConfig],
    *,
    auth: Authenticator,
    admin_key: str | None = None,
    cors_origins: Sequence[str] = (),
    ingest_on_startup: bool = True,
    request_timeout_s: float = 60.0,
    max_concurrent_requests: int = 16,
    expose_debug: bool = False,
    enable_docs: bool = False,
) -> FastAPI:
    """Build the ASGI app. Agents are addressed by the slug of their config `name`."""
    items = [agents] if isinstance(agents, Agent | RAGConfig) else list(agents)
    if not items:
        raise ConfigError("create_app needs at least one agent or config.")
    hosted: dict[str, HostedAgent] = {}
    for item in items:
        agent = _host(item)
        agent_id = slugify(agent.config.name)
        if agent_id in hosted:
            raise ConfigError(
                f"Two agents share the name {agent.config.name!r}; names must be unique."
            )
        hosted[agent_id] = HostedAgent(
            agent_id, agent, slots=asyncio.Semaphore(max_concurrent_requests)
        )
    unprotected = [h.name for h in hosted.values() if h.kb.access.enabled]
    if unprotected and isinstance(auth, NoAuth):
        raise ConfigError(
            f"Access control is enabled for {', '.join(unprotected)} but the server has no authentication, "
            "so no caller could ever be identified. Use API-key or JWT auth."
        )
    if admin_key is not None and len(admin_key) < 16:
        raise ConfigError("The admin key must be at least 16 characters.")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if ingest_on_startup:
            for host in hosted.values():
                report = await host.kb.aingest()  # fail fast: a server with no index is not ready
                log.info("agent %s indexed: %s", host.id, report.summary().replace("\n", " | "))
        yield

    app = FastAPI(
        title="kbsdk",
        lifespan=lifespan,
        docs_url="/docs" if enable_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if enable_docs else None,
    )
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
            expose_headers=["X-Request-ID"],
            allow_credentials=False,
        )

    @app.middleware("http")
    async def request_id_and_no_store(request: Request, call_next: Any) -> Any:
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"  # answers can be private to the caller
        return response

    def error_response(
        request: Request, status: int, code: str, message: str, headers: Any = None
    ) -> JSONResponse:
        request_id = getattr(request.state, "request_id", "")
        body = {"error": {"code": code, "message": message, "request_id": request_id}}
        # Set here as well as in the middleware: the catch-all 500 handler runs outside it.
        merged = {"X-Request-ID": request_id, "Cache-Control": "no-store", **(headers or {})}
        return JSONResponse(body, status_code=status, headers=merged)

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return error_response(request, exc.status, exc.code, exc.message, exc.headers)

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({".".join(str(p) for p in e["loc"][1:]) or "body" for e in exc.errors()})
        return error_response(
            request, 422, "invalid_request", f"Invalid request: {', '.join(fields)}."
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.error("request %s failed", getattr(request.state, "request_id", "?"), exc_info=exc)
        return error_response(request, 500, "internal_error", "The request could not be completed.")

    async def identity(request: Request) -> RequestContext | None:
        try:
            return await auth.authenticate(request.headers)
        except AuthError as exc:
            raise ApiError(401, "unauthorized", str(exc), {"WWW-Authenticate": "Bearer"}) from exc

    def find(agent_id: str) -> HostedAgent:
        host = hosted.get(agent_id)
        if host is None:
            raise ApiError(404, "not_found", "No such agent.")
        return host

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> JSONResponse:
        counts = [await h.kb.acount() for h in hosted.values()]
        ready = all(c > 0 for c in counts)
        return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @app.get("/v1/agents")
    async def list_agents(_: RequestContext | None = Depends(identity)) -> dict[str, Any]:
        return {"agents": [{"id": h.id, "name": h.name} for h in hosted.values()]}

    @app.post("/v1/agents/{agent_id}/ask", response_model=AskResponse)
    async def ask(
        agent_id: str,
        body: AskRequest,
        request: Request,
        caller: RequestContext | None = Depends(identity),
    ) -> AskResponse:
        host = find(agent_id)
        context = caller
        if body.context is not None:
            if not auth.trusts_body_context:
                raise ApiError(
                    400,
                    "context_not_accepted",
                    "The caller's identity comes from the credential; do not send `context`.",
                )
            context = body.context
        if sum(len(m.content) for m in body.history) > MAX_HISTORY_CHARS:
            raise ApiError(422, "invalid_request", "Invalid request: history is too long.")

        started = time.perf_counter()
        request_id = request.state.request_id

        async def run() -> Any:
            assert host.slots is not None
            async with host.slots:
                return await host.agent.aask(body.question, history=body.history, context=context)

        try:
            answer = await asyncio.wait_for(run(), timeout=request_timeout_s)
        except AccessDeniedError as exc:
            log.info("request %s denied: %s", request_id, exc)
            raise ApiError(
                403, "access_denied", "You do not have access to this agent's knowledge."
            ) from exc
        except BudgetExceededError as exc:
            log.warning("request %s over budget: %s", request_id, exc)
            raise ApiError(
                429, "budget_exceeded", "The service has reached its usage limit."
            ) from exc
        except TimeoutError as exc:
            log.warning("request %s timed out after %.0fs", request_id, request_timeout_s)
            raise ApiError(504, "timeout", "The request took too long.") from exc
        except KbsdkError as exc:
            log.error("request %s failed: %s", request_id, exc, exc_info=exc)
            raise ApiError(500, "internal_error", "The request could not be completed.") from exc

        log.info(
            "request %s agent=%s abstained=%s llm_calls=%d ms=%d",
            request_id,
            host.id,
            answer.abstained,
            answer.usage.llm_calls,
            (time.perf_counter() - started) * 1000,
        )
        debug = None
        if expose_debug:
            debug = {
                "retrieved": [r.model_dump(mode="json") for r in answer.retrieved],
                "trace": [t.model_dump(mode="json") for t in answer.trace],
                "cost_usd": answer.usage.cost_usd,
            }
        return AskResponse(
            request_id=request_id,
            text=answer.text,
            citations=answer.citations,
            abstained=answer.abstained,
            abstain_reason=answer.abstain_reason,
            escalation=answer.escalation,
            warnings=answer.warnings,
            confidence=answer.confidence,
            usage=UsageOut(
                input_tokens=answer.usage.input_tokens,
                output_tokens=answer.usage.output_tokens,
                llm_calls=answer.usage.llm_calls,
            ),
            model=answer.model,
            debug=debug,
        )

    @app.post("/v1/agents/{agent_id}/ingest")
    async def ingest(agent_id: str, request: Request) -> dict[str, Any]:
        if admin_key is None:
            raise ApiError(403, "admin_disabled", "Admin endpoints are not enabled on this server.")
        if not secrets_match(request.headers.get(ADMIN_HEADER), [admin_key]):
            raise ApiError(
                401, "unauthorized", "Invalid or missing admin key.", {"WWW-Authenticate": "Bearer"}
            )
        host = find(agent_id)
        if host.ingest_lock.locked():
            raise ApiError(
                409, "ingest_in_progress", "An ingest is already running for this agent."
            )
        async with host.ingest_lock:
            report = await host.kb.aingest()
        log.info("agent %s re-indexed: %s", host.id, report.summary().replace("\n", " | "))
        return report.model_dump(mode="json")

    return app
