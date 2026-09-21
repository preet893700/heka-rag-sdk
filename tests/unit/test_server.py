import asyncio
import base64
import hashlib
import hmac
import json
import logging
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("jwt")

import httpx  # noqa: E402
import jwt  # noqa: E402
from cryptography.hazmat.primitives import serialization  # noqa: E402
from cryptography.hazmat.primitives.asymmetric import rsa  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from conftest import CORPUS, ScriptedLLM, answering_reply  # noqa: E402
from kbsdk import Agent, ConfigError, KnowledgeBase  # noqa: E402
from kbsdk.cli import main  # noqa: E402
from kbsdk.errors import AccessDeniedError, BudgetExceededError, KbsdkError  # noqa: E402
from kbsdk.server import ApiKeyAuth, CustomAuth, JwtAuth, NoAuth, create_app  # noqa: E402
from kbsdk.server.auth import AuthError  # noqa: E402
from kbsdk.types import RequestContext  # noqa: E402

API_KEY = "test-api-key-0123456789abcdef"
ADMIN_KEY = "test-admin-key-0123456789abcd"
JWT_SECRET = "s" * 40


@pytest.fixture
def secured_config(make_config, docs_dir):
    """leave.md is HR-only; expenses.md is for everyone."""
    knowledge = make_config().to_dict()["knowledge"]
    knowledge["sources"] = [
        {"location": str(docs_dir / "leave.md"), "metadata": {"roles": ["hr"]}},
        {"location": str(docs_dir / "expenses.md"), "metadata": {"roles": ["*"]}},
    ]
    return make_config(knowledge=knowledge, access={"roles_field": "roles"})


def agent_for(config, llm=None):
    kb = KnowledgeBase(config)
    kb.ingest()
    return Agent(kb, config, llm=llm or ScriptedLLM(answering_reply))


def client_for(agent, **kwargs):
    kwargs.setdefault("auth", ApiKeyAuth([API_KEY]))
    kwargs.setdefault("ingest_on_startup", False)
    return TestClient(create_app(agent, **kwargs), raise_server_exceptions=False)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def token(claims=None, secret=JWT_SECRET, algorithm="HS256", **kw):
    payload = {"sub": "u1", "tenant_id": "acme", "roles": ["employee"], "exp": time.time() + 300}
    payload.update(claims or {})
    for key in kw.get("drop", ()):
        payload.pop(key, None)
    return jwt.encode(payload, secret, algorithm=algorithm)


LEAVE_Q = {"question": "How many days of casual leave do I get?"}


def cited_sources(response):
    return {c["source"] for c in response.json()["citations"]}


# -- basics --------------------------------------------------------------------------------------


def test_health_ready_and_agent_listing(make_config):
    agent = agent_for(make_config())
    with client_for(agent) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").json() == {"ready": True}
        listing = client.get("/v1/agents", headers=bearer(API_KEY)).json()
        assert listing == {"agents": [{"id": "test-agent", "name": "Test Agent"}]}


def test_not_ready_until_the_index_exists(make_config):
    kb = KnowledgeBase(make_config())  # never ingested
    client = client_for(Agent(kb, make_config(), llm=ScriptedLLM(answering_reply)))
    assert client.get("/readyz").status_code == 503


def test_startup_ingest_builds_the_index(make_config):
    kb = KnowledgeBase(make_config())
    agent = Agent(kb, make_config(), llm=ScriptedLLM(answering_reply))
    with TestClient(create_app(agent, auth=ApiKeyAuth([API_KEY]))) as client:
        assert client.get("/readyz").status_code == 200
        response = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY))
        assert response.status_code == 200


def test_ask_returns_a_verified_answer_with_metadata(make_config):
    with client_for(agent_for(make_config())) as client:
        response = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY))
    body = response.json()
    assert response.status_code == 200 and not body["abstained"]
    assert "12 days" in body["text"] and body["citations"][0]["source"] == "leave.md"
    assert body["usage"]["llm_calls"] >= 1 and set(body["usage"]) == {
        "input_tokens",
        "output_tokens",
        "llm_calls",
    }
    assert body["request_id"] == response.headers["x-request-id"]
    assert response.headers["cache-control"] == "no-store"
    assert "debug" not in body or body["debug"] is None


def test_history_is_passed_to_the_agent(make_config):
    llm = ScriptedLLM(answering_reply)
    payload = {
        **LEAVE_Q,
        "history": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ],
    }
    with client_for(agent_for(make_config(), llm)) as client:
        assert (
            client.post(
                "/v1/agents/test-agent/ask", json=payload, headers=bearer(API_KEY)
            ).status_code
            == 200
        )
    assert any("hello" in m.content for messages, _ in llm.calls for m in messages)


# -- authentication ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        bearer("wrong-key-0123456789abcdef"),
        {"X-API-Key": "nope"},
        {"Authorization": "Basic abc"},
    ],
)
def test_requests_without_a_valid_api_key_are_rejected(make_config, headers):
    with client_for(agent_for(make_config())) as client:
        for method, path in [("post", "/v1/agents/test-agent/ask"), ("get", "/v1/agents")]:
            kwargs = {"json": LEAVE_Q} if method == "post" else {}
            response = getattr(client, method)(path, headers=headers, **kwargs)
            assert response.status_code == 401 and response.headers["www-authenticate"] == "Bearer"
            assert response.json()["error"]["code"] == "unauthorized"


def test_the_api_key_can_come_in_a_custom_header(make_config):
    with client_for(agent_for(make_config())) as client:
        ok = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers={"X-API-Key": API_KEY})
        assert ok.status_code == 200


def test_unknown_agents_are_404_only_after_authentication(make_config):
    with client_for(agent_for(make_config())) as client:
        assert client.post("/v1/agents/nope/ask", json=LEAVE_Q).status_code == 401  # no enumeration
        missing = client.post("/v1/agents/nope/ask", json=LEAVE_Q, headers=bearer(API_KEY))
        assert missing.status_code == 404 and missing.json()["error"]["code"] == "not_found"


def test_api_keys_must_be_strong_and_present():
    with pytest.raises(ConfigError, match="16 characters"):
        ApiKeyAuth(["short"])
    with pytest.raises(ConfigError, match="at least one"):
        ApiKeyAuth([" ", ""])


def test_no_auth_is_refused_for_access_controlled_agents(secured_config, make_config):
    with pytest.raises(ConfigError, match="no authentication"):
        create_app(agent_for(secured_config), auth=NoAuth())
    assert (
        client_for(agent_for(make_config()), auth=NoAuth())
        .post("/v1/agents/test-agent/ask", json=LEAVE_Q)
        .status_code
        == 200
    )


def test_a_custom_authenticator_can_reject_or_identify(secured_config):
    def check(headers):
        if headers.get("x-session") != "good":
            raise AuthError("Please sign in.")
        return RequestContext(roles=["hr"])

    with client_for(agent_for(secured_config), auth=CustomAuth(check)) as client:
        denied = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q)
        assert denied.status_code == 401 and denied.json()["error"]["message"] == "Please sign in."
        ok = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers={"x-session": "good"})
        assert "leave.md" in cited_sources(ok)


# -- identity reaches access control -------------------------------------------------------------


def test_jwt_identity_decides_what_the_caller_may_read(secured_config):
    auth = JwtAuth(JWT_SECRET)
    with client_for(agent_for(secured_config), auth=auth) as client:
        employee = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(token()))
        hr = client.post(
            "/v1/agents/test-agent/ask",
            json=LEAVE_Q,
            headers=bearer(token({"roles": ["hr"]})),
        )
    assert employee.status_code == 200 and "leave.md" not in cited_sources(employee)
    assert "12 days" not in employee.json()["text"]
    assert "leave.md" in cited_sources(hr)


def test_the_body_cannot_claim_an_identity_by_default(secured_config):
    auth = JwtAuth(JWT_SECRET)
    payload = {**LEAVE_Q, "context": {"roles": ["hr"]}}
    with client_for(agent_for(secured_config), auth=auth) as client:
        response = client.post("/v1/agents/test-agent/ask", json=payload, headers=bearer(token()))
    assert (
        response.status_code == 400 and response.json()["error"]["code"] == "context_not_accepted"
    )


def test_a_trusted_backend_may_send_the_end_users_identity(secured_config):
    auth = ApiKeyAuth([API_KEY], trust_caller_context=True)
    with client_for(agent_for(secured_config), auth=auth) as client:
        hr = client.post(
            "/v1/agents/test-agent/ask",
            json={**LEAVE_Q, "context": {"roles": ["hr"]}},
            headers=bearer(API_KEY),
        )
        employee = client.post(
            "/v1/agents/test-agent/ask",
            json={**LEAVE_Q, "context": {"roles": ["employee"]}},
            headers=bearer(API_KEY),
        )
        anonymous = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY))
    assert "leave.md" in cited_sources(hr) and "leave.md" not in cited_sources(employee)
    assert "leave.md" not in cited_sources(
        anonymous
    )  # no identity sent: sees only shared documents


def test_an_api_key_without_identity_sees_only_shared_documents(secured_config):
    with client_for(agent_for(secured_config)) as client:  # plain key: no roles at all
        response = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY))
    assert response.status_code == 200 and "leave.md" not in cited_sources(response)


# -- JWT verification ----------------------------------------------------------------------------


def jwt_client(make_config, **auth_kwargs):
    return client_for(agent_for(make_config()), auth=JwtAuth(JWT_SECRET, **auth_kwargs))


def ask(client, tok):
    return client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(tok))


def test_jwt_accepts_a_good_token_and_maps_claims(make_config):
    auth = JwtAuth(
        JWT_SECRET, attribute_claims={"region": "reg"}, roles_claim="groups", tenant_claim="org"
    )
    claims = jwt.decode(
        token({"org": "acme", "groups": "hr, manager", "reg": "eu"}),
        JWT_SECRET,
        algorithms=["HS256"],
    )
    context = auth._context(claims)
    assert (context.tenant_id, context.user_id) == ("acme", "u1")
    assert context.roles == ["hr", "manager"] and context.attributes == {"region": "eu"}
    with jwt_client(make_config) as client:
        assert ask(client, token()).status_code == 200


@pytest.mark.parametrize(
    "bad",
    [
        lambda: token({"exp": time.time() - 3600}),  # expired
        lambda: token(secret="x" * 40),  # wrong signature
        lambda: token(drop=("exp",)),  # no expiry: never valid
        lambda: token(secret=JWT_SECRET, algorithm="HS512"),  # not the configured algorithm
        lambda: jwt.encode(
            {"sub": "u", "exp": time.time() + 300}, "", algorithm="none"
        ),  # unsigned
        lambda: "not.a.jwt",
        lambda: "",
    ],
    ids=["expired", "wrong-signature", "no-exp", "wrong-algorithm", "alg-none", "garbage", "empty"],
)
def test_jwt_rejects_bad_tokens(make_config, bad):
    with jwt_client(make_config) as client:
        response = ask(client, bad())
    assert response.status_code == 401
    assert response.json()["error"]["message"] in {
        "Invalid or expired token.",
        "A bearer token is required.",
    }


def test_jwt_audience_and_issuer_are_enforced_when_configured(make_config):
    with jwt_client(make_config, audience="kb", issuer="idp") as client:
        assert ask(client, token({"aud": "kb", "iss": "idp"})).status_code == 200
        assert ask(client, token({"aud": "other", "iss": "idp"})).status_code == 401
        assert ask(client, token({"aud": "kb", "iss": "evil"})).status_code == 401
        assert ask(client, token()).status_code == 401  # claims required, not optional


def test_jwt_secrets_and_algorithms_are_validated_up_front():
    with pytest.raises(ConfigError, match="at least 32 bytes"):
        JwtAuth("too-short")
    with pytest.raises(ConfigError, match="never accepted"):
        JwtAuth(JWT_SECRET, algorithm="none")


def test_rs256_verification_and_algorithm_confusion(make_config):
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    private_pem = private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    agent = agent_for(make_config())
    with client_for(agent, auth=JwtAuth(public_pem, algorithm="RS256")) as client:
        assert ask(client, token(secret=private_pem, algorithm="RS256")).status_code == 200

        # classic attack: an HS256 token whose HMAC key is the (public) PEM the server verifies with.
        # PyJWT refuses to build this, so forge it by hand.
        def b64(data):
            return base64.urlsafe_b64encode(data).rstrip(b"=")

        head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
        body = b64(json.dumps({"sub": "attacker", "exp": time.time() + 300}).encode())
        signature = hmac.new(public_pem.encode(), head + b"." + body, hashlib.sha256).digest()
        forged = (head + b"." + body + b"." + b64(signature)).decode()
        assert ask(client, forged).status_code == 401


# -- request validation and error hygiene --------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "x" * 4001},
        {"question": "ok", "extra": 1},
        {"question": "ok", "history": [{"role": "system", "content": "x"}]},
        {"question": "ok", "history": [{"role": "user", "content": "x"}] * 21},
        {"question": "ok", "history": [{"role": "user", "content": "x" * 11000}] * 2},
    ],
)
def test_malformed_requests_are_rejected_without_echoing_input(make_config, payload):
    with client_for(agent_for(make_config())) as client:
        response = client.post("/v1/agents/test-agent/ask", json=payload, headers=bearer(API_KEY))
    assert response.status_code == 422 and response.json()["error"]["code"] == "invalid_request"
    assert "x" * 50 not in response.text


class Exploding(ScriptedLLM):
    def __init__(self, error):
        super().__init__(answering_reply)
        self.error = error

    async def generate(self, messages, **kwargs):
        raise self.error


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (RuntimeError("db password is hunter2"), 500, "internal_error"),
        (KbsdkError("internal path C:/secret/index"), 500, "internal_error"),
        (AccessDeniedError("policy detail: tenant_id required"), 403, "access_denied"),
        (BudgetExceededError("spent $4.99 of $5"), 429, "budget_exceeded"),
    ],
)
def test_failures_map_to_generic_errors_and_leak_nothing(make_config, caplog, error, status, code):
    caplog.set_level(logging.INFO, logger="kbsdk.server")
    with client_for(agent_for(make_config(), Exploding(error))) as client:
        response = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY))
    assert response.status_code == status and response.json()["error"]["code"] == code
    for secret in ("hunter2", "C:/secret", "tenant_id", "$4.99"):
        assert secret not in response.text
    assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


def test_slow_requests_time_out(make_config):
    class Slow(ScriptedLLM):
        async def generate(self, messages, **kwargs):
            await asyncio.sleep(2)
            return await super().generate(messages, **kwargs)

    agent = agent_for(make_config(), Slow(answering_reply))
    with client_for(agent, request_timeout_s=0.1) as client:
        response = client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY))
    assert response.status_code == 504 and response.json()["error"]["code"] == "timeout"


async def test_the_concurrency_cap_limits_questions_in_flight(make_config):
    class Tracking(ScriptedLLM):
        active = peak = 0

        async def generate(self, messages, **kwargs):
            type(self).active += 1
            type(self).peak = max(type(self).peak, type(self).active)
            await asyncio.sleep(0.05)
            try:
                return await super().generate(messages, **kwargs)
            finally:
                type(self).active -= 1

    agent = agent_for(make_config(), Tracking(answering_reply))
    app = create_app(
        agent, auth=ApiKeyAuth([API_KEY]), ingest_on_startup=False, max_concurrent_requests=1
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            *(
                client.post("/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY))
                for _ in range(4)
            )
        )
    assert all(r.status_code == 200 for r in responses)
    assert Tracking.peak == 1


def test_question_and_answer_text_never_reach_the_logs(make_config, caplog):
    caplog.set_level(logging.DEBUG)
    with client_for(agent_for(make_config())) as client:
        client.post(
            "/v1/agents/test-agent/ask",
            json={"question": "zebra-secret-question casual leave?"},
            headers=bearer(API_KEY),
        )
    assert "zebra-secret-question" not in caplog.text
    assert "request " in caplog.text  # but the request itself is logged, with its id


# -- admin, CORS, docs, debug ----------------------------------------------------------------------


def test_ingest_is_disabled_without_an_admin_key(make_config):
    with client_for(agent_for(make_config())) as client:
        response = client.post("/v1/agents/test-agent/ingest", headers=bearer(API_KEY))
    assert response.status_code == 403 and response.json()["error"]["code"] == "admin_disabled"


def test_ingest_needs_the_admin_key_and_reindexes(make_config, docs_dir):
    agent = agent_for(make_config())
    with client_for(agent, admin_key=ADMIN_KEY) as client:
        url = "/v1/agents/test-agent/ingest"
        assert (
            client.post(url, headers=bearer(API_KEY)).status_code == 401
        )  # a user key is not enough
        assert (
            client.post(url, headers={"X-Admin-Key": "wrong-wrong-wrong-1234"}).status_code == 401
        )
        (Path(docs_dir) / "new.md").write_text(
            "# Perks\n\nFree coffee on Fridays.\n", encoding="utf-8"
        )
        report = client.post(url, headers={"X-Admin-Key": ADMIN_KEY})
        assert report.status_code == 200
        assert report.json()["files_new"] == 1 and report.json()["chunks_added"] >= 1
        assert (
            client.post("/v1/agents/nope/ingest", headers={"X-Admin-Key": ADMIN_KEY}).status_code
            == 404
        )


def test_a_short_admin_key_is_refused(make_config):
    with pytest.raises(ConfigError, match="admin key"):
        create_app(agent_for(make_config()), auth=NoAuth(), admin_key="short")


def test_concurrent_ingests_are_refused_not_stacked(make_config):
    agent = agent_for(make_config())
    app = create_app(
        agent, auth=ApiKeyAuth([API_KEY]), admin_key=ADMIN_KEY, ingest_on_startup=False
    )
    started = None

    async def scenario():
        nonlocal started
        started = asyncio.Event()
        original = agent.kb.aingest

        async def slow_ingest():
            started.set()
            await asyncio.sleep(0.2)
            return await original()

        agent.kb.aingest = slow_ingest
        headers = {"X-Admin-Key": ADMIN_KEY}
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            first = asyncio.create_task(
                client.post("/v1/agents/test-agent/ingest", headers=headers)
            )
            await started.wait()
            second = await client.post("/v1/agents/test-agent/ingest", headers=headers)
            return second, await first

    second, first = asyncio.run(scenario())
    assert second.status_code == 409 and second.json()["error"]["code"] == "ingest_in_progress"
    assert first.status_code == 200


def test_cors_is_off_unless_configured(make_config):
    origin = {"Origin": "https://intranet.example", "Access-Control-Request-Method": "POST"}
    with client_for(agent_for(make_config())) as client:
        assert (
            "access-control-allow-origin"
            not in client.options("/v1/agents/test-agent/ask", headers=origin).headers
        )
    with client_for(agent_for(make_config()), cors_origins=["https://intranet.example"]) as client:
        allowed = client.options("/v1/agents/test-agent/ask", headers=origin)
        assert allowed.headers["access-control-allow-origin"] == "https://intranet.example"
        other = client.options(
            "/v1/agents/test-agent/ask", headers={**origin, "Origin": "https://evil.example"}
        )
        assert "access-control-allow-origin" not in other.headers


def test_docs_are_hidden_by_default(make_config):
    with client_for(agent_for(make_config())) as client:
        assert (
            client.get("/docs").status_code == 404
            and client.get("/openapi.json").status_code == 404
        )
    with client_for(agent_for(make_config()), enable_docs=True) as client:
        assert client.get("/openapi.json").status_code == 200


def test_debug_details_only_when_explicitly_exposed(make_config):
    with client_for(agent_for(make_config()), expose_debug=True) as client:
        body = client.post(
            "/v1/agents/test-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY)
        ).json()
    assert body["debug"]["retrieved"] and body["debug"]["trace"]


def test_agent_names_must_be_unique(make_config):
    with pytest.raises(ConfigError, match="share the name"):
        create_app([agent_for(make_config()), agent_for(make_config())], auth=NoAuth())
    with pytest.raises(ConfigError, match="at least one"):
        create_app([], auth=NoAuth())


def test_several_agents_are_hosted_side_by_side(make_config):
    first, second = agent_for(make_config()), agent_for(make_config(name="Second Agent"))
    app = create_app([first, second], auth=ApiKeyAuth([API_KEY]), ingest_on_startup=False)
    with TestClient(app) as client:
        ids = [a["id"] for a in client.get("/v1/agents", headers=bearer(API_KEY)).json()["agents"]]
        assert ids == ["test-agent", "second-agent"]
        assert (
            client.post(
                "/v1/agents/second-agent/ask", json=LEAVE_Q, headers=bearer(API_KEY)
            ).status_code
            == 200
        )


def test_a_missing_web_stack_gives_an_install_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "fastapi", None)
    monkeypatch.delitem(sys.modules, "kbsdk.server.app", raising=False)
    with pytest.raises(KbsdkError, match=r"pip install 'kbsdk\[server\]'"):
        create_app([], auth=NoAuth())


# -- CLI -----------------------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    config = tmp_path / "agent.yaml"
    config.write_text(
        "name: Served Agent\n"
        "embedder: {provider: hashing}\n"
        "knowledge:\n  sources: [{location: ./docs}]\n  persist_dir: ./idx\n"
        "generation: {llm: {provider: scripted}}\n",
        encoding="utf-8",
    )
    return tmp_path, str(config)


@pytest.fixture
def served(monkeypatch):
    """Capture the app the CLI would run instead of starting uvicorn."""
    captured = {}
    monkeypatch.setattr("kbsdk.server.serve", lambda app, **kw: captured.update(app=app, **kw))
    return captured


def run(capsys, *argv):
    code = main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


def test_cli_serve_with_api_keys(project, served, monkeypatch, capsys):
    _, config = project
    monkeypatch.setenv("KBSDK_API_KEYS", f"{API_KEY}, second-key-0123456789abcdef")
    monkeypatch.setenv("KBSDK_ADMIN_KEY", ADMIN_KEY)
    code, _, err = run(capsys, "serve", "-c", config, "--auth", "apikey", "--port", "9123")
    assert code == 0 and served["port"] == 9123 and served["host"] == "127.0.0.1"
    assert "auth: apikey" in err and "admin endpoints: on" in err
    with TestClient(served["app"]) as client:  # ingest_on_startup builds the index
        ok = client.post(
            "/v1/agents/served-agent/ask",
            json=LEAVE_Q,
            headers=bearer("second-key-0123456789abcdef"),
        )
        assert ok.status_code == 200
        assert client.post("/v1/agents/served-agent/ask", json=LEAVE_Q).status_code == 401
        assert (
            client.post(
                "/v1/agents/served-agent/ingest", headers={"X-Admin-Key": ADMIN_KEY}
            ).status_code
            == 200
        )


def test_cli_serve_needs_secrets_from_the_environment(project, served, monkeypatch, capsys):
    _, config = project
    monkeypatch.delenv("KBSDK_API_KEYS", raising=False)
    code, _, err = run(capsys, "serve", "-c", config, "--auth", "apikey")
    assert code == 2 and "KBSDK_API_KEYS" in err and not served
    monkeypatch.delenv("KBSDK_JWT_SECRET", raising=False)
    code, _, err = run(capsys, "serve", "-c", config, "--auth", "jwt")
    assert code == 2 and "KBSDK_JWT_SECRET" in err


def test_cli_serve_with_jwt_maps_claims_and_checks_audience(project, served, monkeypatch, capsys):
    _, config = project
    monkeypatch.setenv("KBSDK_JWT_SECRET", JWT_SECRET)
    code, _, _ = run(capsys, "serve", "-c", config, "--auth", "jwt", "--jwt-audience", "kb")
    assert code == 0
    url = "/v1/agents/served-agent/ask"
    with TestClient(served["app"]) as client:
        good = client.post(url, json=LEAVE_Q, headers=bearer(token({"aud": "kb"})))
        assert good.status_code == 200
        assert (
            client.post(url, json=LEAVE_Q, headers=bearer(token({"aud": "x"}))).status_code == 401
        )
        assert client.post(url, json=LEAVE_Q, headers=bearer(token())).status_code == 401


def test_cli_refuses_unsafe_combinations(project, served, monkeypatch, capsys):
    _, config = project
    code, _, err = run(capsys, "serve", "-c", config, "--auth", "none", "--host", "0.0.0.0")
    assert code == 2 and "local development only" in err
    monkeypatch.setenv("KBSDK_API_KEYS", API_KEY)
    code, _, err = run(
        capsys, "serve", "-c", config, "--auth", "apikey", "--host", "0.0.0.0", "--debug-responses"
    )
    assert code == 2 and "localhost" in err
    monkeypatch.setenv("KBSDK_JWT_SECRET", JWT_SECRET)
    code, _, err = run(capsys, "serve", "-c", config, "--auth", "jwt", "--trust-caller-context")
    assert code == 2 and "apikey" in err
    assert not served


def test_cli_serve_none_is_refused_for_access_controlled_agents(project, served, capsys):
    root, config = project
    text = (
        Path(config)
        .read_text(encoding="utf-8")
        .replace(
            "sources: [{location: ./docs}]",
            "sources: [{location: ./docs, metadata: {roles: ['*']}}]",
        )
    )
    Path(config).write_text(text + "access: {roles_field: roles}\n", encoding="utf-8")
    code, _, err = run(capsys, "serve", "-c", config, "--auth", "none")
    assert code == 2 and "no authentication" in err
