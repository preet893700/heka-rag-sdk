"""Who is calling? The server never takes identity from the request body by default.

Access control inside the SDK is only as trustworthy as the `RequestContext` it is given, so the
server derives it from something the caller cannot forge: a verified JWT, or your own authenticator.
An `Authenticator` looks at the request headers and returns the `RequestContext` for the caller, or
raises `AuthError` (HTTP 401). Nothing here imports the web framework.
"""

from __future__ import annotations

import hmac
import inspect
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any, Protocol

from heka.rag.errors import ConfigError
from heka.rag.types import RequestContext

MIN_HMAC_SECRET_BYTES = 32


class AuthError(Exception):
    """The credential is missing or invalid. The message is safe to show to the caller."""


class Authenticator(Protocol):
    #: True only when a trusted backend vouches for the end user and may send `context` in the body.
    trusts_body_context: bool

    async def authenticate(self, headers: Mapping[str, str]) -> RequestContext | None: ...


def bearer_token(headers: Mapping[str, str]) -> str | None:
    """The token from `Authorization: Bearer <token>` (header names are matched case-insensitively)."""
    value = next((v for k, v in headers.items() if k.lower() == "authorization"), "")
    scheme, _, token = value.partition(" ")
    return token.strip() or None if scheme.lower() == "bearer" else None


def _header(headers: Mapping[str, str], name: str) -> str | None:
    return next((v for k, v in headers.items() if k.lower() == name.lower()), None)


def secrets_match(candidate: str | None, allowed: Iterable[str]) -> bool:
    """Constant-time comparison against every allowed secret (no early exit on a match)."""
    if not candidate:
        return False
    matched = False
    for secret in allowed:
        matched |= hmac.compare_digest(candidate.encode("utf-8"), secret.encode("utf-8"))
    return matched


class NoAuth:
    """No credential required, and no identity: for local development only.

    Agents with access control enabled reject every request from a caller with no identity, and the
    server refuses to start with NoAuth for such agents or on a non-loopback address.
    """

    trusts_body_context = False

    async def authenticate(self, headers: Mapping[str, str]) -> RequestContext | None:
        return None


class ApiKeyAuth:
    """Shared secret(s) in `Authorization: Bearer <key>` or `X-API-Key`.

    An API key proves *which application* is calling, not which end user. With
    `trust_caller_context=True` the caller (your own backend, after it has authenticated the user) may
    send the end user's `context` in the request body; only enable that when the key is held by a
    server you control, never by a browser.
    """

    def __init__(self, keys: Iterable[str], *, trust_caller_context: bool = False) -> None:
        self.keys = [k for k in (key.strip() for key in keys) if k]
        if not self.keys:
            raise ConfigError("ApiKeyAuth needs at least one non-empty key.")
        weak = [k for k in self.keys if len(k) < 16]
        if weak:
            raise ConfigError(
                "API keys must be at least 16 characters (use a random 32+ character value)."
            )
        self.trusts_body_context = trust_caller_context

    async def authenticate(self, headers: Mapping[str, str]) -> RequestContext | None:
        presented = bearer_token(headers) or _header(headers, "x-api-key")
        if not secrets_match(presented, self.keys):
            raise AuthError("Invalid or missing API key.")
        return RequestContext()


class JwtAuth:
    """Verify a signed JWT (`Authorization: Bearer <jwt>`) and map its claims to a `RequestContext`.

    Only the configured algorithm is accepted (never "none", never a different family than the key
    you supplied), `exp` is required, and `audience`/`issuer` are checked when given. HS* needs a
    secret of at least 32 bytes; RS*/ES* take the issuer's public key in PEM form.
    """

    trusts_body_context = False

    def __init__(
        self,
        key: str,
        *,
        algorithm: str = "HS256",
        audience: str | None = None,
        issuer: str | None = None,
        tenant_claim: str | None = "tenant_id",
        user_claim: str | None = "sub",
        roles_claim: str | None = "roles",
        attribute_claims: Mapping[str, str] | None = None,
        leeway_s: int = 30,
    ) -> None:
        if algorithm.lower() == "none" or not algorithm:
            raise ConfigError('JwtAuth: an algorithm is required ("none" is never accepted).')
        if algorithm.upper().startswith("HS") and len(key.encode("utf-8")) < MIN_HMAC_SECRET_BYTES:
            raise ConfigError(
                f"JwtAuth: an HMAC secret must be at least {MIN_HMAC_SECRET_BYTES} bytes long."
            )
        try:
            import jwt
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ConfigError("JWT auth needs PyJWT: pip install 'heka-rag-sdk[server]'") from exc
        self._jwt = jwt
        self.key = key
        self.algorithm = algorithm
        self.audience = audience
        self.issuer = issuer
        self.tenant_claim = tenant_claim
        self.user_claim = user_claim
        self.roles_claim = roles_claim
        self.attribute_claims = dict(attribute_claims or {})
        self.leeway_s = leeway_s

    async def authenticate(self, headers: Mapping[str, str]) -> RequestContext | None:
        token = bearer_token(headers)
        if not token:
            raise AuthError("A bearer token is required.")
        required = ["exp"] + (["aud"] if self.audience else []) + (["iss"] if self.issuer else [])
        try:
            claims = self._jwt.decode(
                token,
                self.key,
                algorithms=[self.algorithm],
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway_s,
                options={"require": required},
            )
        except self._jwt.PyJWTError as exc:
            # Deliberately vague: which check failed is for the server log, not the caller.
            raise AuthError("Invalid or expired token.") from exc
        return self._context(claims)

    def _context(self, claims: Mapping[str, Any]) -> RequestContext:
        def text(claim: str | None) -> str | None:
            value = claims.get(claim) if claim else None
            return str(value) if isinstance(value, str | int) and str(value) else None

        roles: list[str] = []
        raw = claims.get(self.roles_claim) if self.roles_claim else None
        if isinstance(raw, str):
            roles = [r for r in raw.replace(",", " ").split() if r]
        elif isinstance(raw, list):
            roles = [str(r) for r in raw if isinstance(r, str | int)]
        attributes = {
            name: claims[claim] for name, claim in self.attribute_claims.items() if claim in claims
        }
        return RequestContext(
            tenant_id=text(self.tenant_claim),
            user_id=text(self.user_claim),
            roles=roles,
            attributes=attributes,
        )


class CustomAuth:
    """Wrap your own function: `(headers) -> RequestContext | None`, sync or async.

    Raise `AuthError` to reject the request. Use this to plug in an existing session or SSO check.
    """

    trusts_body_context = False

    def __init__(
        self,
        func: Callable[
            [Mapping[str, str]], RequestContext | None | Awaitable[RequestContext | None]
        ],
    ) -> None:
        self.func = func

    async def authenticate(self, headers: Mapping[str, str]) -> RequestContext | None:
        result = self.func(headers)
        if inspect.isawaitable(result):
            result = await result
        return result
