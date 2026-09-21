"""REST server for kbsdk agents. Install with `pip install 'kbsdk[server]'`.

    from kbsdk import RAGConfig
    from kbsdk.server import JwtAuth, create_app

    app = create_app(RAGConfig.from_file("hr.yaml"), auth=JwtAuth(os.environ["JWT_SECRET"]))
    # run with:  uvicorn module:app        or:  kbsdk serve -c hr.yaml --auth jwt

The authenticators need no web framework; only `create_app` and `serve` do.
"""

from __future__ import annotations

from typing import Any

from kbsdk.errors import KbsdkError
from kbsdk.server.auth import (
    ApiKeyAuth,
    Authenticator,
    AuthError,
    CustomAuth,
    JwtAuth,
    NoAuth,
)

__all__ = [
    "ApiKeyAuth",
    "AuthError",
    "Authenticator",
    "CustomAuth",
    "JwtAuth",
    "NoAuth",
    "create_app",
    "serve",
]

_HINT = (
    "The REST server needs optional dependencies. Install them with: pip install 'kbsdk[server]'"
)


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Build the ASGI application; see `kbsdk.server.app.create_app` for the parameters."""
    try:
        from kbsdk.server.app import create_app as build
    except ImportError as exc:
        raise KbsdkError(f"{_HINT} (missing: {exc.name})") from exc
    return build(*args, **kwargs)


def serve(app: Any, *, host: str = "127.0.0.1", port: int = 8000, log_level: str = "info") -> None:
    """Run `app` with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise KbsdkError(f"{_HINT} (missing: {exc.name})") from exc
    uvicorn.run(app, host=host, port=port, log_level=log_level)
