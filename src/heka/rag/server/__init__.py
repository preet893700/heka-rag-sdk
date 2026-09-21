"""REST server for heka-rag-sdk agents. Install with `pip install 'heka-rag-sdk[server]'`.

    from heka.rag import RAGConfig
    from heka.rag.server import JwtAuth, create_app

    app = create_app(RAGConfig.from_file("hr.yaml"), auth=JwtAuth(os.environ["JWT_SECRET"]))
    # run with:  uvicorn module:app        or:  heka-rag serve -c hr.yaml --auth jwt

The authenticators need no web framework; only `create_app` and `serve` do.
"""

from __future__ import annotations

from typing import Any

from heka.rag.errors import HekaRagError
from heka.rag.server.auth import (
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

_HINT = "The REST server needs optional dependencies. Install them with: pip install 'heka-rag-sdk[server]'"


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Build the ASGI application; see `heka.rag.server.app.create_app` for the parameters."""
    try:
        from heka.rag.server.app import create_app as build
    except ImportError as exc:
        raise HekaRagError(f"{_HINT} (missing: {exc.name})") from exc
    return build(*args, **kwargs)


def serve(app: Any, *, host: str = "127.0.0.1", port: int = 8000, log_level: str = "info") -> None:
    """Run `app` with uvicorn (blocking)."""
    try:
        import uvicorn
    except ImportError as exc:
        raise HekaRagError(f"{_HINT} (missing: {exc.name})") from exc
    uvicorn.run(app, host=host, port=port, log_level=log_level)
