"""kbsdk - a configurable, plug-and-play RAG SDK (placeholder name)."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from kbsdk import adapters as _adapters  # noqa: F401  (registers the built-in adapters)
from kbsdk.agent import Agent
from kbsdk.config import ComponentConfig, RAGConfig, available_presets
from kbsdk.env import load_env_file
from kbsdk.errors import (
    AccessDeniedError,
    BudgetExceededError,
    ConfigError,
    KbsdkError,
    MissingExtraError,
    UnknownProviderError,
)
from kbsdk.knowledge_base import KnowledgeBase
from kbsdk.pipelines.ingest import IngestReport
from kbsdk.registry import register, register_lazy, registry
from kbsdk.types import (
    Answer,
    Chunk,
    Citation,
    Document,
    Filter,
    Message,
    RequestContext,
    ScoredChunk,
    Usage,
)

try:  # one source of truth: the version in pyproject.toml, as installed
    __version__ = _pkg_version("kbsdk")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "AccessDeniedError",
    "Agent",
    "BudgetExceededError",
    "Answer",
    "Chunk",
    "Citation",
    "ComponentConfig",
    "ConfigError",
    "Document",
    "Filter",
    "IngestReport",
    "KbsdkError",
    "KnowledgeBase",
    "Message",
    "MissingExtraError",
    "RAGConfig",
    "RequestContext",
    "ScoredChunk",
    "UnknownProviderError",
    "Usage",
    "__version__",
    "available_presets",
    "load_env_file",
    "register",
    "register_lazy",
    "registry",
]
