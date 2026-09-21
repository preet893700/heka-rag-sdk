"""heka-rag-sdk - a configurable, plug-and-play RAG SDK."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from heka.rag import adapters as _adapters  # noqa: F401  (registers the built-in adapters)
from heka.rag.agent import Agent
from heka.rag.config import ComponentConfig, RAGConfig, available_presets
from heka.rag.env import load_env_file
from heka.rag.errors import (
    AccessDeniedError,
    BudgetExceededError,
    ConfigError,
    HekaRagError,
    MissingExtraError,
    UnknownProviderError,
)
from heka.rag.knowledge_base import KnowledgeBase
from heka.rag.pipelines.ingest import IngestReport
from heka.rag.registry import register, register_lazy, registry
from heka.rag.types import (
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
    __version__ = _pkg_version("heka-rag-sdk")
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
    "HekaRagError",
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
