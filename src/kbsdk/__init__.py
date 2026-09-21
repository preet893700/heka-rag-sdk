"""kbsdk - a configurable, plug-and-play RAG SDK (placeholder name)."""

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

__version__ = "0.0.1"

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
