"""The configuration schema. One `RAGConfig` fully describes an agent.

Every pluggable stage is a `ComponentConfig` (`provider` + `params`), so any choice can be written
in Python or in a YAML/TOML/JSON file, stored as data, and validated before anything runs. A config
may start from a named preset and override only what differs:

    RAGConfig.preset("free-tier-dev", knowledge={"sources": [{"location": "./docs/hr"}]})
"""

from __future__ import annotations

import json
import os
import re
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from kbsdk.errors import ConfigError
from kbsdk.types import Filter

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised on Python 3.10 only
    import tomli as tomllib

# Parameter names that look like they hold a secret. Keys must come from the environment; configs
# are meant to be stored, shared and committed. Use `api_key_env: MY_ENV_VAR` to name the variable.
_SECRET_NAME = re.compile(
    r"^(.*[_-])?(api[_-]?key|secret([_-]?key)?|password|passwd|credentials?"
    r"|(access|auth|bearer)[_-]?token|token)$",
    re.IGNORECASE,
)


def _reject_secrets(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and _SECRET_NAME.match(key):
                raise ValueError(
                    f"{path}.{key}: secrets must not be stored in configuration. "
                    f"Read them from the environment instead, e.g. '{key}_env: MY_ENV_VAR'."
                )
            _reject_secrets(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, f"{path}[{index}]")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ComponentConfig(_Model):
    """One pluggable choice: which provider, and its settings."""

    provider: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _no_secrets(self) -> ComponentConfig:
        _reject_secrets(self.params, f"{self.provider}.params")
        return self


class SourceConfig(_Model):
    location: str  # directory, glob, file, URL or connector URI (e.g. "sharepoint://site/library")
    loader: str | None = None  # force a loader provider; default is chosen by extension / scheme
    # Glob patterns (relative to the source folder) to leave out, e.g. ["code-of-conduct.md", "drafts/*"].
    # Use this to carve a restricted file out of a broader source instead of listing it twice.
    exclude: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)  # attached to every document from here

    @model_validator(mode="after")
    def _no_secrets(self) -> SourceConfig:
        _reject_secrets(self.params, f"source[{self.location}].params")
        return self


class KnowledgeConfig(_Model):
    sources: list[SourceConfig] = Field(default_factory=list)
    persist_dir: str = ".kbsdk"
    update_mode: Literal["rebuild", "incremental", "versioned"] = "incremental"
    extraction: Literal["text", "layout", "multimodal"] = "layout"  # how tables/figures are read
    ocr: ComponentConfig | None = None  # set to enable OCR for scanned pages and images


class AgenticConfig(_Model):
    """Multi-step retrieval for questions that span documents. Each step beyond the first costs one
    model call (to judge whether more is needed) plus one search."""

    max_steps: int = Field(default=3, ge=2, le=8)  # searches per question, including the first


class RetrievalConfig(_Model):
    # Only "dense" is built so far; "hybrid" becomes the default once it lands in Phase 2.
    mode: Literal["dense", "sparse", "hybrid"] = "dense"
    top_k: int = Field(default=20, ge=1)  # candidates fetched from the store
    final_k: int = Field(default=5, ge=1)  # chunks kept after reranking, passed to the model
    fusion: Literal["rrf", "weighted"] = "rrf"  # how dense and sparse results are combined
    # RRF damping constant (also used to fuse multi-query lists)
    rrf_k: int = Field(default=60, ge=1)
    dense_weight: float = Field(default=0.5, ge=0.0, le=1.0)  # hybrid: share given to dense results
    # rewrite, multi-query, HyDE
    query_transforms: list[ComponentConfig] = Field(default_factory=list)
    reranker: ComponentConfig | None = None
    agentic: AgenticConfig | None = None  # multi-step retrieval (needs the `agentic` extra)
    # Turn "what about contractors?" into a standalone query. Arrives in Phase 2, so off for now.
    condense_followups: bool = False
    filter: Filter | None = None  # static metadata filter applied to every query

    @model_validator(mode="after")
    def _final_within_top(self) -> RetrievalConfig:
        if self.final_k > self.top_k:
            raise ValueError(f"final_k ({self.final_k}) cannot exceed top_k ({self.top_k})")
        return self


class CitationConfig(_Model):
    # "native" uses the provider's own citation feature
    mode: Literal["sdk", "native", "off"] = "sdk"
    require_verified: bool = True  # drop or flag citations whose quote is not found in the chunk


class AbstentionConfig(_Model):
    enabled: bool = True
    min_top_score: float | None = None  # abstain if the best retrieved score is below this
    min_chunks: int = Field(default=1, ge=0)
    message: str = "I couldn't find this in the available documents, so I can't answer it reliably."


class GenerationConfig(_Model):
    llm: ComponentConfig  # required: there is no sensible default without credentials
    citations: CitationConfig = Field(default_factory=CitationConfig)
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    verify_answer: bool = False  # second pass: check each claim is supported by its cited text
    # What to do with an answer the verification pass says is not fully supported:
    # "abstain" declines it; "flag" returns it with the unsupported claims listed in `warnings`.
    verify_action: Literal["abstain", "flag"] = "abstain"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)


class AccessConfig(_Model):
    """Who may see which chunks. Enforced inside retrieval, so neither the model nor a caller-supplied
    filter can widen it. Chunks are tagged through `knowledge.sources[].metadata`.

    Fail-closed: a chunk missing a required tag is visible to nobody. Use the `shared_value` ("*") to
    mean "everyone", e.g. `allowed_roles: ["*"]` for a public policy.
    """

    # chunk metadata holding its tenant, matched to context.tenant_id
    tenant_field: str | None = None
    # chunk metadata list of roles; needs an overlap with context.roles
    roles_field: str | None = None
    # context attribute name -> chunk metadata field it must match (e.g. {"region": "region"})
    attribute_fields: dict[str, str] = Field(default_factory=dict)
    shared_value: str = "*"
    # asking without a RequestContext raises (True) instead of seeing only shared content
    require_context: bool = True

    @property
    def enabled(self) -> bool:
        return bool(self.tenant_field or self.roles_field or self.attribute_fields)


class ModelPrice(_Model):
    """US dollars per million tokens. Supplied by you: prices change and differ per contract."""

    input_per_mtok: float = Field(ge=0.0)
    output_per_mtok: float = Field(ge=0.0)


class BudgetConfig(_Model):
    max_total_usd: float | None = Field(default=None, gt=0)  # stop once this much has been spent
    max_llm_calls_per_question: int | None = Field(default=None, ge=1)  # guards runaway loops


class ReliabilityConfig(_Model):
    max_retries: int = Field(default=3, ge=0)
    requests_per_minute: int | None = Field(default=None, ge=1)  # client-side throttle
    timeout_s: float = Field(default=60.0, gt=0)
    fallback_llms: list[ComponentConfig] = Field(default_factory=list)


class EvalConfig(_Model):
    judge_llm: ComponentConfig | None = None  # defaults to the generation model when unset
    metrics: list[ComponentConfig] = Field(default_factory=list)  # empty = the built-in set
    thresholds: dict[str, float] = Field(default_factory=dict)  # metric name -> minimum, for gating
    sample_size: int | None = Field(default=None, ge=1)  # judge only a sample (saves cost / quota)


def _default_embedder() -> ComponentConfig:
    return ComponentConfig(provider="local")


def _default_store() -> ComponentConfig:
    return ComponentConfig(provider="local")


def _default_chunker() -> ComponentConfig:
    return ComponentConfig(provider="structure_aware")


class RAGConfig(_Model):
    """Complete description of one agent."""

    config_version: Literal[1] = 1
    name: str = "agent"
    instructions: str = ""  # domain guidance and tone, added to the SDK's built-in grounding rules
    # Opt-in: a .env file the SDK loads (existing environment variables win). Never read implicitly.
    env_file: str | None = None
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    chunker: ComponentConfig = Field(default_factory=_default_chunker)
    embedder: ComponentConfig = Field(default_factory=_default_embedder)
    store: ComponentConfig = Field(default_factory=_default_store)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig
    access: AccessConfig = Field(default_factory=AccessConfig)
    guardrails: list[ComponentConfig] = Field(default_factory=list)
    cache: ComponentConfig | None = None
    tracing: list[ComponentConfig] = Field(default_factory=list)
    # "<provider>:<model>" (as reported by the LLM's name) -> price. Without an entry, cost is unknown.
    pricing: dict[str, ModelPrice] = Field(default_factory=dict)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    reliability: ReliabilityConfig = Field(default_factory=ReliabilityConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)

    # -- construction -------------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGConfig:
        """Build a config; a top-level `preset: <name>` key is used as the base and merged over."""
        data = dict(data)
        preset_name = data.pop("preset", None)
        if preset_name is not None:
            data = deep_merge(load_preset(str(preset_name)), data)
        return cls.model_validate(data)

    @classmethod
    def preset(cls, preset_name: str, /, **overrides: Any) -> RAGConfig:
        # Positional-only so that config fields such as `name=` can be passed as overrides.
        return cls.from_dict({"preset": preset_name, **overrides})

    @classmethod
    def from_file(cls, path: str | Path) -> RAGConfig:
        path = Path(path)
        try:
            if path.suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
            elif path.suffix == ".toml":
                with path.open("rb") as handle:
                    data = tomllib.load(handle)
            elif path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
            else:
                raise ConfigError(
                    f"Unsupported config format {path.suffix!r} (use .yaml, .toml or .json)"
                )
        except OSError as exc:
            raise ConfigError(f"Cannot read config file {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"{path} must contain a mapping at the top level")
        return _resolve_paths(cls.from_dict(data), path.resolve().parent)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def json_schema(cls) -> dict[str, Any]:
        schema: dict[str, Any] = cls.model_json_schema()
        return schema


def _resolve_paths(config: RAGConfig, base: Path) -> RAGConfig:
    """Make relative paths in a config *file* relative to that file, not to the working directory."""

    def resolve(value: str) -> str:
        if "://" in value or Path(value).is_absolute():
            return value
        return os.path.normpath(base / value)

    knowledge = config.knowledge.model_copy(
        update={
            "persist_dir": resolve(config.knowledge.persist_dir),
            "sources": [
                source.model_copy(update={"location": resolve(source.location)})
                for source in config.knowledge.sources
            ],
        }
    )
    update: dict[str, Any] = {"knowledge": knowledge}
    if config.env_file:
        update["env_file"] = resolve(config.env_file)
    return config.model_copy(update=update)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge `override` into `base`. Dicts merge recursively; everything else is replaced.

    A component (a dict with a `provider` key) whose provider changes is replaced wholesale, so that
    switching from one provider to another never inherits the old provider's params.
    """
    merged = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            same_component = "provider" not in value or value.get("provider") == current.get(
                "provider"
            )
            merged[key] = deep_merge(current, value) if same_component else value
        else:
            merged[key] = value
    return merged


def available_presets() -> list[str]:
    root = resources.files("kbsdk.presets")
    return sorted(
        item.name.removesuffix(".yaml") for item in root.iterdir() if item.name.endswith(".yaml")
    )


def load_preset(name: str) -> dict[str, Any]:
    root = resources.files("kbsdk.presets")
    target = root.joinpath(f"{name}.yaml")
    if not target.is_file():
        raise ConfigError(
            f"Unknown preset {name!r}. Available presets: {', '.join(available_presets())}"
        )
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"Preset {name!r} is malformed")
    return data
