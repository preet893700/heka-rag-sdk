"""Plug-in registry: (stage, provider name) -> adapter class.

Adapters are looked up by the names used in config files. They can be registered in three ways:

* eagerly, with the `@register("stage", "name")` decorator;
* lazily, with `register_lazy(stage, name, "module:Class", extra="pdf")`, so an adapter's heavy
  dependencies are only imported when that provider is actually used;
* by third-party packages through the `kbsdk.plugins` entry-point group.

Adapter convention: a class with an optional `settings_model` (a pydantic model). When present, the
config's `params` are validated against it and the adapter is constructed as `cls(settings)`;
otherwise it is constructed as `cls(**params)`.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from typing import Any, TypeVar

from kbsdk.errors import MissingExtraError, UnknownProviderError

STAGES = (
    "loader",
    "chunker",
    "embedder",
    "store",
    "retriever",
    "reranker",
    "query_transform",
    "llm",
    "guardrail",
    "cache",
    "tracer",
    "metric",
    "ocr",
)

ENTRY_POINT_GROUP = "kbsdk.plugins"

T = TypeVar("T")


@dataclass
class _Entry:
    target: Any  # the class, or a "module:Attr" string until first use
    extra: str | None = None


class Registry:
    def __init__(self) -> None:
        self._entries: dict[str, dict[str, _Entry]] = {stage: {} for stage in STAGES}
        self._plugins_loaded = False

    # -- registration -------------------------------------------------------------------------

    def register(
        self, stage: str, name: str, *, extra: str | None = None
    ) -> Callable[[type[T]], type[T]]:
        def decorator(cls: type[T]) -> type[T]:
            self._add(stage, name, _Entry(cls, extra))
            return cls

        return decorator

    def register_lazy(
        self, stage: str, name: str, target: str, *, extra: str | None = None
    ) -> None:
        if ":" not in target:
            raise ValueError(
                f"Lazy target must look like 'package.module:ClassName', got {target!r}"
            )
        self._add(stage, name, _Entry(target, extra))

    def _add(self, stage: str, name: str, entry: _Entry) -> None:
        self._check_stage(stage)
        self._entries[stage][name] = entry

    @staticmethod
    def _check_stage(stage: str) -> None:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage {stage!r}. Valid stages: {', '.join(STAGES)}")

    # -- lookup -------------------------------------------------------------------------------

    def names(self, stage: str) -> list[str]:
        self._check_stage(stage)
        self.load_plugins()
        return sorted(self._entries[stage])

    def resolve(self, stage: str, name: str) -> type[Any]:
        """Return the adapter class, importing it now if it was registered lazily."""
        self._check_stage(stage)
        self.load_plugins()
        entry = self._entries[stage].get(name)
        if entry is None:
            raise UnknownProviderError(stage, name, sorted(self._entries[stage]))
        if isinstance(entry.target, str):
            module_name, _, attr = entry.target.partition(":")
            try:
                entry.target = getattr(importlib.import_module(module_name), attr)
            except ImportError as exc:
                if entry.extra:
                    raise MissingExtraError(stage, name, entry.extra, exc) from exc
                raise
        cls: type[Any] = entry.target
        return cls

    def create(
        self,
        stage: str,
        name: str,
        params: dict[str, Any] | None = None,
        *,
        deps: Mapping[str, Any] | None = None,
    ) -> Any:
        """Instantiate a provider from its config `params`, validated if the adapter says how.

        `deps` are runtime collaborators (an LLM, an embedder, ...). Each is passed as a keyword
        argument only if the adapter's constructor declares it, so adapters that don't need a
        collaborator are never handed one.
        """
        cls = self.resolve(stage, name)
        params = params or {}
        extra = self._accepted(cls, deps or {})
        settings_model = getattr(cls, "settings_model", None)
        if settings_model is not None:
            return cls(settings_model(**params), **extra)
        return cls(**params, **extra)

    @staticmethod
    def _accepted(cls: type[Any], deps: Mapping[str, Any]) -> dict[str, Any]:
        if not deps:
            return {}
        try:
            parameters = inspect.signature(cls).parameters
        except (TypeError, ValueError):
            return {}
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return dict(deps)
        return {name: value for name, value in deps.items() if name in parameters}

    def settings_schema(self, stage: str, name: str) -> dict[str, Any] | None:
        """JSON schema of a provider's settings (lets an admin UI render forms automatically)."""
        settings_model = getattr(self.resolve(stage, name), "settings_model", None)
        if settings_model is None:
            return None
        schema: dict[str, Any] = settings_model.model_json_schema()
        return schema

    # -- third-party plug-ins -----------------------------------------------------------------

    def load_plugins(self) -> None:
        """Import every `kbsdk.plugins` entry point once; importing registers the adapters."""
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
            entry_point.load()


registry = Registry()
register = registry.register
register_lazy = registry.register_lazy
