"""Import helper for optional dependencies, with a clear 'pip install kbsdk[extra]' error."""

from __future__ import annotations

import importlib
import os
from typing import Any

from pydantic import BaseModel, ConfigDict

from kbsdk.errors import ConfigError, MissingExtraError


class Settings(BaseModel):
    """Base for adapter settings. Unknown keys are errors, so a typo in a config is caught."""

    model_config = ConfigDict(extra="forbid")


def require(module: str, *, stage: str, provider: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingExtraError(stage, provider, extra, exc) from exc


def api_key_from_env(env_name: str, provider: str) -> str:
    """Read a secret from the environment. Keys are never accepted from configuration."""
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ConfigError(
            f"The {provider} provider needs an API key: set the {env_name} environment variable "
            "(keys are read from the environment, never from config files)."
        )
    return value
