"""Exception hierarchy. Everything the SDK raises on purpose derives from `HekaRagError`."""

from __future__ import annotations


class HekaRagError(Exception):
    """Base class for all SDK errors."""


class ConfigError(HekaRagError, ValueError):
    """The configuration is invalid or cannot be loaded."""


class AccessDeniedError(HekaRagError, PermissionError):
    """A question was asked without the identity the access policy requires."""


class BudgetExceededError(HekaRagError):
    """A configured cost or call budget was used up."""


class UnknownProviderError(HekaRagError, KeyError):
    """A config names a provider that no adapter has registered."""

    def __init__(self, stage: str, name: str, available: list[str]) -> None:
        self.stage = stage
        self.name = name
        self.available = available
        options = ", ".join(available) if available else "none registered"
        super().__init__(f"Unknown {stage} provider {name!r}. Available: {options}.")

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])


class MissingExtraError(HekaRagError, ImportError):
    """An adapter needs an optional dependency group that is not installed."""

    def __init__(self, stage: str, name: str, extra: str, cause: ImportError) -> None:
        self.extra = extra
        super().__init__(
            f"The {stage} provider {name!r} needs optional dependencies "
            f"(missing: {cause.name or cause}). Install them with: pip install 'heka-rag-sdk[{extra}]'"
        )
