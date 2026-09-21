"""Opt-in `.env` file support.

Nothing is read unless you ask for it: pass `--env-file` to the CLI, call `load_env_file(...)`, or set
`env_file:` in a config. Rules that keep it safe:

* variables already present in the real environment always win (unless `override=True`);
* values are never returned, logged or included in error messages - only variable *names*;
* an explicitly requested file that does not exist is an error, not a silent no-op.

Supported syntax: `KEY=value`, `export KEY=value`, single or double quotes (double quotes understand
`\\n`, `\\t`, `\\"` and `\\\\`), and `#` comments (whole-line, or after whitespace on unquoted values).
There is no variable interpolation and no multi-line values.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from heka.rag.errors import ConfigError

_ASSIGNMENT = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_.\-]*)\s*=\s*(.*)$")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


@dataclass
class EnvLoadResult:
    """What a load did. Holds names only, never values."""

    path: Path
    loaded: list[str] = field(default_factory=list)  # newly set in this process
    skipped_existing: list[str] = field(default_factory=list)  # already set, left untouched

    def summary(self) -> str:
        parts = [f"{self.path}: loaded {len(self.loaded)} variable(s)"]
        if self.loaded:
            parts.append(f"({', '.join(self.loaded)})")
        if self.skipped_existing:
            parts.append(f"- kept existing: {', '.join(self.skipped_existing)}")
        return " ".join(parts)


def _parse_value(raw: str, *, where: str) -> str:
    raw = raw.strip()
    if raw[:1] == '"':
        out: list[str] = []
        index = 1
        while index < len(raw):
            char = raw[index]
            if char == "\\" and index + 1 < len(raw):
                out.append(_ESCAPES.get(raw[index + 1], "\\" + raw[index + 1]))
                index += 2
                continue
            if char == '"':
                return "".join(out)
            out.append(char)
            index += 1
        raise ConfigError(f"{where}: unterminated double quote")
    if raw[:1] == "'":
        end = raw.find("'", 1)
        if end < 0:
            raise ConfigError(f"{where}: unterminated single quote")
        return raw[1:end]
    return re.split(r"\s+#", raw, maxsplit=1)[0].strip()


def parse_env_text(text: str, *, source: str = ".env") -> dict[str, str]:
    """Parse `.env` text into a dict. Later assignments of the same name win."""
    values: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _ASSIGNMENT.match(stripped)
        if match is None:
            raise ConfigError(f"{source}:{number}: expected KEY=value")
        values[match.group(1)] = _parse_value(match.group(2), where=f"{source}:{number}")
    return values


def load_env_file(
    path: str | Path = ".env", *, override: bool = False, must_exist: bool = True
) -> EnvLoadResult:
    """Read a `.env` file into `os.environ`. Existing variables win unless `override=True`."""
    target = Path(path)
    result = EnvLoadResult(path=target)
    if not target.is_file():
        if must_exist:
            raise ConfigError(f"Env file not found: {target}")
        return result
    try:
        text = target.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"Cannot read env file {target}: {exc.strerror or exc}") from exc
    for name, value in parse_env_text(text, source=str(target)).items():
        if name in os.environ and not override:
            result.skipped_existing.append(name)
            continue
        os.environ[name] = value
        result.loaded.append(name)
    return result
