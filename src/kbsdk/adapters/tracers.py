"""Tracers: where a question's `TraceEvent`s go.

Every answer carries its own trace (`answer.trace`); a tracer additionally ships events somewhere
durable. By default question and answer *text* is stripped from events before they leave the process,
because logs are a common place for personal data to leak; set `include_text: true` to keep it.
Tracing never breaks answering: a failing tracer is logged and skipped.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from kbsdk.adapters._deps import Settings, require
from kbsdk.types import TraceEvent

_TEXT_KEYS = {"question", "answer", "original", "standalone"}


def _is_text(event: TraceEvent, key: str, value: Any) -> bool:
    """Whether `event.data[key]` holds user- or model-written text rather than a count or a name."""
    if key in _TEXT_KEYS:
        return True
    if key == "queries" and isinstance(value, list):  # rewritten queries (retrieve's is a count)
        return True
    return key == "reason" and event.name == "model_abstained"  # the model's own words


class TracerSettings(Settings):
    include_text: bool = False


def scrub(event: TraceEvent, *, include_text: bool) -> TraceEvent:
    """A copy of `event` without question/answer-derived text unless asked to keep it."""
    if include_text or not any(_is_text(event, k, v) for k, v in event.data.items()):
        return event
    return event.model_copy(
        update={"data": {k: v for k, v in event.data.items() if not _is_text(event, k, v)}}
    )


class MemoryTracer:
    """Collects events in a list. Handy in tests and notebooks."""

    settings_model = TracerSettings

    def __init__(self, settings: TracerSettings | None = None) -> None:
        self.settings = settings or TracerSettings()
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(scrub(event, include_text=self.settings.include_text))


class LoggingSettings(TracerSettings):
    logger: str = "kbsdk.trace"
    level: str = "INFO"


class LoggingTracer:
    """One JSON object per event through the standard `logging` module."""

    settings_model = LoggingSettings

    def __init__(self, settings: LoggingSettings | None = None) -> None:
        self.settings = settings or LoggingSettings()
        self._logger = logging.getLogger(self.settings.logger)
        self._level = logging.getLevelName(self.settings.level.upper())
        if not isinstance(self._level, int):
            raise ValueError(f"Unknown log level {self.settings.level!r}")

    def emit(self, event: TraceEvent) -> None:
        clean = scrub(event, include_text=self.settings.include_text)
        self._logger.log(self._level, json.dumps(clean.model_dump(), default=str))


class JsonlSettings(TracerSettings):
    path: str


class JsonlTracer:
    """Appends one JSON object per line to a file (easy to grep, load into pandas, or ship on)."""

    settings_model = JsonlSettings

    def __init__(self, settings: JsonlSettings) -> None:
        self.settings = settings
        self._path = Path(settings.path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def emit(self, event: TraceEvent) -> None:
        clean = scrub(event, include_text=self.settings.include_text)
        line = json.dumps(clean.model_dump(), default=str)
        with self._lock, self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def _attribute(value: Any) -> Any:
    """OpenTelemetry attributes must be primitives or homogeneous lists of them."""
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple) and value and all(isinstance(v, str) for v in value):
        return list(value)
    if (
        isinstance(value, list | tuple)
        and value
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in value)
    ):
        return list(value)
    return json.dumps(value, default=str)


class OtelSettings(TracerSettings):
    tracer_name: str = "kbsdk"


class OtelTracer:
    """OpenTelemetry spans, one per pipeline step, using whatever TracerProvider your app configured
    (Langfuse, Jaeger, Datadog, ... all accept OTLP). With none configured the spans are no-ops."""

    settings_model = OtelSettings

    def __init__(self, settings: OtelSettings | None = None) -> None:
        self.settings = settings or OtelSettings()
        module = require("opentelemetry.trace", stage="tracer", provider="otel", extra="otel")
        self._tracer = module.get_tracer(self.settings.tracer_name)

    def emit(self, event: TraceEvent) -> None:
        clean = scrub(event, include_text=self.settings.include_text)
        end_ns = int(clean.timestamp * 1e9)
        start_ns = end_ns - int((clean.duration_ms or 0.0) * 1e6)
        span = self._tracer.start_span(f"kbsdk.{clean.stage}.{clean.name}", start_time=start_ns)
        span.set_attribute("kbsdk.stage", clean.stage)
        for key, value in clean.data.items():
            if value is not None:
                span.set_attribute(f"kbsdk.{key}", _attribute(value))
        span.end(end_time=end_ns)
