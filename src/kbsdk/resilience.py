"""Client-side throttling and retry, so free-tier and rate-limited providers don't derail long runs.

* `AsyncRateLimiter` spaces out call starts to at most N per minute.
* `with_retries` retries transient failures (429/5xx/timeouts) with jittered exponential backoff and
  honours a provider's "retry in Ns" hint when the error carries one.
"""

from __future__ import annotations

import asyncio
import random
import re
import time
import weakref
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")

_SHARED_LIMITERS: dict[str, AsyncRateLimiter] = {}


def shared_limiter(key: str, requests_per_minute: float | None) -> AsyncRateLimiter:
    """One limiter per (provider, model): the generator and the judge share a provider's quota."""
    limiter = _SHARED_LIMITERS.get(key)
    if limiter is None or limiter.requests_per_minute != requests_per_minute:
        limiter = _SHARED_LIMITERS[key] = AsyncRateLimiter(requests_per_minute)
    return limiter


_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_RETRYABLE_NAMES = (
    "ratelimit",
    "resourceexhausted",
    "toomanyrequests",
    "serviceunavailable",
    "internalserver",
    "apiconnection",
    "timeout",
    "overloaded",
    "unavailable",
)
_RETRY_HINT = re.compile(r"retry[^0-9]{0,20}(\d+(?:\.\d+)?)\s*(ms|s)", re.IGNORECASE)


class AsyncRateLimiter:
    """Allow at most `requests_per_minute` call starts; `None` disables throttling."""

    def __init__(
        self,
        requests_per_minute: float | None,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.requests_per_minute = requests_per_minute
        self._interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._clock = clock
        self._sleep = sleep
        self._next_slot = 0.0
        # asyncio locks belong to one event loop; keep one per loop so a limiter can outlive
        # several `asyncio.run` calls (sync wrappers, tests) without cross-loop errors.
        self._locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = self._locks[loop] = asyncio.Lock()
        return lock

    async def wait(self) -> None:
        if not self._interval:
            return
        async with self._lock():
            now = self._clock()
            delay = self._next_slot - now
            if delay > 0:
                await self._sleep(delay)
                now = self._clock()
            self._next_slot = max(now, self._next_slot) + self._interval


def _status_code(exc: BaseException) -> int | None:
    for attr in ("status_code", "code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return False
    if isinstance(exc, TimeoutError | ConnectionError):
        return True
    status = _status_code(exc)
    if status is not None:
        return status in _RETRYABLE_STATUS
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return any(n in name for n in _RETRYABLE_NAMES) or "resource_exhausted" in text


def retry_after_seconds(exc: BaseException) -> float | None:
    """A server-suggested wait (e.g. Gemini's 'Please retry in 12.3s'), if the error carries one."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            return float(headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    match = _RETRY_HINT.search(str(exc))
    if match:
        seconds = float(match.group(1))
        return seconds / 1000 if match.group(2).lower() == "ms" else seconds
    return None


async def with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    jitter: Callable[[], float] = random.random,
    before_attempt: Callable[[], Awaitable[None]] | None = None,
) -> T:
    """Call `call()`, retrying retryable failures up to `max_retries` times."""
    attempt = 0
    pause = sleep or asyncio.sleep  # looked up per call so tests can substitute it
    while True:
        if before_attempt is not None:
            await before_attempt()
        try:
            return await call()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable(exc):
                raise
            suggested = retry_after_seconds(exc)
            backoff = min(max_delay, base_delay * (2**attempt)) * (0.5 + jitter() / 2)
            await pause(min(max(suggested + 0.25, backoff) if suggested else backoff, 120.0))
            attempt += 1
