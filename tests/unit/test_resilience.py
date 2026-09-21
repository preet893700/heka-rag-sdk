import asyncio

import pytest

from heka.rag.resilience import (
    AsyncRateLimiter,
    is_retryable,
    retry_after_seconds,
    shared_limiter,
    with_retries,
)


class Http(Exception):
    def __init__(self, status_code, message=""):
        super().__init__(message or f"HTTP {status_code}")
        self.status_code = status_code


async def no_sleep(_):
    return None


def test_retryable_classification():
    assert is_retryable(Http(429))
    assert is_retryable(Http(503))
    assert not is_retryable(Http(400))
    assert not is_retryable(Http(401))
    assert is_retryable(TimeoutError())
    assert is_retryable(type("RateLimitError", (Exception,), {})("slow down"))
    assert is_retryable(Exception("RESOURCE_EXHAUSTED: quota"))
    assert not is_retryable(ValueError("bad input"))


def test_retry_after_hints():
    assert retry_after_seconds(Exception("Please retry in 12.5s.")) == 12.5
    assert retry_after_seconds(Exception("retry after 500ms")) == 0.5
    assert retry_after_seconds(Exception("nothing useful")) is None


async def test_retries_then_succeeds():
    attempts = []
    sleeps = []

    async def call():
        attempts.append(1)
        if len(attempts) < 3:
            raise Http(429)
        return "ok"

    async def sleep(seconds):
        sleeps.append(seconds)

    assert await with_retries(call, max_retries=3, sleep=sleep, jitter=lambda: 0.0) == "ok"
    assert len(attempts) == 3
    assert sleeps[1] > sleeps[0]  # exponential backoff


async def test_gives_up_after_max_retries():
    attempts = []

    async def call():
        attempts.append(1)
        raise Http(503)

    with pytest.raises(Http):
        await with_retries(call, max_retries=2, sleep=no_sleep)
    assert len(attempts) == 3


async def test_non_retryable_fails_immediately():
    attempts = []

    async def call():
        attempts.append(1)
        raise Http(400)

    with pytest.raises(Http):
        await with_retries(call, max_retries=5, sleep=no_sleep)
    assert len(attempts) == 1


async def test_server_suggested_delay_is_honoured():
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    calls = []

    async def call():
        calls.append(1)
        if len(calls) == 1:
            raise Http(429, "quota exceeded. Please retry in 20s")
        return "ok"

    await with_retries(call, max_retries=1, sleep=sleep, base_delay=0.1, jitter=lambda: 0.0)
    assert sleeps[0] >= 20


async def test_rate_limiter_spaces_calls():
    now = [0.0]
    slept = []

    async def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = AsyncRateLimiter(30, clock=lambda: now[0], sleep=sleep)  # one call per 2 seconds
    for _ in range(3):
        await limiter.wait()
    assert slept == [2.0, 2.0]


async def test_rate_limiter_disabled_never_sleeps():
    limiter = AsyncRateLimiter(None, sleep=no_sleep)
    await asyncio.gather(*(limiter.wait() for _ in range(5)))


def test_limiter_survives_multiple_event_loops():
    limiter = AsyncRateLimiter(6000)

    async def burst():
        await asyncio.gather(limiter.wait(), limiter.wait())

    for _ in range(2):  # each asyncio.run is a new loop; a shared asyncio.Lock would break here
        asyncio.run(burst())


def test_shared_limiter_is_shared_per_key():
    assert shared_limiter("k1", 10) is shared_limiter("k1", 10)
    assert shared_limiter("k1", 10) is not shared_limiter("k2", 10)
    assert shared_limiter("k1", 20).requests_per_minute == 20
