"""Async helpers. The SDK is async at its core; sync methods are thin wrappers over it."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_sync(coro: Awaitable[T]) -> T:
    """Run a coroutine to completion from synchronous code.

    Works from plain scripts and also from inside a running loop (notebooks, async apps), where it
    runs the coroutine on a helper thread with its own event loop.
    """

    async def _wrapper() -> T:
        return await coro

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_wrapper())
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _wrapper()).result()


async def gather_limited(
    items: Iterable[T], worker: Callable[[T], Awaitable[R]], *, limit: int
) -> list[R]:
    """Run `worker` over `items` with at most `limit` in flight; results keep input order."""
    semaphore = asyncio.Semaphore(max(1, limit))

    async def _run(item: T) -> R:
        async with semaphore:
            return await worker(item)

    return list(await asyncio.gather(*(_run(item) for item in items)))
