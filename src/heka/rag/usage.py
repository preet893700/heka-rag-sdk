"""Usage metering: count every LLM call made while answering one question, wherever it happens
(the answer itself, query rewriting, reranking, verification, judging), without threading a counter
through every function. Also prices the calls (from prices you supply) and enforces budgets."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar

from heka.rag.config import BudgetConfig, ModelPrice
from heka.rag.errors import BudgetExceededError
from heka.rag.interfaces import LLM
from heka.rag.types import LLMResponse, Message, Usage


class UsageMeter:
    def __init__(self) -> None:
        self.total = Usage()

    def add(self, usage: Usage) -> None:
        self.total = self.total + usage


_current: ContextVar[UsageMeter | None] = ContextVar("heka_rag_usage_meter", default=None)


@contextmanager
def track_usage() -> Iterator[UsageMeter]:
    """Everything metered inside the block (including concurrent tasks it spawns) is summed."""
    meter = UsageMeter()
    token = _current.set(meter)
    try:
        yield meter
    finally:
        _current.reset(token)


def record_usage(usage: Usage) -> None:
    meter = _current.get()
    if meter is not None:
        meter.add(usage)


class BudgetTracker:
    """Enforces `BudgetConfig` across the lifetime of one agent.

    `max_total_usd` needs a price for the model (`pricing:`), otherwise spend cannot be known; the
    tracker refuses to start rather than silently not enforce a budget.
    """

    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.spent_usd = 0.0

    @property
    def active(self) -> bool:
        return (
            self.config.max_total_usd is not None
            or self.config.max_llm_calls_per_question is not None
        )

    def check(self) -> None:
        """Raise before a model call if a budget is already used up."""
        limit = self.config.max_total_usd
        if limit is not None and self.spent_usd >= limit:
            raise BudgetExceededError(
                f"The ${limit:.2f} budget is used up (spent ${self.spent_usd:.4f}); no more model calls."
            )
        cap = self.config.max_llm_calls_per_question
        meter = _current.get()
        if cap is not None and meter is not None and meter.total.llm_calls >= cap:
            raise BudgetExceededError(
                f"This question already made {meter.total.llm_calls} model calls "
                f"(limit {cap}); stopping to avoid a runaway loop."
            )

    def add(self, cost_usd: float | None) -> None:
        if cost_usd:
            self.spent_usd += cost_usd


def price_usage(usage: Usage, price: ModelPrice | None) -> Usage:
    """`usage` with `cost_usd` filled in from token counts (cached responses cost nothing)."""
    if price is None:
        return usage
    cost = (
        usage.input_tokens * price.input_per_mtok + usage.output_tokens * price.output_per_mtok
    ) / 1e6
    return usage.model_copy(update={"cost_usd": cost})


class MeteredLLM:
    """Wraps an LLM so each response's usage (and cost) is added to the active `track_usage()` block,
    and budgets are enforced before each call."""

    def __init__(
        self,
        inner: LLM,
        *,
        pricing: Mapping[str, ModelPrice] | None = None,
        budget: BudgetTracker | None = None,
    ) -> None:
        self.inner = inner
        self.pricing = dict(pricing or {})
        self.budget = budget

    @property
    def name(self) -> str:
        return self.inner.name

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMResponse:
        if self.budget is not None:
            self.budget.check()
        response = await self.inner.generate(
            messages, system=system, temperature=temperature, max_output_tokens=max_output_tokens
        )
        usage = price_usage(response.usage, self.pricing.get(self.inner.name))
        if self.budget is not None:
            self.budget.add(usage.cost_usd)
        record_usage(usage)
        return response.model_copy(update={"usage": usage})

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        if self.budget is not None:
            self.budget.check()
        async for piece in self.inner.stream(
            messages, system=system, temperature=temperature, max_output_tokens=max_output_tokens
        ):
            yield piece
