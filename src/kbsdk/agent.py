"""`Agent`: answers questions from a `KnowledgeBase` with verified citations and abstention."""

from __future__ import annotations

from collections.abc import Sequence

from kbsdk import factory
from kbsdk.aio import run_sync
from kbsdk.config import RAGConfig
from kbsdk.errors import ConfigError
from kbsdk.interfaces import LLM
from kbsdk.knowledge_base import KnowledgeBase
from kbsdk.pipelines.query import QueryPipeline
from kbsdk.types import Answer, Filter, Message, RequestContext
from kbsdk.usage import BudgetTracker, MeteredLLM


class Agent:
    def __init__(
        self, kb: KnowledgeBase, config: RAGConfig | None = None, *, llm: LLM | None = None
    ) -> None:
        self.kb = kb
        self.config = config or kb.config
        factory.validate_supported(self.config)
        factory.apply_env_file(self.config)
        self.budget = BudgetTracker(self.config.budget)
        # Metered so that every model call made for one question (answer, query rewriting,
        # reranking, verification) is summed into `Answer.usage`, priced, and held to the budget.
        self.llm: LLM = MeteredLLM(
            llm
            or factory.build_llm(
                self.config,
                self.config.generation.llm,
                kb.cache,
                defaults=factory.generation_defaults(self.config),
            ),
            pricing=self.config.pricing,
            budget=self.budget if self.budget.active else None,
        )
        if (
            self.config.budget.max_total_usd is not None
            and self.llm.name not in self.config.pricing
        ):
            raise ConfigError(
                f"budget.max_total_usd needs a price for {self.llm.name!r} under `pricing:` "
                "(dollars per million tokens), otherwise spend cannot be measured."
            )
        self.retrieval = kb.build_retrieval(self.llm, self.config)
        self.pipeline = QueryPipeline(
            self.config,
            self.retrieval,
            self.llm,
            guardrails=factory.build_guardrails(self.config),
            tracers=factory.build_tracers(self.config),
        )

    async def aask(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        context: RequestContext | None = None,
        filter: Filter | None = None,
    ) -> Answer:
        """Answer from the documents. `context` says who is asking (set by your app, never the model)."""
        return await self.pipeline.run(question, history=history, context=context, filter=filter)

    def ask(
        self,
        question: str,
        *,
        history: Sequence[Message] = (),
        context: RequestContext | None = None,
        filter: Filter | None = None,
    ) -> Answer:
        return run_sync(self.aask(question, history=history, context=context, filter=filter))
