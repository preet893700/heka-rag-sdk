"""Agentic (multi-step) retrieval, as a LangGraph loop around any retriever.

Some questions cannot be answered from the passages one search returns: "What are the opening hours
where Alice works?" needs Alice's office from one document and that office's hours from another.
The loop is

    search -> assess ("is this enough?") --sufficient--> done
                 ^                    \\--needs more--> new query
                 |______________________________________/

bounded by `max_steps`. Each extra step costs one model call (to assess) plus one search. Results from
every step are merged with reciprocal rank fusion. Because this is just a `Retriever`, the rest of the
pipeline (reranking, access filters, parent expansion) applies to its output unchanged.

Caveat: `answer.usage.embedding_calls` counts one embedding per query, so it undercounts here.
"""

from __future__ import annotations

from typing import Any, TypedDict

from kbsdk.adapters._deps import require
from kbsdk.adapters.retrievers import fuse_rrf
from kbsdk.interfaces import LLM, Retriever
from kbsdk.prompts import source_label
from kbsdk.text import extract_json
from kbsdk.types import Filter, Message, RequestContext, ScoredChunk

ASSESS_SYSTEM = """You decide whether a document search has found enough to answer a question.
You see the question and the passages found so far. The answer may need facts from several documents.
- If the passages contain everything needed, reply {"sufficient": true, "next_query": ""}.
- If a fact is missing, reply {"sufficient": false, "next_query": "<one short search query for the \
missing fact>"}. Use specific names and terms taken from the passages (e.g. an office or team name
they mention), not the original wording.
Reply with only JSON."""

_PREVIEW_CHARS = 300
_PREVIEW_PASSAGES = 6


class _State(TypedDict):
    question: str
    pending: str
    asked: list[str]
    lists: list[list[ScoredChunk]]
    step: int
    done: bool
    k: int
    filter: Filter | None
    context: RequestContext | None


class AgenticRetriever:
    def __init__(self, base: Retriever, llm: LLM, *, max_steps: int = 3) -> None:
        self.base = base
        self.llm = llm
        self.max_steps = max_steps
        self._graph: Any = None

    # -- graph --------------------------------------------------------------------------------

    def _compiled(self) -> Any:
        if self._graph is None:
            module = require(
                "langgraph.graph", stage="retriever", provider="agentic", extra="agentic"
            )
            graph = module.StateGraph(_State)
            graph.add_node("search", self._search)
            graph.add_node("assess", self._assess)
            graph.add_edge(module.START, "search")
            graph.add_edge("search", "assess")
            graph.add_conditional_edges(
                "assess",
                lambda s: "end" if s["done"] else "search",
                {"search": "search", "end": module.END},
            )
            self._graph = graph.compile()
        return self._graph

    async def _search(self, state: _State) -> dict[str, Any]:
        found = await self.base.retrieve(
            state["pending"], k=state["k"], filter=state["filter"], context=state["context"]
        )
        return {"lists": [*state["lists"], found], "step": state["step"] + 1}

    async def _assess(self, state: _State) -> dict[str, Any]:
        if state["step"] >= self.max_steps:
            return {"done": True}
        so_far = fuse_rrf(state["lists"], k=_PREVIEW_PASSAGES, label="agentic")
        passages = "\n\n".join(
            f"[{n}] {source_label(c)}\n{' '.join(c.chunk.text.split())[:_PREVIEW_CHARS]}"
            for n, c in enumerate(so_far, start=1)
        )
        prompt = (
            f"Question: {state['question']}\n\nSearches so far: {'; '.join(state['asked'])}\n\n"
            f"Passages found so far:\n{passages or '(none)'}"
        )
        reply = await self.llm.generate(
            [Message(role="user", content=prompt)], system=ASSESS_SYSTEM
        )
        verdict = extract_json(reply.text) or {}
        sufficient = verdict.get("sufficient", True)
        if isinstance(sufficient, str):
            sufficient = sufficient.strip().lower() == "true"
        next_query = " ".join(str(verdict.get("next_query") or "").split())
        seen = {q.casefold() for q in state["asked"]}
        # An unusable reply, "enough", or a query we already ran all mean: stop and use what we have.
        if sufficient or not next_query or next_query.casefold() in seen:
            return {"done": True}
        return {"pending": next_query, "asked": [*state["asked"], next_query]}

    # -- Retriever ----------------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        k: int,
        filter: Filter | None = None,
        context: RequestContext | None = None,
    ) -> list[ScoredChunk]:
        final = await self._compiled().ainvoke(
            {
                "question": query,
                "pending": query,
                "asked": [query],
                "lists": [],
                "step": 0,
                "done": False,
                "k": k,
                "filter": filter,
                "context": context,
            }
        )
        lists = final["lists"]
        return lists[0] if len(lists) == 1 else fuse_rrf(lists, k=k, label="agentic")
