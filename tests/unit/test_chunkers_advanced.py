import pytest
from pydantic import ValidationError

from heka.rag import KnowledgeBase, registry
from heka.rag.adapters.chunkers_advanced import (
    ParentChildChunker,
    ParentChildSettings,
    SemanticChunker,
    SemanticSettings,
)
from heka.rag.adapters.embedders import HashingEmbedder
from heka.rag.types import Document

LONG_SECTION = (
    "# Handbook\n\n## Travel\n\n"
    + " ".join(f"Travel sentence number {i} covers flights and hotels." for i in range(20))
    + "\n\nHotel bookings above the limit need director approval and a written justification.\n"
)


def doc(text):
    return Document(id="d1", source="handbook.md", text=text, metadata={"title": "Handbook"})


# -- parent-child -------------------------------------------------------------------------------


async def test_children_are_small_and_carry_their_parent_section():
    chunker = ParentChildChunker(
        ParentChildSettings(parent_size=1200, child_size=300, child_overlap=40)
    )
    chunks = await chunker.chunk(doc(LONG_SECTION))
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 40 for c in chunks)
    parents = {c.metadata["parent_id"] for c in chunks}
    assert len(parents) >= 1 and all(c.parent_id == c.metadata["parent_id"] for c in chunks)
    for chunk in chunks:
        assert chunk.metadata["parent_text"].count("Travel sentence") >= 1
        assert (
            chunk.text in chunk.metadata["parent_text"]
            or chunk.text[:30] in chunk.metadata["parent_text"]
        )
        assert chunk.metadata["heading_path"] == "Handbook > Travel"
        assert chunk.metadata["context"] == "Handbook > Travel"
    assert len({c.id for c in chunks}) == len(chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


async def test_short_sections_produce_one_child_equal_to_the_parent():
    chunks = await ParentChildChunker().chunk(doc("# T\n\n## Small\n\nOne short paragraph here.\n"))
    assert len(chunks) == 1 and chunks[0].text == chunks[0].metadata["parent_text"]


def test_parent_child_setting_validation():
    with pytest.raises(ValidationError, match="smaller than parent_size"):
        ParentChildSettings(parent_size=500, child_size=500)
    with pytest.raises(ValidationError, match="child_overlap"):
        ParentChildSettings(child_size=200, child_overlap=200)


async def test_parent_child_end_to_end_expands_to_the_section(make_config, docs_dir):
    (docs_dir / "policy.md").write_text(
        "# Policy\n\n## Travel\n\n"
        + " ".join(f"Filler sentence {i} about nothing in particular." for i in range(30))
        + " The secret approval code is ZEBRA-99. "
        + " ".join(f"More filler sentence {i} about other things." for i in range(30))
        + "\n",
        encoding="utf-8",
    )
    kb = KnowledgeBase(
        make_config(
            chunker={
                "provider": "parent_child",
                "params": {"parent_size": 4000, "child_size": 300},
            },
            retrieval={"mode": "hybrid", "final_k": 2},
        )
    )
    await kb.aingest()
    top = (await kb.aretrieve("ZEBRA-99 approval code", k=1))[0]
    assert "ZEBRA-99" in top.chunk.text
    assert len(top.chunk.text) > 1500  # the model gets the whole section, not the 300-char child
    assert "parent_text" not in top.chunk.metadata


# -- semantic -----------------------------------------------------------------------------------

TWO_TOPICS = (
    "# Guide\n\n## Mixed\n\n"
    + " ".join(
        f"Employees request annual leave through the portal and managers approve leave request {i}."
        for i in range(6)
    )
    + " "
    + " ".join(
        f"Expense receipts are scanned and finance reimburses the expense claim total {i}."
        for i in range(6)
    )
)


async def test_semantic_chunker_cuts_where_the_topic_changes():
    chunker = SemanticChunker(
        SemanticSettings(min_chunk_size=100, max_chunk_size=900, breakpoint_percentile=90),
        embedder=HashingEmbedder(),
    )
    chunks = await chunker.chunk(doc(TWO_TOPICS))
    assert len(chunks) >= 2
    leave = [c for c in chunks if "annual leave" in c.text]
    expense = [c for c in chunks if "Expense receipts" in c.text]
    assert leave and expense
    assert not any("annual leave" in c.text and "Expense receipts" in c.text for c in chunks), (
        "the boundary should fall between the two topics"
    )
    assert all(c.metadata["heading_path"] == "Guide > Mixed" for c in chunks)


async def test_semantic_chunker_respects_max_size_and_keeps_tables_whole():
    table = "| A | B |\n| --- | --- |\n" + "\n".join(f"| r{i} | {i} |" for i in range(5))
    text = (
        "# T\n\n"
        + " ".join(f"Plain sentence {i} about leave policy." for i in range(40))
        + "\n\n"
        + table
    )
    chunker = SemanticChunker(
        SemanticSettings(max_chunk_size=400, min_chunk_size=100), embedder=HashingEmbedder()
    )
    chunks = await chunker.chunk(doc(text))
    assert all(len(c.text) <= 400 or c.text.startswith("|") for c in chunks)
    assert any("| r0 | 0 |" in c.text and "| r4 | 4 |" in c.text for c in chunks)


async def test_semantic_chunker_handles_tiny_and_empty_documents():
    chunker = SemanticChunker(embedder=HashingEmbedder())
    assert await chunker.chunk(doc("")) == []
    one = await chunker.chunk(doc("# T\n\nJust one sentence."))
    assert [c.text for c in one] == ["Just one sentence."]


def test_semantic_setting_validation():
    with pytest.raises(ValidationError, match="smaller than max_chunk_size"):
        SemanticSettings(min_chunk_size=500, max_chunk_size=400)


async def test_semantic_chunker_is_built_with_the_embedder_by_the_factory(make_config):
    kb = KnowledgeBase(make_config(chunker={"provider": "semantic"}))
    assert isinstance(kb.chunker, SemanticChunker) and kb.chunker.embedder is kb.embedder
    report = await kb.aingest()
    assert report.chunks_total > 0 and not report.failed


def test_both_are_registered():
    assert {"structure_aware", "fixed", "parent_child", "semantic"} <= set(
        registry.names("chunker")
    )
