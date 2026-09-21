import pytest
from pydantic import ValidationError

from kbsdk.adapters.chunkers import ChunkerSettings, FixedChunker, StructureAwareChunker
from kbsdk.types import Document

HANDBOOK = """# Leave Policy

Effective 1 January.

## Casual leave

Employees get 12 days. Unused days lapse.

## Sick leave

### Certificates

A certificate is needed after 2 days.

## Earned leave

Earned leave accrues monthly.
"""


def doc(text, **metadata):
    return Document(
        id="d1", source="leave.md", text=text, metadata={"title": "Leave Policy", **metadata}
    )


async def test_sections_carry_heading_paths():
    chunks = await StructureAwareChunker().chunk(doc(HANDBOOK))
    paths = [c.metadata["heading_path"] for c in chunks]
    assert "Leave Policy > Casual leave" in paths
    assert "Leave Policy > Sick leave > Certificates" in paths
    casual = next(c for c in chunks if c.metadata["heading"] == "Casual leave")
    assert casual.text == "Employees get 12 days. Unused days lapse."  # verbatim, no heading prefix
    assert casual.metadata["context"] == "Leave Policy > Casual leave"  # title not duplicated
    assert casual.metadata["source"] == "leave.md"


async def test_ids_are_stable_and_ordered():
    first = await StructureAwareChunker().chunk(doc(HANDBOOK))
    second = await StructureAwareChunker().chunk(doc(HANDBOOK))
    assert [c.id for c in first] == [c.id for c in second]
    assert [c.index for c in first] == list(range(len(first)))


async def test_long_sections_split_within_size_with_overlap():
    sentences = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunker = StructureAwareChunker(ChunkerSettings(chunk_size=300, chunk_overlap=60))
    chunks = await chunker.chunk(doc(f"# T\n\n## Long\n\n{sentences}\n"))
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 60 for c in chunks)
    assert all(c.metadata["heading_path"] == "T > Long" for c in chunks)
    # overlap: the start of each chunk repeats the end of the previous one
    for previous, following in zip(chunks, chunks[1:], strict=False):
        assert following.text.split()[0] in previous.text


async def test_tables_stay_whole_or_repeat_their_header():
    rows = "\n".join(f"| Row{i} | {i} |" for i in range(40))
    table = f"| Name | Value |\n| --- | --- |\n{rows}"
    chunker = StructureAwareChunker(ChunkerSettings(chunk_size=200, chunk_overlap=20))
    chunks = await chunker.chunk(doc(f"# T\n\n{table}\n"))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text.startswith("| Name | Value |\n| --- | --- |")


async def test_pages_are_tracked_through_form_feeds():
    text = "# Policy\n\nPage one text.\f\nPage two text.\f\nPage three text."
    chunker = StructureAwareChunker(ChunkerSettings(chunk_size=100, chunk_overlap=0))
    chunks = await chunker.chunk(doc(text))
    assert chunks[0].metadata["page"] == 1
    assert chunks[-1].metadata["page_end"] == 3
    assert all("\f" not in c.text for c in chunks)


async def test_code_fences_do_not_create_headings():
    text = "# Guide\n\n```\n# not a heading\nprint('x')\n```\n\nAfter the code."
    chunks = await StructureAwareChunker().chunk(doc(text))
    assert {c.metadata["heading_path"] for c in chunks} == {"Guide"}
    assert any("# not a heading" in c.text for c in chunks)


async def test_document_without_headings_still_chunks():
    chunks = await StructureAwareChunker().chunk(doc("Just plain text.\n\nAnother paragraph."))
    assert len(chunks) == 1
    assert chunks[0].metadata["heading_path"] == ""
    assert chunks[0].metadata["context"] == "Leave Policy"


async def test_empty_document_yields_no_chunks():
    assert await StructureAwareChunker().chunk(doc("  \n\n")) == []
    assert await FixedChunker().chunk(doc("")) == []


async def test_metadata_is_merged_into_chunks():
    chunks = await StructureAwareChunker().chunk(doc(HANDBOOK, department="hr"))
    assert all(c.metadata["department"] == "hr" for c in chunks)


async def test_fixed_chunker_windows_and_pages():
    text = ("word " * 200).strip() + "\f" + ("other " * 200).strip()
    chunker = FixedChunker(ChunkerSettings(chunk_size=200, chunk_overlap=40))
    chunks = await chunker.chunk(doc(text))
    assert len(chunks) > 4
    assert all(len(c.text) <= 200 for c in chunks)
    assert chunks[0].metadata["page"] == 1
    assert chunks[-1].metadata["page"] == 2
    assert chunks[0].metadata["heading_path"] == ""


def test_settings_validation():
    with pytest.raises(ValidationError, match="smaller"):
        ChunkerSettings(chunk_size=200, chunk_overlap=200)
    with pytest.raises(ValidationError):
        ChunkerSettings(chunk_size=200, typo=1)
