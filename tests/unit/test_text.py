import pytest

from heka.rag.text import extract_json, quote_in_text, slugify, stable_hash

SOURCE = (
    "## Casual leave\n\nFull-time employees receive **12 days** of casual leave per calendar year. "
    "Unused casual leave doesn’t carry over.\n\n| Plan | Cost |\n| --- | --- |\n| Family | 90 |"
)


@pytest.mark.parametrize(
    "quote",
    [
        "12 days of casual leave per calendar year",
        "FULL-TIME   EMPLOYEES receive",  # case and whitespace are ignored
        "Unused casual leave doesn't carry over",  # curly vs straight apostrophe
        "receive 12 days of casual leave",  # markdown bold in the source is ignored
        "Full-time employees ... per calendar year",  # ellipsis skips material
        "Family | 90",  # table cells
    ],
)
def test_real_quotes_verify(quote):
    assert quote_in_text(quote, SOURCE)


@pytest.mark.parametrize(
    "quote",
    [
        "24 days of casual leave per calendar year",  # a changed fact must not verify
        "employees receive 12 days of sick leave",
        "per calendar year ... Full-time employees",  # fragments out of order
        "",
        "ab",  # too short to mean anything
        "carry over. Unused casual leave",  # wrong order
    ],
)
def test_fabricated_quotes_fail(quote):
    assert not quote_in_text(quote, SOURCE)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure! ```json\n{"a": 1}\n``` done') == {"a": 1}
    assert extract_json('prefix {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    assert extract_json("no json here") is None
    assert extract_json("{broken") is None
    assert extract_json("[1, 2]") is None


def test_hash_and_slug():
    assert stable_hash("a", 1) == stable_hash("a", 1)
    assert stable_hash("a", 1) != stable_hash("a", 2)
    assert len(stable_hash("x", length=8)) == 8
    assert slugify("HR Assistant (v2)") == "hr-assistant-v2"
    assert slugify("!!!") == "agent"
