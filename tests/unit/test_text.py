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


@pytest.mark.parametrize(
    "quote",
    [
        "eligible non‑citizen",  # non-breaking hyphen: what some models write
        "self‐employed",  # Unicode hyphen
        "non−citizen",  # minus sign
        "eli­gible non-citizen",  # soft hyphen
        "U.S.​ citizen",  # zero-width space
        "café",  # decomposed e-acute
        "café",  # composed e-acute
        "It's free",  # straight apostrophe vs curly in the source
        "ﬁnancial",  # ligature vs "fi"
    ],
)
def test_quotes_survive_lookalike_unicode(quote):
    source = "a U.S. citizen or eligible non-citizen. It’s free. self-employed financial café"
    assert quote_in_text(quote, source)


@pytest.mark.parametrize("quote", ["eligible non-citizens must pay a fee", "5 percent", "self-employed and retired"])
def test_folding_does_not_verify_things_the_source_does_not_say(quote):
    assert not quote_in_text(quote, "a U.S. citizen or eligible non-citizen. It’s free. self-employed")


SPACED_SOURCE = "an eligible noncitizen ; have a valid Social Security number , and the DMV Document Guide [10] lists papers. You ' ll need $ 25 .50."


@pytest.mark.parametrize(
    "quote",
    [
        "an eligible noncitizen; have a valid Social Security number, and",  # natural spacing
        "the DMV Document Guide lists papers",  # link marker left out
        "You'll need $25.50",  # apostrophe and currency
    ],
)
def test_quotes_ignore_extraction_spacing_around_punctuation(quote):
    assert quote_in_text(quote, SPACED_SOURCE)


@pytest.mark.parametrize(
    "quote",
    [
        "an eligible noncitizen, have a valid",  # different punctuation is a different quote
        "an eligible citizen; have a valid",  # different word
        "You'll need $25.60",  # different amount
        "You must not be an eligible noncitizen",  # meaning flipped
    ],
)
def test_spacing_tolerance_does_not_verify_different_quotes(quote):
    assert not quote_in_text(quote, SPACED_SOURCE)
