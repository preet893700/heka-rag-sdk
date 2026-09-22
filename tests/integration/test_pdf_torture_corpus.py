"""Regression tests for the PDF resolver (Addendum 3) against its real, provenance-logged torture
corpus (see `examples/public_corpora/build_pdf_torture_corpus.py` and this folder's `manifest.json`).

Unlike `tests/unit/test_loaders.py`'s hand-built PDFs, these are the actual documents the resolver
was built for: a real two-column Federal Register issue, a real IRS publication with a running
header repeated on every page, and a few small PDFs (borderless table, password, AcroForm) it is
impractical to reliably find "in the wild" with known-good expected content, so are constructed.

The corpus lives outside the repo's git history (`data/` is git-ignored, like every other public
corpus this project uses) and is not fetched in CI, so every test here skips cleanly when the
corpus folder is missing locally - run
`python examples/public_corpora/build_pdf_torture_corpus.py --out data/public-corpus/pdf-torture`
once to build it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from heka.rag import ConfigError
from heka.rag.adapters.loaders import PdfLoader, _split_line_at_column_gap

# Kept in sync with examples/public_corpora/build_pdf_torture_corpus.py by hand.
TORTURE_PDF_PASSWORD = "torture-test-2026"
TORTURE_FORM_FIELD = "ApplicantName"
TORTURE_FORM_VALUE = "Jordan Alvarez"

DOCS = Path(__file__).resolve().parents[2] / "data" / "public-corpus" / "pdf-torture" / "docs"

pytestmark = pytest.mark.skipif(
    not DOCS.is_dir(),
    reason=(
        "PDF torture corpus not built locally - run "
        "`python examples/public_corpora/build_pdf_torture_corpus.py "
        "--out data/public-corpus/pdf-torture`"
    ),
)


def _headings(text: str) -> list[str]:
    return [line for line in text.splitlines() if re.match(r"^#{1,6} ", line)]


async def load_one(loader: PdfLoader, path: Path):
    docs = [d async for d in loader.load(str(path))]
    assert len(docs) == 1
    return docs[0]


async def test_real_federal_register_two_column_issue_gets_far_more_headings_than_regex_alone():
    """The confirmed root-cause case (Addendum 3's Context section), proven on a document that was
    never seen while writing the fix: a genuinely two-column government publication, not a
    hand-built test PDF."""
    path = DOCS / "federal-register-2024-01-02-excerpt.pdf"
    regex_only = await load_one(PdfLoader(extraction="text"), path)
    font_aware = await load_one(PdfLoader(extraction="layout"), path)

    assert font_aware.metadata["pages"] == 15
    assert "needs_ocr" not in font_aware.metadata  # a real text PDF, not a scan
    assert len(font_aware.text) > 30_000  # real, substantial content, not an extraction failure

    # The whole point: font-size detection finds far more real headings than wording-based regex
    # alone can, on a document neither was tuned against.
    assert len(_headings(font_aware.text)) > 3 * len(_headings(regex_only.text))


async def test_real_federal_register_column_splitting_actually_engages():
    """Proves the two-column mechanism fires on real content, not just the hand-built test page in
    `tests/unit/test_loaders.py`. Checking this indirectly through the final text turned out to be
    unreliable while writing this test: pdfplumber's own un-split merge of two columns into one line
    is not obviously shorter or choppier than correctly split text (production/print banner lines
    that are deliberately full-width skew any such average) - so this checks the mechanism directly,
    the same way `test_reading_order_two_columns_reset_at_a_spanning_heading` checks the ordering
    algorithm directly, rather than inferring it from text statistics.
    """
    pdfplumber = pytest.importorskip("pdfplumber")
    path = DOCS / "federal-register-2024-01-02-excerpt.pdf"
    with pdfplumber.open(str(path)) as pdf:
        # Pages confirmed genuinely two-column when this corpus was built (see the module docstring
        # of build_pdf_torture_corpus.py); a real front-matter/table-of-contents page is skipped.
        split_lines = 0
        for index in (2, 3, 10, 11, 12, 13):
            page = pdf.pages[index]
            mid = float(page.width) / 2
            split_lines += sum(
                1
                for line in page.extract_text_lines()
                if len(_split_line_at_column_gap(line, mid)) == 2
            )
    assert split_lines > 50


async def test_real_irs_publication_running_header_is_stripped_on_every_repeated_page():
    """The literal real-world case `strip_repeated_lines` was written for (see Addendum 3's Context
    section): "Page N of 59 ... 12:33 - 15-Dec-2025" repeated on (almost) every page."""
    path = DOCS / "irs-p15-2026-repeated-headers-excerpt.pdf"
    loaded = await load_one(PdfLoader(extraction="text"), path)
    pages = loaded.text.split("\f")
    assert len(pages) == 10

    # Pages 2-10 share the exact repeated header/footer; it must be gone from all of them (page 1's
    # own one-time-only proof-draft banner is deliberately left alone: it is not a repeat).
    for page in pages[1:]:
        assert "Fileid" not in page
        assert "of 59" not in page

    # Stripping the boilerplate must not have taken the real content with it.
    assert "withhold" in loaded.text.lower()
    assert "wages" in loaded.text.lower()


async def test_borderless_table_recovered_from_a_real_style_roster():
    path = DOCS / "borderless-roster.pdf"
    loaded = await load_one(PdfLoader(extraction="layout"), path)
    assert "| Unit | Monthly Rent | Occupant |" in loaded.text
    assert "| 102 | 1525 | M. Alvarez |" in loaded.text


@pytest.mark.parametrize("extraction", ["layout", "text"])
async def test_locked_memo_needs_the_right_password(extraction):
    path = DOCS / "locked-memo.pdf"
    with pytest.raises(ValueError, match="password-protected"):
        await load_one(PdfLoader(extraction=extraction), path)


async def test_locked_memo_opens_with_the_documented_password(monkeypatch):
    path = DOCS / "locked-memo.pdf"
    monkeypatch.setenv("TORTURE_PDF_PASSWORD", TORTURE_PDF_PASSWORD)
    loaded = await load_one(PdfLoader(password_env="TORTURE_PDF_PASSWORD"), path)
    assert "confidential until signed" in loaded.text


async def test_locked_memo_rejects_the_wrong_password(monkeypatch):
    path = DOCS / "locked-memo.pdf"
    monkeypatch.setenv("TORTURE_PDF_PASSWORD", "definitely-not-it")
    with pytest.raises(ValueError, match="TORTURE_PDF_PASSWORD"):
        await load_one(PdfLoader(password_env="TORTURE_PDF_PASSWORD"), path)


async def test_password_env_named_but_unset_is_a_config_error(monkeypatch):
    path = DOCS / "locked-memo.pdf"
    monkeypatch.delenv("TORTURE_PDF_PASSWORD", raising=False)
    with pytest.raises(ConfigError, match="TORTURE_PDF_PASSWORD"):
        await load_one(PdfLoader(password_env="TORTURE_PDF_PASSWORD"), path)


async def test_benefits_application_form_field_is_extracted():
    path = DOCS / "benefits-application.pdf"
    loaded = await load_one(PdfLoader(extraction="text"), path)
    assert "## Form fields" in loaded.text
    assert TORTURE_FORM_FIELD in loaded.text
    assert TORTURE_FORM_VALUE in loaded.text
    assert loaded.metadata["form_fields"] == 1
