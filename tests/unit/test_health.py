import pytest

from kbsdk import KnowledgeBase
from kbsdk.cli import main
from kbsdk.pipelines.health import (
    FileHealth,
    HealthReport,
    count_headings,
    count_tables,
    cross_document_notes,
    file_health,
    quality_flags,
    repeated_line_share,
)

PROSE = (
    "Employees receive twelve days of casual leave each calendar year, credited on the first of "
    "January. Unused days do not carry over. Requests should be submitted through the HR portal at "
    "least three working days in advance, and managers respond within two working days. "
) * 3


def flags_for(text, loader="text", **meta):
    return file_health("doc", loader, text, meta, [400] * 4).flags


# -- the small counters --------------------------------------------------------------------------


def test_tables_are_counted_as_blocks_and_headings_by_markdown_marker():
    text = "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\ntext\n\n## Next\n\n| c |\n"
    assert count_tables(text) == 2 and count_headings(text) == 2
    assert count_tables("no tables") == 0 and count_headings("#hashtag not a heading") == 0


def test_repeated_page_furniture_is_measured():
    page = (
        "Acme Corp confidential - internal use only\nSome unique sentence number {n} about leave.\n"
    )
    text = "".join(page.format(n=n) for n in range(10))
    assert repeated_line_share(text) > 0.3
    assert repeated_line_share(PROSE) == 0.0 and repeated_line_share("") == 0.0


def test_repeated_markdown_structure_is_not_page_furniture():
    """Found on 80 real GOV.UK guides: table separators and repeated headings inflated the share."""
    structure = (
        "## What you can claim\n| --- | --- |\n| a | b |\nUnique sentence {n} about jury service.\n"
    )
    text = "".join(structure.format(n=n) for n in range(8))
    assert repeated_line_share(text) == 0.0
    real = "Call the Jury Central Summoning Bureau on the number above.\n" * 4
    assert repeated_line_share(real + "one unique line of ordinary text here") > 0.5


# -- text quality --------------------------------------------------------------------------------


def test_healthy_prose_raises_no_quality_flag():
    assert quality_flags(PROSE, loader="text") == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (" ".join(PROSE.replace(" ", "")), "spaced apart"),  # T h e  c o m p a n y
        (PROSE + chr(0xFFFD) * 20, "unreadable characters"),
        (PROSE.replace(" ", ""), "average word length"),  # words glued together
        ("!@#$%^&*()_+=- " * 60 + "ab ", "non-letters"),
        (
            (
                "Confidential draft - do not distribute\nA unique line {n} of body text.\n" * 1
            ).format(n=1)
            + "".join(
                f"Confidential draft - do not distribute\nBody sentence number {n} here.\n"
                for n in range(12)
            ),
            "header/footer",
        ),
    ],
)
def test_noise_is_flagged(text, expected):
    assert any(expected in flag for flag in quality_flags(text, loader="text")), expected


def test_spreadsheets_are_allowed_to_be_mostly_numbers():
    numbers = "| 1 | 2 | 3 |\n" * 100 + "1234 5678 9012 " * 40
    assert quality_flags(numbers, loader="csv") == []


def test_too_little_text_is_left_to_the_size_checks():
    assert quality_flags("tiny", loader="text") == []


# -- per-file flags ------------------------------------------------------------------------------


def test_a_healthy_file_has_statistics_and_no_flags():
    text = "# Leave\n\n" + PROSE + "\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    health = file_health("leave.md", "text", text, {}, [420, 380, 400, 390])
    assert health.flags == [] and health.tables == 1 and health.headings == 1
    assert health.chunks == 4 and health.median_chunk_chars == 395 and health.characters > 400


def test_an_empty_file_is_flagged_and_needs_no_other_checks():
    assert flags_for("   \n") == ["no text was extracted"]


def test_scanned_pages_are_reported_with_the_right_remedy():
    without_ocr = file_health("s.pdf", "pdf", PROSE, {"pages": 5, "pages_without_text": 3}, [400])
    assert any(
        "3 page(s) have no text layer" in f and "set knowledge.ocr" in f for f in without_ocr.flags
    )
    with_ocr = file_health(
        "s.pdf", "pdf", PROSE, {"pages": 5, "pages_without_text": 3}, [400], ocr_configured=True
    )
    assert any("OCR found nothing" in f for f in with_ocr.flags)
    read = file_health("s.pdf", "pdf", PROSE, {"pages": 5, "ocr_pages": 2}, [400])
    assert any("2 page(s) were read with OCR" in f for f in read.flags)


def test_pages_with_almost_no_text_are_flagged_for_pdfs_only():
    thin = file_health("f.pdf", "pdf", "Figure 1. Chart of results. " * 4, {"pages": 10}, [100])
    assert any("characters per page" in f for f in thin.flags)
    short = file_health("n.md", "text", "Short note about parking.", {}, [25])
    assert any("only 25 characters" in f for f in short.flags)
    table = file_health("bands.md", "text", "| band | pay |\n|---|---|\n| L1 | 50k |\n", {}, [40])
    assert not any("only" in f for f in table.flags)  # a short table is a legitimate document


def test_short_sections_are_normal_but_line_by_line_chunks_are_not():
    assert not any(
        "fragmented" in f
        for f in file_health("a.md", "text", "# H\n\n" + PROSE, {}, [110] * 9).flags
    )


def test_a_long_document_without_headings_is_flagged_but_a_spreadsheet_is_not():
    long_plain = "\n\n".join([PROSE] * 8)
    assert any("no headings" in f for f in flags_for(long_plain))
    assert not any("no headings" in f for f in flags_for(long_plain, loader="csv"))
    assert not any("no headings" in f for f in flags_for("# H\n\n" + long_plain))


def test_fragmented_chunks_are_flagged():
    health = file_health("d.md", "text", "# H\n\n" + PROSE, {}, [60, 70, 80, 50, 90, 65])
    assert any("fragmented" in f for f in health.flags)


def test_page_counts_from_several_documents_add_up_in_the_summary_table():
    report = HealthReport(
        files=[
            FileHealth(path="a.pdf", loader="pdf", characters=1000, pages=4, chunks=3),
            FileHealth(path="b.md", loader="text", characters=500, chunks=2, flags=["x"]),
            FileHealth(path="c.docx", loader="docx", error="BadZipFile: nope"),
        ]
    )
    text = report.summary()
    assert "3 files | 5 chunks | 1,500 characters | 2 need a look" in text
    assert "c.docx: could not be read: BadZipFile: nope" in text and "b.md: x" in text
    assert HealthReport().summary() == "no documents found"


# -- across documents ----------------------------------------------------------------------------


def test_identical_content_and_version_families_are_noticed():
    notes = cross_document_notes(
        [
            ("policy_v1.pdf", "Old policy text about leave."),
            ("policy_v2.pdf", "New policy text about leave, now longer."),
            ("holidays-2025.md", "Holidays for 2025."),
            ("holidays-2026.md", "Holidays for 2026."),
            ("handbook.pdf", "Same words in both."),
            (
                "handbook (1).pdf",
                "Same   words in  both.",
            ),  # whitespace differs, content is the same
            ("supplement-france.md", "France."),
            ("supplement-germany.md", "Germany."),
            ("code-of-conduct.md", "Conduct."),
        ]
    )
    joined = "\n".join(notes)
    assert "identical content: handbook (1).pdf, handbook.pdf" in joined
    assert "possible versions of one document: policy_v1.pdf, policy_v2.pdf" in joined
    assert "holidays-2025.md, holidays-2026.md" in joined
    assert "supplement" not in joined and "conduct" not in joined
    assert (
        sum("handbook" in note for note in notes) == 1
    )  # copies: reported once, not as "versions" too


def test_no_notes_for_unrelated_documents_and_empty_files_are_ignored():
    assert cross_document_notes([("a.md", "alpha"), ("b.md", "beta")]) == []
    assert cross_document_notes([("a.md", ""), ("b.md", "")]) == []


# -- through the knowledge base and CLI ------------------------------------------------------------


async def test_analyze_reads_every_file_without_touching_the_index(make_config, docs_dir):
    (docs_dir / "empty.md").write_text("   \n", encoding="utf-8")
    (docs_dir / "broken.docx").write_bytes(b"not a real docx")
    (docs_dir / "data.bin").write_bytes(b"\x00")
    kb = KnowledgeBase(make_config())
    report = await kb.aanalyze()

    by_path = {f.path: f for f in report.files}
    assert set(by_path) == {"leave.md", "expenses.md", "empty.md", "broken.docx"}
    assert by_path["leave.md"].chunks > 0 and by_path["leave.md"].headings >= 2
    assert by_path["empty.md"].flags == ["no text was extracted"]
    assert by_path["broken.docx"].error and not by_path["broken.docx"].chunks
    assert any("unsupported types" in note and "data.bin" in note for note in report.skipped)
    assert await kb.acount() == 0 and not (kb.dir / "manifest.json").exists()  # nothing was indexed


async def test_analyze_flags_a_scanned_pdf_and_duplicate_files(make_config, docs_dir):
    pytest.importorskip("pdfplumber")
    from pdf_helpers import make_pdf

    (docs_dir / "scan.pdf").write_bytes(
        make_pdf(["A page with real text on it, plenty of it here.", ""])
    )
    (docs_dir / "leave-copy.md").write_text(
        (docs_dir / "leave.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    report = await KnowledgeBase(make_config()).aanalyze()
    scan = next(f for f in report.files if f.path == "scan.pdf")
    assert scan.pages == 2 and scan.pages_without_text == 1
    assert any("no text layer" in flag for flag in scan.flags)
    assert any(
        "identical content" in n and "leave.md" in n and "leave-copy.md" in n
        for n in report.cross_document
    )


def test_sync_analyze_matches_async(make_config):
    assert {f.path for f in KnowledgeBase(make_config()).analyze().files} == {
        "leave.md",
        "expenses.md",
    }


@pytest.fixture
def project(tmp_path):
    from conftest import CORPUS

    docs = tmp_path / "docs"
    docs.mkdir()
    for name, text in CORPUS.items():
        (docs / name).write_text(text, encoding="utf-8")
    config = tmp_path / "agent.yaml"
    config.write_text(
        "name: Health Agent\nembedder: {provider: hashing}\n"
        "knowledge:\n  sources: [{location: ./docs}]\n  persist_dir: ./idx\n"
        "generation: {llm: {provider: scripted}}\n",
        encoding="utf-8",
    )
    return tmp_path, str(config)


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_report_only_indexes_nothing(project, capsys):
    root, config = project
    code, out, _ = run(capsys, "ingest", "-c", config, "--report-only")
    assert code == 0 and "documents: 2 files" in out and "leave.md" in out
    assert "files:" not in out.split("documents:")[0]  # no ingest summary was printed
    assert not (root / "idx").exists()  # nothing written, not even an empty index folder


def test_cli_report_follows_the_ingest_summary(project, capsys):
    root, config = project
    code, out, _ = run(capsys, "ingest", "-c", config, "--report")
    assert code == 0 and out.index("2 seen") < out.index("documents: 2 files")
    assert (root / "idx" / "health-agent" / "manifest.json").exists()


def test_cli_without_report_flags_prints_only_the_ingest_summary(project, capsys):
    _, config = project
    _, out, _ = run(capsys, "ingest", "-c", config)
    assert "2 seen" in out and "documents:" not in out
