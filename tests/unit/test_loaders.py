import pytest

from heka.rag import ConfigError, MissingExtraError
from heka.rag.adapters.loaders import (
    CsvLoader,
    DocxLoader,
    HtmlLoader,
    ImageLoader,
    PdfLoader,
    PptxLoader,
    TextLoader,
    XlsxLoader,
    promote_headings,
    strip_repeated_lines,
    to_markdown_table,
)
from pdf_helpers import (
    borderless_table_stream,
    gappy_prose_ops,
    gappy_prose_with_table_stream,
    heading_body_stream,
    make_encrypted_pdf,
    make_pdf,
    make_pdf_from_streams,
    make_pdf_with_form_field,
    table_page_stream,
    three_column_prose_stream,
    two_column_stream,
)


async def load_one(loader, path):
    docs = [d async for d in loader.load(str(path))]
    assert len(docs) == 1
    return docs[0]


def test_markdown_table_helper():
    table = to_markdown_table([["A", "B"], ["1", "x|y"], ["", ""]])
    assert table.splitlines()[0] == "| A | B |"
    assert table.splitlines()[1] == "| --- | --- |"
    assert "x\\|y" in table
    assert to_markdown_table([]) == ""


def test_headings_on_the_first_line_of_a_later_page_are_found():
    text = "First page ends here.\f3.2 Casual leave\nBody text of the second page."
    pages = promote_headings(text).split("\f")
    assert pages[1].startswith("## 3.2 Casual leave")
    assert pages[0] == "First page ends here."


def test_promote_headings_is_conservative():
    text = "\n".join(
        [
            "LEAVE POLICY",  # short all-caps -> heading
            "3.2 Casual leave",  # dotted numbering -> heading
            "3.2.1 Carry over rules",  # deeper level
            "1. Submit the form",  # plain numbered list item -> untouched
            "Employees receive 12 days of leave.",
            "The 2.5 percent rule applies.",  # not a heading
            "HR",  # too short
        ]
    )
    lines = promote_headings(text).split("\n")
    assert lines[0] == "# LEAVE POLICY"
    assert lines[1] == "## 3.2 Casual leave"
    assert lines[2] == "### 3.2.1 Carry over rules"
    assert lines[3:] == text.split("\n")[3:]


async def test_text_and_markdown(tmp_path):
    path = tmp_path / "policy.md"
    path.write_text("# Travel Policy\n\nBook early.", encoding="utf-8")
    document = await load_one(TextLoader(), path)
    assert document.metadata["title"] == "Travel Policy"
    assert document.metadata["file_type"] == "text"
    assert document.source == "policy.md"
    assert "Book early." in document.text


async def test_csv_becomes_a_table(tmp_path):
    path = tmp_path / "plans.csv"
    path.write_text("Plan,Cost\nBasic,0\nFamily,90\n", encoding="utf-8")
    document = await load_one(CsvLoader(), path)
    assert "| Plan | Cost |" in document.text
    assert "| Family | 90 |" in document.text


async def test_html_strips_boilerplate_and_keeps_structure(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><title>Holidays</title></head><body><nav>Home | HR</nav>"
        "<h1>Holidays 2025</h1><p>Nine paid holidays.</p><ul><li>New Year</li><li>Labour Day</li></ul>"
        "<table><tr><th>Day</th><th>Date</th></tr><tr><td>New Year</td><td>1 Jan</td></tr></table>"
        "<script>track()</script><footer>Copyright</footer></body></html>",
        encoding="utf-8",
    )
    document = await load_one(HtmlLoader(), path)
    assert document.metadata["title"] == "Holidays"
    assert "# Holidays 2025" in document.text
    assert "- New Year" in document.text
    assert "| New Year | 1 Jan |" in document.text
    for junk in ("Home | HR", "track()", "Copyright"):
        assert junk not in document.text


async def test_docx_headings_lists_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "policy.docx"
    document = docx.Document()
    document.add_heading("Leave Policy", level=1)
    document.add_paragraph("Employees get 12 days.")
    document.add_heading("Sick leave", level=2)
    document.add_paragraph("Bring a certificate.", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Plan", "Cost"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "Family", "90"
    document.save(path)

    loaded = await load_one(DocxLoader(), path)
    assert "# Leave Policy" in loaded.text
    assert "## Sick leave" in loaded.text
    assert "- Bring a certificate." in loaded.text
    assert "| Family | 90 |" in loaded.text


async def test_pptx_slides_titles_and_notes(tmp_path):
    pptx = pytest.importorskip("pptx")
    path = tmp_path / "deck.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[1])
    slide.shapes.title.text = "Onboarding"
    slide.placeholders[1].text = "Collect your laptop on day one."
    slide.notes_slide.notes_text_frame.text = "Mention the IT desk."
    presentation.save(path)

    loaded = await load_one(PptxLoader(), path)
    assert "## Slide 1: Onboarding" in loaded.text
    assert "Collect your laptop on day one." in loaded.text
    assert "Speaker notes: Mention the IT desk." in loaded.text


async def test_xlsx_sheets_become_tables(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "rates.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Rates"
    sheet.append(["City", "Hotel limit"])
    sheet.append(["Tier-1", 180])
    workbook.save(path)

    loaded = await load_one(XlsxLoader(), path)
    assert "## Sheet: Rates" in loaded.text
    assert "| Tier-1 | 180 |" in loaded.text


async def test_pptx_text_inside_grouped_shapes_is_kept(tmp_path):
    pptx = pytest.importorskip("pptx")
    from pptx.util import Inches

    path = tmp_path / "grouped.pptx"
    presentation = pptx.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Process"
    group = slide.shapes.add_group_shape()
    box = group.shapes.add_textbox(Inches(1), Inches(2), Inches(3), Inches(1))
    box.text_frame.text = "Step one: file the form"
    inner = group.shapes.add_group_shape()
    nested = inner.shapes.add_textbox(Inches(1), Inches(3), Inches(3), Inches(1))
    nested.text_frame.text = "Step two: wait 5 days"
    presentation.save(path)

    loaded = await load_one(PptxLoader(), path)
    assert "Step one: file the form" in loaded.text
    assert "Step two: wait 5 days" in loaded.text


async def test_xlsx_uncached_formulas_are_not_silently_blank(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "budget.xlsx"
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Budget"
    sheet.append(["Item", "Cost"])
    sheet.append(["Rent", 1000])
    sheet.append(["Food", 400])
    sheet.append(["Total", "=SUM(B2:B3)"])  # written by a library: no cached result
    workbook.save(path)

    loaded = await load_one(XlsxLoader(), path)
    assert "=SUM(B2:B3)" in loaded.text  # the formula is shown rather than an empty cell


async def test_xlsx_hidden_sheets_are_labelled(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    path = tmp_path / "hidden.xlsx"
    workbook = openpyxl.Workbook()
    workbook.active.title = "Public"
    workbook.active.append(["Rate", 5])
    secret = workbook.create_sheet("Old rates")
    secret.append(["Rate", 3])
    secret.sheet_state = "hidden"
    workbook.save(path)

    loaded = await load_one(XlsxLoader(), path)
    assert "## Sheet: Public" in loaded.text
    assert "## Sheet: Old rates (hidden)" in loaded.text


async def test_docx_headers_and_footers_are_kept_once(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "policy.docx"
    document = docx.Document()
    document.sections[0].header.paragraphs[0].text = "Travel Policy v3 - effective 1 March 2025"
    document.sections[0].footer.paragraphs[0].text = "Confidential - HR use only"
    document.add_paragraph("Book flights 14 days ahead.")
    document.add_page_break()
    document.add_paragraph("Hotels are capped at 180.")
    document.save(path)

    loaded = await load_one(DocxLoader(), path)
    assert "Book flights 14 days ahead." in loaded.text
    assert loaded.text.count("Travel Policy v3 - effective 1 March 2025") == 1
    assert loaded.text.count("Confidential - HR use only") == 1


class FakeOcr:
    """An OCR engine that 'reads' a fixed string and records what it was given."""

    def __init__(self, text="Scanned text read by OCR."):
        self.text = text
        self.images = []

    async def recognize(self, image):
        self.images.append(image)
        return self.text


@pytest.mark.parametrize("extraction", ["layout", "text"])
async def test_pdf_text_pages_and_headings(tmp_path, extraction):
    pytest.importorskip("pypdf")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "policy.pdf"
    path.write_bytes(
        make_pdf(["3.2 Casual leave\nEmployees get 12 days.", "Page two has plenty of text here."])
    )
    loaded = await load_one(PdfLoader(extraction=extraction), path)
    assert loaded.metadata["pages"] == 2
    assert "\f" in loaded.text  # page separator
    assert "## 3.2 Casual leave" in loaded.text
    assert "Employees get 12 days." in loaded.text
    assert "needs_ocr" not in loaded.metadata


async def test_layout_extraction_keeps_tables_as_tables(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "plans.pdf"
    rows = [["Plan", "Cost", "Covers"], ["Basic", "0", "None"], ["Family", "90", "Spouse"]]
    path.write_bytes(
        make_pdf_from_streams([table_page_stream("Health plans", rows, "Prices are monthly.")])
    )
    layout = (await load_one(PdfLoader(extraction="layout"), path)).text
    assert "| Plan | Cost | Covers |" in layout
    assert "| Family | 90 | Spouse |" in layout
    assert (
        layout.index("Health plans")
        < layout.index("| Plan |")
        < layout.index("Prices are monthly.")
    )
    assert layout.count("Family") == 1  # the table is not also repeated as running text

    flattened = (await load_one(PdfLoader(extraction="text"), path)).text
    assert "| Family |" not in flattened  # text mode cannot recover the table structure


async def test_scanned_pdf_pages_are_flagged_without_ocr(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf(["Has text here, plenty of it.", ""]))
    loaded = await load_one(PdfLoader(), path)
    assert loaded.metadata["needs_ocr"] is True
    assert loaded.metadata["pages_without_text"] == 1
    assert "ocr_pages" not in loaded.metadata


async def test_scanned_pages_are_read_with_ocr_when_configured(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "scan.pdf"
    path.write_bytes(
        make_pdf(["Has text here, plenty of it.", "", "A third page that has real text too."])
    )
    ocr = FakeOcr("4.2 Notice period\nThirty days notice is required.")
    loaded = await load_one(PdfLoader(ocr=ocr), path)
    assert len(ocr.images) == 1  # only the blank page was rendered and recognised
    assert loaded.metadata["ocr_pages"] == 1 and "needs_ocr" not in loaded.metadata
    pages = loaded.text.split("\f")
    assert "Thirty days notice" in pages[1] and "## 4.2 Notice period" in pages[1]
    assert "Has text here" in pages[0]  # pages that had text are untouched


async def test_pages_where_ocr_finds_nothing_are_still_flagged(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "blank.pdf"
    path.write_bytes(make_pdf(["Has text here, plenty of it.", ""]))
    loaded = await load_one(PdfLoader(ocr=FakeOcr("")), path)
    assert loaded.metadata["ocr_pages"] == 0
    assert loaded.metadata["needs_ocr"] is True and loaded.metadata["pages_without_text"] == 1


async def test_image_loader_needs_ocr_and_reads_frames(tmp_path):
    pil = pytest.importorskip("PIL.Image")
    path = tmp_path / "scan.png"
    pil.new("RGB", (200, 100), "white").save(path)
    with pytest.raises(ConfigError, match="needs OCR"):
        await load_one(ImageLoader(), path)
    ocr = FakeOcr("Photographed policy text.")
    loaded = await load_one(ImageLoader(ocr=ocr), path)
    assert loaded.text == "Photographed policy text."
    assert loaded.metadata["file_type"] == "image" and loaded.metadata["ocr_pages"] == 1

    tiff = tmp_path / "multi.tiff"
    first = pil.new("RGB", (100, 50), "white")
    first.save(tiff, save_all=True, append_images=[pil.new("RGB", (100, 50), "black")])
    multi = await load_one(ImageLoader(ocr=FakeOcr("page text")), tiff)
    assert multi.metadata["pages"] == 2 and multi.text.count("\f") == 1


async def test_ocr_never_replaces_real_text_with_less(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "short.pdf"
    path.write_bytes(make_pdf(["Appendix A", "A full page with lots of real text on it."]))
    loaded = await load_one(
        PdfLoader(ocr=FakeOcr("A")), path
    )  # OCR reads *less* than the text layer
    assert "Appendix A" in loaded.text.split("\f")[0]
    assert loaded.metadata["ocr_pages"] == 0 and "needs_ocr" not in loaded.metadata


def test_missing_ocr_extra_gives_an_install_hint(monkeypatch):
    import importlib

    from heka.rag import registry

    real = importlib.import_module

    def blocked(name, package=None):
        if name == "rapidocr":
            raise ImportError("No module named 'rapidocr'", name="rapidocr")
        return real(name, package)

    monkeypatch.setattr("heka.rag.adapters._deps.importlib.import_module", blocked)
    with pytest.raises(MissingExtraError, match=r"heka-rag-sdk\[ocr\]"):
        registry.create("ocr", "rapidocr")


async def test_real_rapidocr_reads_rendered_text(tmp_path):
    pytest.importorskip("rapidocr")
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    font_module = pytest.importorskip("PIL.ImageFont")
    from heka.rag import registry

    image = image_module.new("RGB", (1000, 200), "white")
    draw = draw_module.Draw(image)
    try:
        font = font_module.truetype("arial.ttf", 30)
    except OSError:
        font = font_module.load_default(30)
    draw.text((30, 30), "Employees receive 12 days of casual leave.", fill="black", font=font)
    draw.text((30, 100), "Unused days do not carry over.", fill="black", font=font)
    path = tmp_path / "scan.png"
    image.save(path)

    engine = registry.create("ocr", "rapidocr")
    loaded = await load_one(ImageLoader(ocr=engine), path)
    text = loaded.text.lower()
    assert "12 days of casual leave" in text and "do not carry over" in text
    assert text.index("casual leave") < text.index("carry over")  # reading order is kept


async def test_real_ocr_recovers_a_scanned_pdf(tmp_path):
    pytest.importorskip("rapidocr")
    pytest.importorskip("pdfplumber")
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    font_module = pytest.importorskip("PIL.ImageFont")
    from heka.rag import registry

    image = image_module.new("RGB", (1240, 400), "white")  # an image-only page: no text layer
    try:
        font = font_module.truetype("arial.ttf", 34)
    except OSError:
        font = font_module.load_default(34)
    draw_module.Draw(image).text((40, 60), "Notice period is thirty days.", fill="black", font=font)
    path = tmp_path / "scanned.pdf"
    image.save(path, "PDF", resolution=150)

    without = await load_one(PdfLoader(), path)
    assert without.metadata["needs_ocr"] is True
    with_ocr = await load_one(PdfLoader(ocr=registry.create("ocr", "rapidocr")), path)
    assert "notice period is thirty days" in with_ocr.text.lower()
    assert with_ocr.metadata["ocr_pages"] == 1 and "needs_ocr" not in with_ocr.metadata


# -- PDF resolver: font-size headings, header/footer stripping, columns, passwords, tables, forms --
# Each fix is additive to the pipeline above: every existing PDF test in this file still exercises
# the unchanged code paths (short documents, ruled tables, single-column pages) untouched.


async def test_font_size_heading_is_detected_without_matching_any_regex(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "irs.pdf"
    body = [
        "Employers must withhold federal income tax from wages paid to employees.",
        "The amount withheld depends on the employee's Form W-4 and pay frequency.",
        "Deposit the withheld tax according to your deposit schedule.",
    ]
    # Real, mixed-case wording that none of the numbered/section/ALL-CAPS regexes would catch -
    # the confirmed root cause this fix exists for (a real IRS publication heading).
    path.write_bytes(make_pdf_from_streams([heading_body_stream("Federal Income Tax Withholding", body)]))
    layout = await load_one(PdfLoader(extraction="layout"), path)
    assert "# Federal Income Tax Withholding" in layout.text

    # extraction="text" has no font data, so it falls back to regex only, as documented: the
    # heading is read fine, just not promoted. This is the pre-existing, unchanged behaviour.
    flattened = await load_one(PdfLoader(extraction="text"), path)
    assert "Federal Income Tax Withholding" in flattened.text
    assert "# Federal Income Tax Withholding" not in flattened.text


async def test_bold_heading_at_body_text_size_is_detected(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "policy.pdf"
    body = [
        "This paragraph explains the policy in plain, regular-weight text.",
        "It continues for a couple more lines so body size is unambiguous.",
        "A third line keeps the body text the clear majority on the page.",
    ]
    path.write_bytes(
        make_pdf_from_streams([heading_body_stream("Remote Work Guidelines", body, bold=True, size=11)])
    )
    loaded = await load_one(PdfLoader(extraction="layout"), path)
    assert "# Remote Work Guidelines" in loaded.text


async def test_font_heading_detection_does_not_promote_ordinary_emphasis(tmp_path):
    """A short bold run inside an otherwise normal line must not turn the whole line into a heading -
    only a line that is *itself* set larger/bolder than the page's body text should qualify."""
    pytest.importorskip("pdfplumber")
    path = tmp_path / "memo.pdf"
    body = [
        "Please submit the form by Friday.",
        "Late submissions will not be accepted this quarter.",
        "Contact HR with any questions about the process.",
    ]
    path.write_bytes(make_pdf_from_streams([heading_body_stream("Reminder", body, size=11)]))
    loaded = await load_one(PdfLoader(extraction="layout"), path)
    assert "# Reminder" not in loaded.text  # same size as body, not bold: not a heading


def test_strip_repeated_lines_needs_several_pages():
    pages = ["Header\nBody one.", "Header\nBody two.", "Header\nBody three."]
    assert strip_repeated_lines(pages) == pages  # only 3 pages: left alone


def test_strip_repeated_lines_removes_headers_and_normalises_page_numbers():
    pages = [f"ACME CORP - CONFIDENTIAL\nBody content {i}.\nPage {i} of 5" for i in range(1, 6)]
    cleaned = strip_repeated_lines(pages)
    assert all("ACME CORP" not in page for page in cleaned)
    assert all("Page" not in page for page in cleaned)  # "Page # of #" matched on every page
    assert all(f"Body content {i}." in cleaned[i - 1] for i in range(1, 6))


async def test_running_header_and_footer_are_stripped_from_a_real_pdf(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "handbook.pdf"
    pages = [
        f"ACME CORP - CONFIDENTIAL\nSection {i} covers a different topic each page.\nPage {i} of 4"
        for i in range(1, 5)
    ]
    path.write_bytes(make_pdf(pages))
    loaded = await load_one(PdfLoader(extraction="text"), path)
    assert "ACME CORP - CONFIDENTIAL" not in loaded.text
    assert "Page 1 of 4" not in loaded.text
    for i in range(1, 5):
        assert f"Section {i} covers a different topic each page." in loaded.text


def test_reading_order_two_columns_reset_at_a_spanning_heading():
    from heka.rag.adapters.loaders import _reading_order

    records = [
        {"top": 0, "x0": 72, "x1": 500, "text": "Title", "size": 16, "bold": True},  # spans
        {"top": 20, "x0": 72, "x1": 200, "text": "L1", "size": 11, "bold": False},
        {"top": 40, "x0": 72, "x1": 200, "text": "L2", "size": 11, "bold": False},
        {"top": 60, "x0": 72, "x1": 200, "text": "L3", "size": 11, "bold": False},
        {"top": 25, "x0": 320, "x1": 450, "text": "R1", "size": 11, "bold": False},
        {"top": 45, "x0": 320, "x1": 450, "text": "R2", "size": 11, "bold": False},
        {"top": 65, "x0": 320, "x1": 450, "text": "R3", "size": 11, "bold": False},
    ]
    ordered = [r["text"] for r in _reading_order(records, page_width=612)]
    assert ordered == ["Title", "L1", "L2", "L3", "R1", "R2", "R3"]


def test_reading_order_leaves_a_single_column_page_alone():
    from heka.rag.adapters.loaders import _reading_order

    records = [
        {"top": i * 10, "x0": 72, "x1": 300, "text": f"line{i}", "size": 11, "bold": False}
        for i in range(8)
    ]
    ordered = _reading_order(records, page_width=612)
    assert [r["text"] for r in ordered] == [f"line{i}" for i in range(8)]


async def test_two_column_page_is_read_left_column_then_right_column(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "briefing.pdf"
    left = [f"Left para {i} text." for i in range(1, 5)]
    right = [f"Right para {i} text." for i in range(1, 5)]
    path.write_bytes(make_pdf_from_streams([two_column_stream("Quarterly Briefing", left, right)]))
    loaded = await load_one(PdfLoader(extraction="layout"), path)
    text = loaded.text
    assert text.index("Quarterly Briefing") < text.index(left[0])
    for a, b in zip(left, left[1:], strict=False):
        assert text.index(a) < text.index(b)  # left column stays in top-to-bottom order
    for a, b in zip(right, right[1:], strict=False):
        assert text.index(a) < text.index(b)  # right column stays in top-to-bottom order
    assert text.index(left[-1]) < text.index(right[0])  # the whole left column comes first


async def test_borderless_table_is_recovered_by_the_fallback_strategy(tmp_path):
    pytest.importorskip("pdfplumber")
    path = tmp_path / "roster.pdf"
    rows = [
        ["Name", "Age", "City"],
        ["Alice", "34", "Boston"],
        ["Bob", "29", "Denver"],
        ["Carol", "41", "Austin"],
    ]
    path.write_bytes(make_pdf_from_streams([borderless_table_stream(rows)]))
    loaded = await load_one(PdfLoader(extraction="layout"), path)
    assert "| Name | Age | City |" in loaded.text
    assert "| Alice | 34 | Boston |" in loaded.text
    assert "| Carol | 41 | Austin |" in loaded.text


@pytest.mark.parametrize("extraction", ["layout", "text"])
async def test_password_protected_pdf_without_password_errors_clearly(tmp_path, extraction):
    pytest.importorskip("pypdf")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "locked.pdf"
    path.write_bytes(make_encrypted_pdf(["Confidential content here."], "letmein"))
    with pytest.raises(ValueError, match="password-protected"):
        await load_one(PdfLoader(extraction=extraction), path)


@pytest.mark.parametrize("extraction", ["layout", "text"])
async def test_password_protected_pdf_is_read_with_the_right_password(tmp_path, monkeypatch, extraction):
    pytest.importorskip("pypdf")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "locked.pdf"
    path.write_bytes(make_encrypted_pdf(["Confidential content here.", "Second page too."], "letmein"))
    monkeypatch.setenv("TEST_PDF_PASSWORD", "letmein")
    loaded = await load_one(PdfLoader(extraction=extraction, password_env="TEST_PDF_PASSWORD"), path)
    assert "Confidential content here." in loaded.text


async def test_wrong_password_gives_a_clear_error_naming_the_env_var(tmp_path, monkeypatch):
    pytest.importorskip("pypdf")
    pytest.importorskip("pdfplumber")
    path = tmp_path / "locked.pdf"
    path.write_bytes(make_encrypted_pdf(["Confidential content here."], "letmein"))
    monkeypatch.setenv("TEST_PDF_PASSWORD", "wrong-guess")
    with pytest.raises(ValueError, match="TEST_PDF_PASSWORD"):
        await load_one(PdfLoader(password_env="TEST_PDF_PASSWORD"), path)


async def test_password_env_set_but_empty_raises_config_error(tmp_path, monkeypatch):
    pytest.importorskip("pypdf")
    path = tmp_path / "locked.pdf"
    path.write_bytes(make_encrypted_pdf(["Confidential content here."], "letmein"))
    monkeypatch.setenv("TEST_PDF_PASSWORD", "")
    with pytest.raises(ConfigError, match="TEST_PDF_PASSWORD"):
        await load_one(PdfLoader(password_env="TEST_PDF_PASSWORD"), path)


async def test_acroform_fields_are_appended_as_an_appendix(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "application.pdf"
    path.write_bytes(make_pdf_with_form_field("Employee application form", "EmployeeName", "Jane Doe"))
    loaded = await load_one(PdfLoader(extraction="text"), path)
    assert "## Form fields" in loaded.text
    assert "EmployeeName" in loaded.text
    assert "Jane Doe" in loaded.text
    assert loaded.metadata["form_fields"] == 1


def _words(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{i}{c}" for i in range(count) for c in "abc"]


async def test_prose_with_wide_gaps_is_not_turned_into_a_table_and_loses_no_words(tmp_path):
    """Regression: justified multi-column prose was read as a 'table' and most of the page was dropped."""
    pytest.importorskip("pdfplumber")
    path = tmp_path / "prose.pdf"
    path.write_bytes(make_pdf_from_streams(["\n".join(gappy_prose_ops(14)).encode()]))
    text = (await load_one(PdfLoader(extraction="layout"), path)).text
    assert "| --- |" not in text  # no table invented
    for word in _words("w", 14):
        assert word in text, f"{word} was dropped"


async def test_prose_above_a_real_borderless_table_keeps_both(tmp_path):
    pytest.importorskip("pdfplumber")
    rows = [["Unit", "Monthly Rent", "Occupant"], ["101", "1450", "Vacant"], ["102", "1525", "Nakamura"], ["201", "1610", "Vacant"]]
    path = tmp_path / "mixed.pdf"
    path.write_bytes(make_pdf_from_streams([gappy_prose_with_table_stream(10, rows)]))
    text = (await load_one(PdfLoader(extraction="layout"), path)).text
    assert "| Unit | Monthly Rent | Occupant |" in text and "| 102 | 1525 | Nakamura |" in text
    for word in _words("w", 10):
        assert word in text, f"{word} was dropped"


async def test_a_table_rendering_can_never_lose_the_pages_words(tmp_path, monkeypatch):
    """The safety net: even if a table heuristic misfires and claims the whole page, the words survive."""
    pytest.importorskip("pdfplumber")
    from heka.rag.adapters import loaders

    def whole_page_table(page):
        words = page.extract_words()
        bbox = (0.0, 0.0, float(page.width), float(page.height))
        return [(bbox, [[words[0]["text"], "", ""], ["x", "y", "z"], ["p", "q", "r"]])]  # keeps almost nothing

    monkeypatch.setattr(loaders, "_borderless_tables", whole_page_table)
    path = tmp_path / "prose.pdf"
    path.write_bytes(make_pdf_from_streams(["\n".join(gappy_prose_ops(8)).encode()]))
    text = (await load_one(PdfLoader(extraction="layout"), path)).text
    for word in _words("w", 8):
        assert word in text, f"{word} was dropped"


async def test_three_columns_of_running_text_are_not_read_as_a_table(tmp_path):
    """Regression: aligned prose columns (the Federal Register's layout) were printed as a 3-column
    'table' of unrelated sentence fragments. Text must be kept, and no table invented."""
    pytest.importorskip("pdfplumber")
    path = tmp_path / "gazette.pdf"
    path.write_bytes(make_pdf_from_streams([three_column_prose_stream(12)]))
    text = (await load_one(PdfLoader(extraction="layout"), path)).text
    assert "| --- |" not in text
    for col in range(3):
        for i in range(12):
            for k in range(5):
                assert f"c{col}l{i}w{k}" in text


async def test_html_inline_markup_adds_no_spaces_to_the_text(tmp_path):
    """Regression: every inline element got a space around it ("noncitizen ;", "You ' ll", "$ 25 .50"), so
    the indexed text no longer matched what a model quotes. Block elements must still be separated."""
    pytest.importorskip("bs4")
    path = tmp_path / "inline.html"
    path.write_text(
        "<html><body><h1>Aid <em>2025</em></h1>"
        "<p>An eligible <a href='/x'>noncitizen</a>; a <strong>Social Security number</strong>, and $<span>25</span>.50.</p>"
        "<p>You<em>'</em>ll need 10<sup>th</sup> grade.<br>Second line.</p>"
        "<ul><li>Item <b>one</b>, also<ul><li>Nested</li></ul></li></ul>"
        "<table><tr><td>Cost <i>each</i></td><td>$<b>5</b></td></tr></table>"
        "<div>Block A</div><div>Block B</div></body></html>",
        encoding="utf-8",
    )
    text = (await load_one(HtmlLoader(), path)).text
    assert "An eligible noncitizen; a Social Security number, and $25.50." in text
    assert "You'll need 10th grade. Second line." in text
    assert "- Item one, also" in text and "- Nested" in text
    assert "| Cost each | $5 |" in text
    assert "# Aid 2025" in text
    assert "Block A" in text and "Block B" in text  # separate blocks stay separate


async def test_html_text_directly_inside_containers_is_kept(tmp_path):
    """Regression: text written directly in a <div>/<section> (not inside a <p>) was silently dropped."""
    pytest.importorskip("bs4")
    path = tmp_path / "bare.html"
    path.write_text(
        "<html><body><h1>Title</h1><div>Fees are waived for veterans.</div>"
        "<div><span>Call 1-800-555-0100 for help.</span></div>"
        "<section>Office hours are 9 to 5.<p>A real paragraph.</p>Trailing text after it.</section>"
        "<nav>Menu</nav></body></html>",
        encoding="utf-8",
    )
    text = (await load_one(HtmlLoader(), path)).text
    for expected in ("Fees are waived for veterans.", "Call 1-800-555-0100 for help.", "Office hours are 9 to 5.", "A real paragraph.", "Trailing text after it."):
        assert expected in text
    assert "Menu" not in text  # navigation is still stripped
