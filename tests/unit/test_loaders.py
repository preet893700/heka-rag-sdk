import pytest

from kbsdk import ConfigError, MissingExtraError
from kbsdk.adapters.loaders import (
    CsvLoader,
    DocxLoader,
    HtmlLoader,
    ImageLoader,
    PdfLoader,
    PptxLoader,
    TextLoader,
    XlsxLoader,
    promote_headings,
    to_markdown_table,
)
from pdf_helpers import make_pdf, make_pdf_from_streams, table_page_stream


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

    from kbsdk import registry

    real = importlib.import_module

    def blocked(name, package=None):
        if name == "rapidocr":
            raise ImportError("No module named 'rapidocr'", name="rapidocr")
        return real(name, package)

    monkeypatch.setattr("kbsdk.adapters._deps.importlib.import_module", blocked)
    with pytest.raises(MissingExtraError, match=r"kbsdk\[ocr\]"):
        registry.create("ocr", "rapidocr")


async def test_real_rapidocr_reads_rendered_text(tmp_path):
    pytest.importorskip("rapidocr")
    image_module = pytest.importorskip("PIL.Image")
    draw_module = pytest.importorskip("PIL.ImageDraw")
    font_module = pytest.importorskip("PIL.ImageFont")
    from kbsdk import registry

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
    from kbsdk import registry

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
