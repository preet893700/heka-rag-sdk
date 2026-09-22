"""Hand-built PDFs for tests, so no PDF-writing dependency is needed."""

from __future__ import annotations


def _assemble(objects: list[bytes]) -> bytes:
    out = b"%PDF-1.4\n"
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return out


def make_pdf_from_streams(streams: list[bytes]) -> bytes:
    """A PDF with one page per raw content stream (Helvetica available as /F1, bold as /F2)."""
    n = len(streams)
    kids = " ".join(f"{5 + 2 * i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    for i, content in enumerate(streams):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {6 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> >> >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream")
    return _assemble(objects)


def make_pdf(pages: list[str]) -> bytes:
    """A PDF whose pages contain the given text (newlines become separate lines); "" = blank page."""
    streams = []
    for text in pages:
        lines = "".join(f"({line}) Tj T* " for line in text.split("\n")) if text else ""
        streams.append(f"BT /F1 12 Tf 14 TL 72 720 Td {lines}ET".encode() if text else b"")
    return make_pdf_from_streams(streams)


def heading_body_stream(
    heading: str, body_lines: list[str], *, bold: bool = False, size: int = 18
) -> bytes:
    """A page with one heading line (optionally bold, at `size`) followed by normal 11pt body text -
    for testing font-size/weight heading detection, where the wording itself is not heading-like."""
    font = "F2" if bold else "F1"
    ops = [f"BT /{font} {size} Tf 72 720 Td ({heading}) Tj ET"]
    y = 690
    for line in body_lines:
        ops.append(f"BT /F1 11 Tf 72 {y} Td ({line}) Tj ET")
        y -= 18
    return "\n".join(ops).encode()


def two_column_stream(title: str, left: list[str], right: list[str]) -> bytes:
    """A page with a full-width title over two side-by-side columns of text. Keep each column's
    lines short (well under ~30 characters at 11pt) so a clear gap survives between them at these
    x-positions on a standard 612pt-wide page."""
    ops = [f"BT /F2 16 Tf 72 750 Td ({title}) Tj ET"]
    y = 700
    for line in left:
        ops.append(f"BT /F1 11 Tf 72 {y} Td ({line}) Tj ET")
        y -= 18
    y = 700
    for line in right:
        ops.append(f"BT /F1 11 Tf 380 {y} Td ({line}) Tj ET")
        y -= 18
    return "\n".join(ops).encode()


def borderless_table_stream(rows: list[list[str]]) -> bytes:
    """A whitespace-aligned table with no ruled lines, so the default ruled-line detector misses it."""
    x_positions = [72, 220, 380]
    ops = []
    y = 720
    for row in rows:
        for cell, x in zip(row, x_positions, strict=False):
            ops.append(f"BT /F1 11 Tf {x} {y} Td ({cell}) Tj ET")
        y -= 20
    return "\n".join(ops).encode()


def make_encrypted_pdf(pages: list[str], password: str) -> bytes:
    """`make_pdf`, then encrypted with a user password (needs pypdf, imported lazily)."""
    import io

    import pypdf

    reader = pypdf.PdfReader(io.BytesIO(make_pdf(pages)))
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password=password)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def make_pdf_with_form_field(page_text: str, field_name: str, field_value: str) -> bytes:
    """A one-page PDF with an AcroForm text field filled in (built by hand: pypdf's writer has no
    high-level "add a form field" helper, so this constructs the /AcroForm dict directly)."""
    content = f"BT /F1 12 Tf 72 720 Td ({page_text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R /AcroForm 6 0 R >>",
        b"<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R "
        b"/Resources << /Font << /F1 3 0 R >> >> /Annots [7 0 R] >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        b"<< /Fields [7 0 R] >>",
        f"<< /Type /Annot /Subtype /Widget /FT /Tx /T ({field_name}) /V ({field_value}) "
        f"/Rect [100 700 300 720] /P 4 0 R >>".encode(),
    ]
    return _assemble(objects)


def table_page_stream(heading: str, rows: list[list[str]], footer: str) -> bytes:
    """A page with a heading, a ruled table (so table detectors find it) and a footer line."""
    col_width, row_height, x0, y0 = 120, 22, 72, 690
    n_rows, n_cols = len(rows), len(rows[0])
    ops = [f"BT /F1 14 Tf {x0} 720 Td ({heading}) Tj ET", "0.8 w"]
    for r in range(n_rows + 1):
        y = y0 - r * row_height
        ops.append(f"{x0} {y} m {x0 + n_cols * col_width} {y} l S")
    for c in range(n_cols + 1):
        x = x0 + c * col_width
        ops.append(f"{x} {y0} m {x} {y0 - n_rows * row_height} l S")
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            ops.append(
                f"BT /F1 10 Tf {x0 + c * col_width + 6} {y0 - r * row_height - 15} Td ({cell}) Tj ET"
            )
    ops.append(f"BT /F1 12 Tf {x0} {y0 - n_rows * row_height - 40} Td ({footer}) Tj ET")
    return "\n".join(ops).encode()
