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
    """A PDF with one page per raw content stream (Helvetica available as /F1)."""
    n = len(streams)
    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(n))
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, content in enumerate(streams):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {5 + 2 * i} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>".encode()
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
