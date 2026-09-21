"""OCR engines: read text from scanned pages and images.

RapidOCR runs the PP-OCR models on ONNX Runtime. Its models ship inside the pip package, so it needs
no system install (unlike Tesseract) and no network access. OCR is imperfect: expect occasional
misread characters, which can stop a citation from verifying against the extracted text.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from pydantic import Field

from kbsdk.adapters._deps import Settings, require


class RapidOcrSettings(Settings):
    min_confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Drop text fragments the engine is less sure of"
    )


class RapidOcr:
    settings_model = RapidOcrSettings

    def __init__(self, settings: RapidOcrSettings | None = None) -> None:
        self.settings = settings or RapidOcrSettings()
        module = require("rapidocr", stage="ocr", provider="rapidocr", extra="ocr")
        logging.getLogger("RapidOCR").setLevel(logging.WARNING)  # it logs every model load at INFO
        self._cls = module.RapidOCR
        self._engine: Any = None
        self._lock = threading.Lock()  # one engine instance, not assumed thread-safe

    def _run(self, image: Any) -> str:
        with self._lock:
            if self._engine is None:
                self._engine = self._cls()
            result = self._engine(image)
        texts = getattr(result, "txts", None)
        if not texts:
            return ""
        boxes, scores = result.boxes, result.scores
        rows = [
            (float(box[:, 1].min()), float(box[:, 0].min()), text)
            for box, text, score in zip(boxes, texts, scores, strict=True)
            if score >= self.settings.min_confidence
        ]
        rows.sort()  # top to bottom, then left to right
        return "\n".join(text for _, _, text in rows)

    async def recognize(self, image: Any) -> str:
        """`image` may be a PIL image, a NumPy array or a file path."""
        return await asyncio.to_thread(self._run, image)
