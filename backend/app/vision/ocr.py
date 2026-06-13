"""Text recognition via Tesseract OCR.

Reads text (product labels / SKUs) from image crops — normally the
bounding-box regions the detector produced. Isolated behind a small class for
the same reasons as the detector: it's mockable in tests and ``pytesseract``
is imported lazily so the module loads without Tesseract present.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OcrResult:
    text: str          # Recognised text, "" if nothing legible was found.
    confidence: float  # Mean per-word confidence in [0, 1].


class OcrReader:
    """Runs Tesseract over image regions."""

    def __init__(self, tesseract_cmd: str = "tesseract") -> None:
        self._tesseract_cmd = tesseract_cmd
        self._configured = False  # Ensures we point pytesseract at the binary once.

    def configure(self) -> None:
        """Tell pytesseract which tesseract binary to call. Idempotent."""
        if self._configured:
            return
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
        self._configured = True

    def read(self, image) -> OcrResult:
        """Extract text and a mean confidence from an image crop.

        We use ``image_to_data`` (not ``image_to_string``) because it returns a
        per-word confidence alongside each word. Tesseract reports confidence
        as 0-100 and uses -1 for non-text boxes, so we drop the -1s, average
        the rest, and rescale to 0-1 to match the detector's convention. That
        single 0-1 number is what reconciliation later weighs.
        """
        self.configure()
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(image, output_type=Output.DICT)

        words: list[str] = []
        confidences: list[float] = []
        for text, conf in zip(data["text"], data["conf"]):
            text = text.strip()
            conf = float(conf)
            if text and conf >= 0:  # Skip blanks and Tesseract's -1 "no text" rows.
                words.append(text)
                confidences.append(conf)

        if not words:
            return OcrResult(text="", confidence=0.0)
        return OcrResult(
            text=" ".join(words),
            confidence=sum(confidences) / len(confidences) / 100.0,  # 0-100 -> 0-1
        )
