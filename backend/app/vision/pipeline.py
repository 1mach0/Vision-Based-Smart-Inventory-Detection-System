"""Vision pipeline — image bytes in, structured observations out.

This is the seam between the messy world of pixels and the clean world of
domain objects. It owns the orchestration (decode -> detect -> crop -> OCR) and
nothing else: it doesn't know about databases, HTTP, or confidence thresholds.
The API calls ``process`` with the raw upload bytes and gets back a list of
``Observation``s that the domain and persistence layers understand.

Accepting raw ``bytes`` (rather than a decoded array) is deliberate: it keeps
all image-decoding and the heavy cv2/numpy imports inside this module, so the
API layer stays free of vision dependencies and is trivial to test with a fake
pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass

from .detector import Detector
from .ocr import OcrReader, OcrResult


@dataclass(frozen=True)
class Observation:
    """A detected item plus any text read from it — the vision layer's output
    contract. Everything downstream (reconciliation, repository) speaks in
    these, never in ultralytics/pytesseract types."""

    label: str                              # Detector class name.
    text: str                               # OCR'd text (may be "").
    detection_confidence: float             # 0-1 from the detector.
    ocr_confidence: float                   # 0-1 from OCR (0 if no text).
    box: tuple[int, int, int, int]          # Source bounding box (x1,y1,x2,y2).


class VisionPipeline:
    def __init__(self, detector: Detector, ocr: OcrReader) -> None:
        # Dependencies are injected rather than constructed here, so tests can
        # pass fakes and production can pass real, configured instances.
        self._detector = detector
        self._ocr = ocr

    def process(self, image_bytes: bytes) -> list[Observation]:
        """Decode image bytes, detect objects, and OCR each detected region."""
        # Imported lazily: cv2/numpy are heavy and only needed at call time.
        import cv2
        import numpy as np

        # Decode the uploaded bytes into an OpenCV BGR array. imdecode returns
        # None for anything that isn't a valid image — we treat that as "no
        # observations" rather than raising, so a bad upload can't crash a
        # request.
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return []

        observations: list[Observation] = []
        for det in self._detector.detect(image):
            x1, y1, x2, y2 = det.box
            # Clamp coordinates to >=0 before slicing so a box that runs off
            # the image edge can't produce a negative index / empty-but-wrong
            # crop.
            crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            # A zero-area crop (degenerate box) would make Tesseract unhappy;
            # skip OCR and record empty text in that case.
            ocr = self._ocr.read(crop) if crop.size else OcrResult("", 0.0)
            observations.append(
                Observation(
                    label=det.label,
                    text=ocr.text,
                    detection_confidence=det.confidence,
                    ocr_confidence=ocr.confidence,
                    box=det.box,
                )
            )
        return observations
