"""Vision-pipeline integration test with fake detector/OCR.

Exercises decode → detect → crop → OCR wiring without YOLO weights or
Tesseract. Skips if OpenCV/NumPy aren't installed.
"""
import pytest

cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from app.vision.detector import Detection
from app.vision.ocr import OcrResult
from app.vision.pipeline import VisionPipeline


class _FakeDetector:
    def detect(self, image):
        return [Detection("widget", 0.9, (0, 0, 5, 5))]


class _FakeOcr:
    def read(self, image):
        return OcrResult("SKU-1", 0.8)


def test_pipeline_wires_detection_and_ocr():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", image)
    assert ok

    pipeline = VisionPipeline(_FakeDetector(), _FakeOcr())
    observations = pipeline.process(buf.tobytes())

    assert len(observations) == 1
    obs = observations[0]
    assert obs.label == "widget"
    assert obs.text == "SKU-1"
    assert obs.detection_confidence == 0.9
    assert obs.ocr_confidence == 0.8


def test_pipeline_returns_empty_on_undecodable_bytes():
    pipeline = VisionPipeline(_FakeDetector(), _FakeOcr())
    assert pipeline.process(b"not-an-image") == []
