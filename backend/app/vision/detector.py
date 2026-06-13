"""YOLO object detection.

Thin wrapper around ultralytics YOLO. Two reasons it's isolated behind this
class rather than called directly from the pipeline:

  1. Swap/mock-ability — the pipeline depends on this small interface
     (``detect`` returning ``Detection``s), so tests inject a fake and a future
     model (a different YOLO version, or an entirely different detector) drops
     in without touching anything downstream.
  2. Lazy loading — ``ultralytics`` (and torch behind it) is a heavy import,
     and the weights file may be large or absent. Importing this module must
     stay cheap and side-effect-free, so the actual import happens inside
     ``load()``, not at module top. That's why the app can boot, and the test
     suite can run, on a machine with no weights installed.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected object. A plain, framework-agnostic value object so nothing
    downstream depends on ultralytics' result types."""

    label: str                              # Class name, e.g. "box", "bottle".
    confidence: float                       # Detector confidence in [0, 1].
    box: tuple[int, int, int, int]          # Pixel coords (x1, y1, x2, y2).


class Detector:
    """Loads a YOLO model once and runs detection on images."""

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._model = None  # Populated on first load(); acts as a "loaded?" flag.

    def load(self) -> None:
        """Load the YOLO weights into memory. Idempotent — safe to call before
        every ``detect`` — so callers don't have to track load state."""
        if self._model is not None:
            return
        # Imported here, not at top of file, to keep module import light and
        # avoid requiring torch/ultralytics just to import the package.
        from ultralytics import YOLO

        self._model = YOLO(self._model_path)

    def detect(self, image) -> list[Detection]:
        """Run detection on a single image (a numpy BGR array from OpenCV).

        ultralytics returns a list of Results (one per image); we passed one
        image, so we iterate and flatten every box into our own Detection type.
        """
        self.load()
        detections: list[Detection] = []
        # verbose=False silences ultralytics' per-frame console logging.
        for result in self._model(image, verbose=False):
            names = result.names  # {class_index: class_name}
            for box in result.boxes:
                # box.xyxy[0] is a tensor [x1, y1, x2, y2]; make it plain ints.
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(
                    Detection(
                        label=names[int(box.cls[0])],
                        confidence=float(box.conf[0]),
                        box=(x1, y1, x2, y2),
                    )
                )
        return detections
