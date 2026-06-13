"""FastAPI dependency providers.

These functions are how endpoints get their collaborators (the vision pipeline
and the repository). Declaring them as dependencies — rather than constructing
objects inside endpoints — is what makes the API testable: a test calls
``app.dependency_overrides[get_pipeline] = ...`` to swap in a fake, with no
change to the endpoint code.
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from .config import settings
from .persistence.database import get_session
from .persistence.repository import InventoryRepository
from .vision.detector import Detector
from .vision.ocr import OcrReader
from .vision.pipeline import VisionPipeline


@lru_cache
def get_pipeline() -> VisionPipeline:
    """Return the process-wide vision pipeline.

    ``lru_cache`` makes it a singleton: the Detector/OcrReader are created once
    and reused across requests, so the (expensive) YOLO model is loaded a
    single time — on the first inference — and stays warm. The model isn't
    loaded here at construction, only when ``detect`` is first called.
    """
    return VisionPipeline(
        Detector(settings.yolo_model_path),
        OcrReader(settings.tesseract_cmd),
    )


def get_repository(session: Session = Depends(get_session)) -> InventoryRepository:
    """Build a repository around this request's database session.

    ``session`` is itself injected by FastAPI via ``get_session``, so the
    request's transaction lifecycle is handled for us.
    """
    return InventoryRepository(session)
