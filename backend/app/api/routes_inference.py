"""Inference endpoint — the write path.

POST an image; the server runs the vision pipeline, reconciles the results by
confidence, persists both the raw observations and the resulting inventory
changes, and returns a summary. This endpoint is pure orchestration: each real
step lives in its own layer (pipeline / reconcile / repository), so the handler
reads as a short recipe.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, UploadFile

from ..config import settings
from ..deps import get_pipeline, get_repository
from ..domain.reconciliation import Disposition, reconcile
from ..persistence.repository import InventoryRepository
from ..vision.pipeline import VisionPipeline

# prefix -> every route here lives under /inference; tags -> groups them in
# the auto-generated /docs UI.
router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/observe")
async def observe(
    image: UploadFile,
    pipeline: VisionPipeline = Depends(get_pipeline),
    repo: InventoryRepository = Depends(get_repository),
):
    """Run the vision pipeline on an uploaded image and reconcile the results.

    Flow: read bytes -> detect+OCR -> decide apply/review per observation ->
    persist each -> commit once (so one image is one atomic transaction) ->
    return counts plus the individual changes.
    """
    contents = await image.read()
    observations = pipeline.process(contents)
    changes = reconcile(observations, settings.review_confidence_threshold)

    # Persist every observation and its change, then commit the batch together.
    recorded = [repo.record(c.observation, c.disposition) for c in changes]
    repo.commit()

    return {
        "observations": len(observations),
        "applied": sum(1 for c in changes if c.disposition is Disposition.APPLY),
        "review": sum(1 for c in changes if c.disposition is Disposition.REVIEW),
        "changes": [
            {
                "id": c.id,
                "observation_id": c.observation_id,
                "product_id": c.product_id,
                "delta": c.delta,
                "disposition": c.disposition,
            }
            for c in recorded
        ],
    }
