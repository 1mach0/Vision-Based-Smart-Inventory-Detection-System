"""Inventory endpoints — the read path.

Two simple GETs the UI polls: current stock, and the queue of low-confidence
changes waiting for a human. Both are thin projections of repository results
into JSON-friendly dicts; no business logic lives here.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from ..deps import get_repository
from ..persistence.repository import InventoryRepository

router = APIRouter(prefix="/inventory", tags=["inventory"])


@router.get("/products")
def list_products(repo: InventoryRepository = Depends(get_repository)):
    """Current inventory: one entry per SKU with its live quantity."""
    return [
        {"sku": p.sku, "name": p.name, "quantity": p.quantity}
        for p in repo.list_products()
    ]


@router.get("/review")
def list_pending_review(repo: InventoryRepository = Depends(get_repository)):
    """Low-confidence changes awaiting human review (stock not yet touched)."""
    return [
        {
            "id": c.id,
            "observation_id": c.observation_id,
            "delta": c.delta,
            "disposition": c.disposition,
        }
        for c in repo.pending_review()
    ]
