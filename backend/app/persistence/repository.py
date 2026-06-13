"""Data access — the only place that reads/writes inventory tables.

The repository pattern keeps SQL/ORM calls out of the API and domain layers.
Endpoints call small, intention-revealing methods (``record``,
``list_products``, ``pending_review``) and never see a ``Session`` query,
which makes both layers easy to test and the storage engine easy to swap.

Write model: ``record`` is append-only for evidence (it always inserts an
Observation and an InventoryChange) and mutating only for stock (it bumps
Product.quantity only when the change is applied).
"""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..domain.reconciliation import Disposition
# The domain-level Observation (a dataclass from the vision layer) and the ORM
# Observation share a name; alias them so both are unambiguous below.
from ..vision.pipeline import Observation as VisionObservation
from .models import InventoryChange, Observation as ObservationRow, Product


class InventoryRepository:
    def __init__(self, session: Session) -> None:
        # One repository wraps one Session (one per request). It does not open
        # or close the session itself — that's the dependency's job.
        self._session = session

    def _upsert_product(self, sku: str, name: str) -> Product:
        """Return the product for ``sku``, creating it (quantity 0) if new.

        ``flush`` sends the INSERT so the new row gets its autoincrement id,
        which the caller needs to link the InventoryChange — but it does NOT
        commit, so the whole ``record`` call stays a single transaction.
        """
        product = self._session.scalar(select(Product).where(Product.sku == sku))
        if product is None:
            product = Product(sku=sku, name=name, quantity=0)
            self._session.add(product)
            self._session.flush()
        return product

    def record(self, obs: VisionObservation, disposition: Disposition) -> InventoryChange:
        """Persist one observation and the inventory change it produced.

        Always stores the raw observation. If the disposition is APPLY it also
        upserts the product and increments stock; if REVIEW it records the
        change with a null product and leaves stock untouched, so an uncertain
        detection is captured without silently altering counts.
        """
        # 1. Store the raw evidence. flush() assigns row.id for the FK below.
        row = ObservationRow(
            label=obs.label,
            text=obs.text,
            detection_confidence=obs.detection_confidence,
            ocr_confidence=obs.ocr_confidence,
        )
        self._session.add(row)
        self._session.flush()

        # 2. Apply to stock only when confident.
        product_id: int | None = None
        delta = 1  # One detected item == one unit. Central place to change later.
        if disposition is Disposition.APPLY:
            # Prefer the OCR'd SKU; fall back to the detector's class label when
            # no text was read, so an item is still tracked under *some* key.
            sku = obs.text.strip() or obs.label
            product = self._upsert_product(sku, obs.label)
            product.quantity += delta
            product_id = product.id

        # 3. Record the decision, linked to the observation (and product, if any).
        change = InventoryChange(
            observation_id=row.id,
            product_id=product_id,
            delta=delta,
            disposition=disposition.value,
        )
        self._session.add(change)
        self._session.flush()
        return change

    def commit(self) -> None:
        """Commit the current transaction. Called once per request after all
        observations from an image have been recorded, so a single image is
        one atomic unit of work."""
        self._session.commit()

    def list_products(self) -> list[Product]:
        """All products, ordered by SKU for a stable UI listing."""
        return list(self._session.scalars(select(Product).order_by(Product.sku)))

    def pending_review(self) -> list[InventoryChange]:
        """Changes awaiting a human decision (disposition == 'review')."""
        return list(
            self._session.scalars(
                select(InventoryChange).where(
                    InventoryChange.disposition == Disposition.REVIEW.value
                )
            )
        )
