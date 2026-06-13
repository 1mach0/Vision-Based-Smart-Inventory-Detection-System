"""ORM models — the relational schema.

Three tables capture the whole pipeline's state:

    products            current stock, one row per SKU
    observations        every raw detection+OCR result the vision layer emitted
    inventory_changes   the decision made about each observation (apply/review)
                        and the stock delta it produced

Keeping observations and inventory_changes as separate append-only tables
means we never lose the raw evidence: even a low-confidence detection that
never touched stock is stored, which is what makes the human-review workflow
and later auditing possible.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class Product(Base):
    """A stocked item, keyed by SKU. ``quantity`` is the current count."""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Unique + indexed: we look products up by SKU on every applied change,
    # and never want two rows for the same SKU.
    sku: Mapped[str] = mapped_column(String, unique=True, index=True)
    name: Mapped[str] = mapped_column(String)
    quantity: Mapped[int] = mapped_column(default=0)


class Observation(Base):
    """One detection+OCR result from the vision pipeline.

    Stored verbatim, including both confidence scores, before any business
    decision is applied. This is the immutable record of "what the camera
    saw".
    """

    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(primary_key=True)
    label: Mapped[str] = mapped_column(String)                 # YOLO class name
    text: Mapped[str] = mapped_column(String, default="")      # OCR'd text (SKU)
    detection_confidence: Mapped[float] = mapped_column(Float)
    ocr_confidence: Mapped[float] = mapped_column(Float)
    # server_default=func.now() -> the database stamps the time on INSERT, so
    # it's correct even for rows created outside the app.
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Convenience navigation: obs.changes -> the InventoryChange(s) derived
    # from this observation. Populated by SQLAlchemy via the FK below.
    changes: Mapped[list["InventoryChange"]] = relationship(back_populates="observation")


class InventoryChange(Base):
    """The decision derived from an observation.

    ``disposition`` is "apply" (stock was updated) or "review" (held for a
    human). ``product_id`` is null for review items, because we deliberately
    don't touch a product until a human confirms the uncertain detection.
    ``delta`` is the change in count the observation represents (currently
    always +1 — one detected item = one unit).
    """

    __tablename__ = "inventory_changes"

    id: Mapped[int] = mapped_column(primary_key=True)
    observation_id: Mapped[int] = mapped_column(ForeignKey("observations.id"))
    # Nullable FK: review items aren't tied to a product yet.
    product_id: Mapped[int | None] = mapped_column(ForeignKey("products.id"), nullable=True)
    delta: Mapped[int] = mapped_column(default=0)
    disposition: Mapped[str] = mapped_column(String)  # "apply" | "review"
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    observation: Mapped["Observation"] = relationship(back_populates="changes")
