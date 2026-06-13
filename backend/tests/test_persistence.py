"""Persistence / repository tests (in-memory SQLite via conftest)."""
from app.domain.reconciliation import Disposition
from app.persistence.repository import InventoryRepository
from app.vision.pipeline import Observation


def _obs(label, text):
    return Observation(label, text, 0.9, 0.9, (0, 0, 1, 1))


def test_apply_creates_and_increments_product(db_session):
    repo = InventoryRepository(db_session)
    repo.record(_obs("widget", "SKU-1"), Disposition.APPLY)
    repo.record(_obs("widget", "SKU-1"), Disposition.APPLY)
    repo.commit()

    products = repo.list_products()
    assert len(products) == 1
    assert products[0].sku == "SKU-1"
    assert products[0].quantity == 2


def test_review_records_change_without_touching_stock(db_session):
    repo = InventoryRepository(db_session)
    change = repo.record(_obs("gadget", ""), Disposition.REVIEW)
    repo.commit()

    assert change.product_id is None
    assert repo.list_products() == []
    assert len(repo.pending_review()) == 1


def test_sku_falls_back_to_label_when_text_missing(db_session):
    repo = InventoryRepository(db_session)
    repo.record(_obs("bolt", ""), Disposition.APPLY)
    repo.commit()

    products = repo.list_products()
    assert products[0].sku == "bolt"
