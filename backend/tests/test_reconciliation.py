"""Domain reconciliation tests."""
from app.domain.reconciliation import Disposition, reconcile
from app.vision.pipeline import Observation


def _obs(label, text, det, ocr):
    return Observation(label, text, det, ocr, (0, 0, 1, 1))


def test_high_confidence_applies():
    (change,) = reconcile([_obs("a", "SKU", 0.9, 0.9)], review_threshold=0.5)
    assert change.disposition is Disposition.APPLY


def test_low_detection_routes_to_review():
    (change,) = reconcile([_obs("a", "", 0.2, 0.0)], review_threshold=0.5)
    assert change.disposition is Disposition.REVIEW


def test_confident_detection_but_weak_ocr_routes_to_review():
    # Text present, so OCR confidence counts and drags the score below threshold.
    (change,) = reconcile([_obs("a", "SKU", 0.95, 0.1)], review_threshold=0.5)
    assert change.disposition is Disposition.REVIEW


def test_threshold_is_exclusive():
    (change,) = reconcile([_obs("a", "", 0.5, 0.0)], review_threshold=0.5)
    assert change.disposition is Disposition.REVIEW
