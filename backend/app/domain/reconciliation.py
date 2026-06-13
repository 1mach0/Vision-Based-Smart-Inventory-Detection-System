"""Confidence-aware reconciliation — the core business rule.

The vision layer is probabilistic: every detection carries a confidence, and
OCR carries its own. Blindly trusting those numbers would let a bad frame
corrupt inventory. Reconciliation is the gate between "what the model thinks"
and "what we record as fact": it converts each observation into a
*disposition* — APPLY (confident enough to update stock) or REVIEW (uncertain,
hand to a human).

Deliberately dependency-free (only the standard library and the Observation
dataclass): it's the most important logic to keep pure and trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..vision.pipeline import Observation


class Disposition(str, Enum):
    """What to do with an observation.

    Subclassing ``str`` makes the enum JSON-serialisable and directly
    comparable to the string stored in the database ("apply"/"review").
    """

    APPLY = "apply"          # High enough confidence to update inventory.
    REVIEW = "review"        # Uncertain: route to a human.


@dataclass(frozen=True)
class ReconciledChange:
    """Pairs an observation with the decision made about it. Frozen because a
    decision, once made, shouldn't be mutated in place."""

    observation: Observation
    disposition: Disposition


def _score(obs: Observation) -> float:
    """Collapse an observation's two confidences into one decision score.

    If OCR read text (a SKU), we depend on *both* the detection and the OCR
    being right, so we take the weaker of the two (a confident box with an
    unreadable label is still risky). If no text was read, only the detection
    confidence is meaningful.
    """
    if obs.text.strip():
        return min(obs.detection_confidence, obs.ocr_confidence)
    return obs.detection_confidence


def reconcile(
    observations: list[Observation],
    review_threshold: float,
) -> list[ReconciledChange]:
    """Assign every observation a disposition based on its score.

    Strictly-greater-than comparison makes the threshold *exclusive*: a score
    exactly equal to the threshold is treated as not-confident-enough and sent
    to review (the cautious choice).
    """
    changes: list[ReconciledChange] = []
    for obs in observations:
        disposition = (
            Disposition.APPLY if _score(obs) > review_threshold else Disposition.REVIEW
        )
        changes.append(ReconciledChange(observation=obs, disposition=disposition))
    return changes
