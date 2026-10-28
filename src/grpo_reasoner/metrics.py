"""Pure evaluation metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of records whose ``correct`` field is truthy."""
    if not records:
        return 0.0
    return sum(bool(record.get("correct")) for record in records) / len(records)


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, int | float]:
    """Return total/correct/accuracy for a set of per-example results."""
    total = len(records)
    correct = sum(bool(record.get("correct")) for record in records)
    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
