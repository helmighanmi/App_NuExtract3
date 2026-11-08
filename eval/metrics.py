"""Tree-based extraction metrics.

The metric is a field-alignment score over the ground-truth tree:

* leaves are compared by name (objects) or index (arrays);
* strings use normalised indel similarity (difflib ratio);
* numbers use a small relative tolerance;
* booleans and nulls require exact equality;
* missing ground-truth leaves score 0;
* extra predicted leaves increase the denominator, i.e. they are penalised.

It is deliberately simple and explainable; for a research-grade metric, use the
NuMind benchmark's tree-alignment implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

MISSING = object()


@dataclass
class Score:
    score: float
    matched: float
    total: int
    extra: int = 0
    per_field: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "matched": round(self.matched, 4),
            "total": self.total,
            "extra": self.extra,
            "per_field": {key: round(value, 4) for key, value in self.per_field.items()},
        }


def count_leaves(value: Any) -> int:
    if isinstance(value, dict):
        return sum(count_leaves(item) for item in value.values()) if value else 1
    if isinstance(value, list):
        return sum(count_leaves(item) for item in value) if value else 1
    return 1


def _similarity(predicted: Any, truth: Any) -> float:
    if truth is MISSING or predicted is MISSING:
        return 0.0
    if truth is None or predicted is None:
        return 1.0 if truth is predicted else 0.0
    if isinstance(truth, bool) or isinstance(predicted, bool):
        return 1.0 if predicted is truth else 0.0
    if isinstance(truth, (int, float)) and isinstance(predicted, (int, float)):
        if isinstance(truth, bool) or isinstance(predicted, bool):
            return 0.0
        return (
            1.0 if math.isclose(float(predicted), float(truth), rel_tol=1e-3, abs_tol=1e-6) else 0.0
        )
    if isinstance(truth, str) and isinstance(predicted, str):
        left = " ".join(truth.lower().split())
        right = " ".join(predicted.lower().split())
        if left == right:
            return 1.0
        return SequenceMatcher(None, left, right).ratio()
    if type(truth) is not type(predicted):
        return 0.0
    return 1.0 if predicted == truth else 0.0


def _matched(predicted: Any, truth: Any) -> float:
    if truth is MISSING or predicted is MISSING:
        return 0.0
    if isinstance(truth, dict):
        if not isinstance(predicted, dict):
            return 0.0
        return sum(_matched(predicted.get(key, MISSING), value) for key, value in truth.items())
    if isinstance(truth, list):
        if not isinstance(predicted, list):
            return 0.0
        return sum(
            _matched(predicted[index] if index < len(predicted) else MISSING, value)
            for index, value in enumerate(truth)
        )
    return _similarity(predicted, truth)


def score_prediction(predicted: Any, truth: Any) -> Score:
    gt_leaves = count_leaves(truth)
    pred_leaves = count_leaves(predicted) if predicted is not MISSING else 0
    denominator = max(gt_leaves, pred_leaves, 1)
    matched = _matched(predicted, truth)

    per_field: dict[str, float] = {}
    if isinstance(truth, dict) and isinstance(predicted, dict):
        for key, value in truth.items():
            per_field[key] = score_prediction(predicted.get(key, MISSING), value).score

    return Score(
        score=matched / denominator,
        matched=matched,
        total=gt_leaves,
        extra=max(0, pred_leaves - gt_leaves),
        per_field=per_field,
    )
