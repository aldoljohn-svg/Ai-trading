"""Probability calibration and reliability measurement.

A raw classifier score is not a probability.  Acting on an uncalibrated "80%"
as though it were an 80% chance is one of the more expensive mistakes available
to a trading system, because position sizing and the confidence gate both treat
that number as meaningful.

Two calibrators are provided:

:class:`PlattCalibrator`
    One-dimensional logistic regression on the raw score.  Robust, needs few
    samples, assumes a sigmoid distortion.
:class:`IsotonicCalibrator`
    Pool-adjacent-violators (PAVA) isotonic regression.  Non-parametric, fits
    any monotone distortion, needs more samples.

:func:`fit_calibrator` picks between them based on the sample count, and both
are measured with the Brier score and expected calibration error so a
calibration that made things *worse* is rejected rather than shipped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _clip01(value: float, eps: float = 1e-6) -> float:
    return min(max(value, eps), 1.0 - eps)


@dataclass(slots=True)
class PlattCalibrator:
    """``p = sigmoid(a * score + b)`` fitted by Newton / gradient descent."""

    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
        iterations: int = 200,
        learning_rate: float = 0.5,
    ) -> "PlattCalibrator":
        if len(scores) < 20 or len(scores) != len(labels):
            self.fitted = False
            return self

        # Platt's prior correction keeps the fit from saturating on small,
        # imbalanced samples.
        positives = sum(1 for y in labels if y == 1)
        negatives = len(labels) - positives
        if positives == 0 or negatives == 0:
            self.fitted = False
            return self
        high_target = (positives + 1.0) / (positives + 2.0)
        low_target = 1.0 / (negatives + 2.0)
        targets = [high_target if y == 1 else low_target for y in labels]

        a, b = 1.0, 0.0
        n = len(scores)
        for _ in range(iterations):
            grad_a = 0.0
            grad_b = 0.0
            for score, target in zip(scores, targets):
                p = _sigmoid(a * score + b)
                error = p - target
                grad_a += error * score
                grad_b += error
            a -= learning_rate * grad_a / n
            b -= learning_rate * grad_b / n

        self.a, self.b = a, b
        self.fitted = True
        return self

    def transform(self, score: float) -> float:
        if not self.fitted:
            return _clip01(score)
        return _clip01(_sigmoid(self.a * score + self.b))

    def as_dict(self) -> dict[str, Any]:
        return {"kind": "platt", "a": self.a, "b": self.b, "fitted": self.fitted}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlattCalibrator":
        return cls(
            a=float(data.get("a", 1.0)),
            b=float(data.get("b", 0.0)),
            fitted=bool(data.get("fitted", False)),
        )


@dataclass(slots=True)
class IsotonicCalibrator:
    """Monotone step function fitted with pool-adjacent-violators."""

    thresholds: list[float] = field(default_factory=list)
    values: list[float] = field(default_factory=list)
    fitted: bool = False

    def fit(self, scores: Sequence[float], labels: Sequence[int]) -> "IsotonicCalibrator":
        if len(scores) < 50 or len(scores) != len(labels):
            self.fitted = False
            return self

        pairs = sorted(zip(scores, labels), key=lambda item: item[0])
        # Each block holds (sum of labels, count, right-most score).
        blocks: list[list[float]] = []
        for score, label in pairs:
            blocks.append([float(label), 1.0, float(score)])
            # Merge while the sequence is not non-decreasing.
            while len(blocks) >= 2 and (blocks[-2][0] / blocks[-2][1]) > (
                blocks[-1][0] / blocks[-1][1]
            ):
                last = blocks.pop()
                previous = blocks.pop()
                blocks.append(
                    [previous[0] + last[0], previous[1] + last[1], last[2]]
                )

        self.thresholds = [block[2] for block in blocks]
        self.values = [_clip01(block[0] / block[1]) for block in blocks]
        self.fitted = bool(self.thresholds)
        return self

    def transform(self, score: float) -> float:
        if not self.fitted:
            return _clip01(score)
        # Binary search for the first threshold >= score.
        low, high = 0, len(self.thresholds) - 1
        if score <= self.thresholds[0]:
            return self.values[0]
        if score >= self.thresholds[-1]:
            return self.values[-1]
        while low < high:
            mid = (low + high) // 2
            if self.thresholds[mid] < score:
                low = mid + 1
            else:
                high = mid
        return self.values[low]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "isotonic",
            "thresholds": self.thresholds,
            "values": self.values,
            "fitted": self.fitted,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IsotonicCalibrator":
        return cls(
            thresholds=[float(v) for v in data.get("thresholds", [])],
            values=[float(v) for v in data.get("values", [])],
            fitted=bool(data.get("fitted", False)),
        )


Calibrator = PlattCalibrator | IsotonicCalibrator


def calibrator_from_dict(data: Mapping[str, Any] | None) -> Calibrator | None:
    if not data:
        return None
    if data.get("kind") == "isotonic":
        return IsotonicCalibrator.from_dict(data)
    return PlattCalibrator.from_dict(data)


def fit_calibrator(
    scores: Sequence[float], labels: Sequence[int]
) -> tuple[Calibrator | None, dict[str, float]]:
    """Fit, measure, and only keep a calibrator that actually improves things."""

    if len(scores) < 20:
        return None, {"reason": 0.0, "samples": float(len(scores))}

    baseline_brier = brier_score(scores, labels)
    baseline_ece = expected_calibration_error(scores, labels)

    candidates: list[Calibrator] = [PlattCalibrator().fit(scores, labels)]
    if len(scores) >= 200:
        candidates.append(IsotonicCalibrator().fit(scores, labels))

    best: Calibrator | None = None
    best_brier = baseline_brier
    metrics = {
        "baseline_brier": round(baseline_brier, 5),
        "baseline_ece": round(baseline_ece, 5),
        "samples": float(len(scores)),
    }

    for candidate in candidates:
        if not candidate.fitted:
            continue
        transformed = [candidate.transform(s) for s in scores]
        brier = brier_score(transformed, labels)
        key = "platt" if isinstance(candidate, PlattCalibrator) else "isotonic"
        metrics[f"{key}_brier"] = round(brier, 5)
        metrics[f"{key}_ece"] = round(expected_calibration_error(transformed, labels), 5)
        if brier < best_brier:
            best_brier = brier
            best = candidate

    metrics["calibrated_brier"] = round(best_brier, 5)
    return best, metrics


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def brier_score(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    """Mean squared error of the probabilities.  Lower is better."""

    if not probabilities:
        return 1.0
    total = sum((p - y) ** 2 for p, y in zip(probabilities, labels))
    return total / len(probabilities)


def expected_calibration_error(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
) -> float:
    """Average gap between predicted confidence and observed frequency."""

    if not probabilities:
        return 1.0
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels):
        index = min(int(probability * bins), bins - 1)
        buckets[index].append((probability, label))

    total = len(probabilities)
    error = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        mean_probability = sum(p for p, _ in bucket) / len(bucket)
        observed = sum(y for _, y in bucket) / len(bucket)
        error += (len(bucket) / total) * abs(mean_probability - observed)
    return error


def reliability_table(
    probabilities: Sequence[float], labels: Sequence[int], bins: int = 10
) -> list[dict[str, float]]:
    """Per-bucket predicted-vs-observed table, for the ``/ai`` command."""

    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in zip(probabilities, labels):
        index = min(int(probability * bins), bins - 1)
        buckets[index].append((probability, label))

    table: list[dict[str, float]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        table.append(
            {
                "bucket_low": round(index / bins, 3),
                "bucket_high": round((index + 1) / bins, 3),
                "count": float(len(bucket)),
                "predicted": round(sum(p for p, _ in bucket) / len(bucket), 4),
                "observed": round(sum(y for _, y in bucket) / len(bucket), 4),
            }
        )
    return table


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-min(x, 60.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(x, -60.0))
    return z / (1.0 + z)


__all__ = [
    "PlattCalibrator",
    "IsotonicCalibrator",
    "Calibrator",
    "fit_calibrator",
    "calibrator_from_dict",
    "brier_score",
    "expected_calibration_error",
    "reliability_table",
]
