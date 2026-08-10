"""Feature vectorisation.

Features arrive as ``{name: value}`` dicts produced by
:meth:`app.scanner.scanner.SymbolAnalysis.features`.  The model stores the
exact feature names it was trained on, and prediction realigns by name: a
feature that disappears becomes its training-time mean, and a new feature that
the model has never seen is ignored.  That makes the model robust to adding
timeframes or indicators without silently shifting every column by one - a
failure mode that produces confident nonsense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass(slots=True)
class FeatureSpec:
    """Feature names plus the standardisation statistics used at train time."""

    names: list[str] = field(default_factory=list)
    means: list[float] = field(default_factory=list)
    stds: list[float] = field(default_factory=list)

    @classmethod
    def fit(cls, rows: Sequence[Mapping[str, float]], names: Sequence[str] | None = None) -> "FeatureSpec":
        if names is None:
            collected: set[str] = set()
            for row in rows:
                collected.update(row.keys())
            names = sorted(collected)
        names = list(names)

        means: list[float] = []
        stds: list[float] = []
        count = max(len(rows), 1)
        for name in names:
            values = [_finite(row.get(name, 0.0)) for row in rows]
            mean = sum(values) / count if values else 0.0
            variance = (
                sum((v - mean) ** 2 for v in values) / count if len(values) > 1 else 0.0
            )
            std = math.sqrt(variance)
            means.append(mean)
            # A zero-variance column would divide by zero; 1.0 keeps it at 0.
            stds.append(std if std > 1e-9 else 1.0)
        return cls(names=names, means=means, stds=stds)

    def transform(self, row: Mapping[str, float]) -> list[float]:
        out: list[float] = []
        for index, name in enumerate(self.names):
            raw = row.get(name)
            value = self.means[index] if raw is None else _finite(raw)
            out.append((value - self.means[index]) / self.stds[index])
        return out

    def transform_many(self, rows: Iterable[Mapping[str, float]]) -> list[list[float]]:
        return [self.transform(row) for row in rows]

    def as_dict(self) -> dict[str, list]:
        return {"names": self.names, "means": self.means, "stds": self.stds}

    @classmethod
    def from_dict(cls, data: Mapping[str, Sequence]) -> "FeatureSpec":
        return cls(
            names=list(data.get("names", [])),
            means=[float(v) for v in data.get("means", [])],
            stds=[float(v) if float(v) > 1e-9 else 1.0 for v in data.get("stds", [])],
        )

    def __len__(self) -> int:
        return len(self.names)


def vectorise(row: Mapping[str, float], spec: FeatureSpec) -> list[float]:
    return spec.transform(row)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    # Clip absurd magnitudes so one bad print cannot dominate standardisation.
    return max(-1e9, min(1e9, number))


__all__ = ["FeatureSpec", "vectorise"]
