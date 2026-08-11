"""Market memory and historical analogues.

Persists a compact vector describing each market state and, when the present
resembles the past, reports what happened next - as a **distribution with its
sample size**, never as a prediction.

Similarity is Euclidean distance over standardised features, which keeps one
large-scale feature from dominating. Only states with a fully-elapsed forward
horizon are eligible, so an analogue can never be built from an outcome that has
not happened yet.
"""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: The vector that describes a market state. Kept short and scale-free.
STATE_FEATURES: tuple[str, ...] = (
    "atr_pct",
    "rsi",
    "ma_stack",
    "trend_quality",
    "range_position",
    "regime_direction",
    "volatility_ratio",
    "relative_volume",
    "funding_rate",
    "premium_discount",
    "order_flow_score",
    "htf_bias",
)


@dataclass(slots=True)
class MarketState:
    symbol: str
    ts: int
    timeframe: str
    features: dict[str, float]
    regime: str = "UNKNOWN"
    #: Forward return over the horizon, filled in once the horizon has elapsed.
    forward_return: float | None = None
    forward_max_up: float | None = None
    forward_max_down: float | None = None
    horizon_bars: int = 0

    @property
    def resolved(self) -> bool:
        return self.forward_return is not None

    def vector(self, names: Sequence[str] = STATE_FEATURES) -> list[float]:
        return [float(self.features.get(name, 0.0) or 0.0) for name in names]

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "ts": self.ts,
            "timeframe": self.timeframe,
            "regime": self.regime,
            "features": self.features,
            "forward_return": self.forward_return,
            "forward_max_up": self.forward_max_up,
            "forward_max_down": self.forward_max_down,
            "horizon_bars": self.horizon_bars,
        }


@dataclass(slots=True)
class Analogue:
    state: MarketState
    distance: float
    similarity: float          # 0..1

    def as_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.state.symbol,
            "ts": self.state.ts,
            "regime": self.state.regime,
            "similarity": round(self.similarity, 4),
            "forward_return": self.state.forward_return,
            "forward_max_up": self.state.forward_max_up,
            "forward_max_down": self.state.forward_max_down,
        }


@dataclass(slots=True)
class AnalogueReport:
    matches: list[Analogue] = field(default_factory=list)
    sample: int = 0
    mean_return: float = 0.0
    median_return: float = 0.0
    positive_share: float = 0.0
    mean_max_up: float = 0.0
    mean_max_down: float = 0.0
    dispersion: float = 0.0
    confidence: str = "NONE"        # NONE | WEAK | MODERATE | STRONG

    @property
    def informative(self) -> bool:
        return self.sample >= 10 and self.confidence != "NONE"

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample": self.sample,
            "confidence": self.confidence,
            "informative": self.informative,
            "mean_return": round(self.mean_return, 5),
            "median_return": round(self.median_return, 5),
            "positive_share": round(self.positive_share, 4),
            "mean_max_up": round(self.mean_max_up, 5),
            "mean_max_down": round(self.mean_max_down, 5),
            "dispersion": round(self.dispersion, 5),
            "matches": [m.as_dict() for m in self.matches[:8]],
        }

    def summary(self) -> str:
        if not self.sample:
            return "no comparable historical states found"
        return (
            f"{self.sample} similar historical state(s) ({self.confidence}): "
            f"median forward return {self.median_return:+.2%}, "
            f"{self.positive_share:.0%} positive, "
            f"dispersion {self.dispersion:.2%}. "
            "History describes what happened before, not what will happen."
        )


class MarketMemory:
    def __init__(
        self,
        repositories: Any = None,
        capacity: int = 20000,
        feature_names: Sequence[str] = STATE_FEATURES,
    ) -> None:
        self.repositories = repositories
        self.capacity = capacity
        self.feature_names = list(feature_names)
        self._states: list[MarketState] = []
        self._means: list[float] | None = None
        self._stds: list[float] | None = None

    # -- recording --------------------------------------------------------

    def remember(self, state: MarketState) -> None:
        self._states.append(state)
        if len(self._states) > self.capacity:
            self._states = self._states[-self.capacity :]
        self._means = None       # invalidate standardisation
        self._persist(state)

    def resolve(
        self,
        symbol: str,
        ts: int,
        forward_return: float,
        forward_max_up: float,
        forward_max_down: float,
    ) -> bool:
        """Fill in what actually happened once the horizon has elapsed."""

        for state in reversed(self._states):
            if state.symbol == symbol and state.ts == ts:
                state.forward_return = forward_return
                state.forward_max_up = forward_max_up
                state.forward_max_down = forward_max_down
                self._persist(state)
                return True
        return False

    @property
    def resolved_states(self) -> list[MarketState]:
        return [s for s in self._states if s.resolved]

    # -- standardisation ----------------------------------------------------

    def _fit_scaler(self) -> None:
        states = self.resolved_states
        if len(states) < 5:
            self._means = [0.0] * len(self.feature_names)
            self._stds = [1.0] * len(self.feature_names)
            return
        vectors = [s.vector(self.feature_names) for s in states]
        self._means = [
            statistics.fmean(column) for column in zip(*vectors)
        ]
        self._stds = [
            max(statistics.pstdev(column), 1e-6) for column in zip(*vectors)
        ]

    def _standardise(self, vector: Sequence[float]) -> list[float]:
        if self._means is None or self._stds is None:
            self._fit_scaler()
        assert self._means is not None and self._stds is not None
        return [
            (value - mean) / std
            for value, mean, std in zip(vector, self._means, self._stds)
        ]

    # -- retrieval ------------------------------------------------------------

    def find_analogues(
        self,
        features: Mapping[str, float],
        top_k: int = 25,
        regime: str | None = None,
        max_distance: float = 3.5,
        exclude_symbol_ts: tuple[str, int] | None = None,
    ) -> AnalogueReport:
        """Find historical states resembling ``features``."""

        report = AnalogueReport()
        candidates = self.resolved_states
        if regime:
            filtered = [s for s in candidates if s.regime == regime]
            # Fall back to all regimes rather than reporting nothing.
            candidates = filtered if len(filtered) >= 10 else candidates
        if exclude_symbol_ts:
            candidates = [
                s
                for s in candidates
                if not (s.symbol == exclude_symbol_ts[0] and s.ts == exclude_symbol_ts[1])
            ]
        if len(candidates) < 5:
            return report

        self._fit_scaler()
        query = self._standardise(
            [float(features.get(name, 0.0) or 0.0) for name in self.feature_names]
        )

        scored: list[Analogue] = []
        for state in candidates:
            vector = self._standardise(state.vector(self.feature_names))
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(query, vector)))
            if distance > max_distance:
                continue
            similarity = math.exp(-distance / 2.0)
            scored.append(Analogue(state=state, distance=distance, similarity=similarity))

        scored.sort(key=lambda a: a.distance)
        matches = scored[:top_k]
        if not matches:
            return report

        returns = [m.state.forward_return or 0.0 for m in matches]
        report.matches = matches
        report.sample = len(matches)
        report.mean_return = statistics.fmean(returns)
        report.median_return = statistics.median(returns)
        report.positive_share = sum(1 for r in returns if r > 0) / len(returns)
        report.dispersion = statistics.pstdev(returns) if len(returns) > 1 else 0.0
        report.mean_max_up = statistics.fmean(
            [m.state.forward_max_up or 0.0 for m in matches]
        )
        report.mean_max_down = statistics.fmean(
            [m.state.forward_max_down or 0.0 for m in matches]
        )

        # Confidence reflects sample size *and* how tight the outcomes were.
        # A large sample with huge dispersion is not informative.
        mean_similarity = statistics.fmean([m.similarity for m in matches])
        if report.sample >= 25 and mean_similarity > 0.5 and report.dispersion < 0.05:
            report.confidence = "STRONG"
        elif report.sample >= 15 and mean_similarity > 0.35:
            report.confidence = "MODERATE"
        elif report.sample >= 8:
            report.confidence = "WEAK"
        else:
            report.confidence = "NONE"
        return report

    def statistics(self) -> dict[str, Any]:
        resolved = self.resolved_states
        return {
            "states": len(self._states),
            "resolved": len(resolved),
            "pending": len(self._states) - len(resolved),
            "symbols": len({s.symbol for s in self._states}),
            "capacity": self.capacity,
        }

    def _persist(self, state: MarketState) -> None:
        if self.repositories is None:
            return
        try:
            self.repositories.memory.save(state.as_dict())
        except Exception:  # noqa: BLE001
            pass

    def load(self, limit: int = 20000) -> int:
        if self.repositories is None:
            return 0
        try:
            rows = self.repositories.memory.recent(limit)
        except Exception:  # noqa: BLE001
            return 0
        for row in rows:
            self._states.append(
                MarketState(
                    symbol=str(row["symbol"]),
                    ts=int(row["ts"]),
                    timeframe=str(row.get("timeframe", "")),
                    features=row.get("features") or {},
                    regime=str(row.get("regime", "UNKNOWN")),
                    forward_return=row.get("forward_return"),
                    forward_max_up=row.get("forward_max_up"),
                    forward_max_down=row.get("forward_max_down"),
                    horizon_bars=int(row.get("horizon_bars", 0) or 0),
                )
            )
        self._means = None
        return len(rows)


__all__ = [
    "MarketMemory",
    "MarketState",
    "Analogue",
    "AnalogueReport",
    "STATE_FEATURES",
]
